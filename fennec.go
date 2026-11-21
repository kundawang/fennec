// Package fennec provides intelligent image compression that dramatically reduces
// file size while preserving perceptual quality. It uses SSIM (Structural Similarity
// Index) guided optimization to find the sweet spot between size and quality.
//
// Fennec — Tiny Fox. Giant Ears. Hears what matters, drops what doesn't.
//
// Unlike traditional image processing libraries that apply fixed compression,
// Fennec analyzes each image and adapts its strategy:
//
//   - SSIM-guided quality search: binary search for minimum file size at target quality
//   - Perceptual color quantization: reduce colors where human eyes can't tell
//   - Content-aware downscaling: smart resize that preserves important detail
//   - Adaptive format selection: picks JPEG/PNG based on image characteristics
//   - EXIF orientation handling: auto-rotates images from camera sensors
//   - Batch processing: concurrent compression with worker pools
package fennec

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"io"
	"os"
)

// CompressFile compresses an image file and writes the result to dst.
// It reads EXIF orientation data and auto-rotates if opts.AutoOrient is true.
// The context can be used to cancel long-running operations.
func CompressFile(ctx context.Context, src, dst string, opts Options) (*Result, error) {
	if err := opts.Validate(); err != nil {
		return nil, err
	}

	if err := opts.reportProgress(ctx, StageAnalyzing, 0); err != nil {
		return nil, err
	}

	img, orient, fileSize, err := openWithOrientation(src)
	if err != nil {
		return nil, err
	}

	result, err := compressImageInternal(ctx, img, orient, opts)
	if err != nil {
		return nil, err
	}
	result.OriginalSize = fileSize
	result.computeStats()

	if err := opts.reportProgress(ctx, StageWriting, 0.9); err != nil {
		return nil, err
	}

	// Write the pre-computed compressed bytes directly.
	data := result.CompressedData
	if len(data) == 0 {
		data, err = encodeToBytes(result.Image, result.Format, result.JPEGQuality)
		if err != nil {
			return nil, err
		}
		result.CompressedData = data
		result.CompressedSize = int64(len(data))
		result.computeStats()
	}

	if err := os.WriteFile(dst, data, 0644); err != nil {
		return nil, fmt.Errorf("fennec: write %q: %w", dst, err)
	}

	if err := opts.reportProgress(ctx, StageWriting, 1.0); err != nil {
		return nil, err
	}

	return result, nil
}

// CompressImage compresses an already-decoded image.
// The context can be used to cancel long-running operations.
func CompressImage(ctx context.Context, img image.Image, opts Options) (*Result, error) {
	if err := opts.Validate(); err != nil {
		return nil, err
	}
	return compressImageInternal(ctx, img, OrientNormal, opts)
}

// Compress reads an image from r and returns the optimally compressed version.
// The context can be used to cancel long-running operations.
func Compress(ctx context.Context, r io.Reader, opts Options) (*Result, error) {
	if err := opts.Validate(); err != nil {
		return nil, err
	}
	img, _, err := image.Decode(r)
	if err != nil {
		return nil, fmt.Errorf("fennec: decode: %w", err)
	}
	return compressImageInternal(ctx, img, OrientNormal, opts)
}

// CompressBytes compresses image data from a byte slice and returns the result.
// This is the most common API for server-side use: receive bytes → compress → return bytes.
func CompressBytes(ctx context.Context, data []byte, opts Options) (*Result, error) {
	return Compress(ctx, bytes.NewReader(data), opts)
}

// compressImageInternal is the shared compression pipeline.
func compressImageInternal(ctx context.Context, img image.Image, orient Orientation, opts Options) (*Result, error) {
	if img == nil {
		return nil, ErrNilImage
	}
	bounds := img.Bounds()
	if bounds.Dx() <= 0 || bounds.Dy() <= 0 {
		return nil, ErrEmptyImage
	}

	result := &Result{OriginalDimensions: image.Pt(bounds.Dx(), bounds.Dy())}
	src := toNRGBA(img)

	// Validate regions against the decoded pixel grid before orientation or
	// resizing, then remap them into display (oriented) coordinates.
	rs, err := newRegionSet(opts.optionsRegionList(), bounds.Dx(), bounds.Dy(), orient,
		opts.AutoOrient)
	if err != nil {
		return nil, err
	}

	if opts.AutoOrient && orient > OrientNormal {
		src = ApplyOrientation(src, orient)
		result.OriginalDimensions = image.Pt(src.Bounds().Dx(), src.Bounds().Dy())
	}
	if err := opts.reportProgress(ctx, StageResizing, 0.1); err != nil {
		return nil, err
	}

	preResizeW, preResizeH := src.Bounds().Dx(), src.Bounds().Dy()
	if opts.MaxWidth > 0 || opts.MaxHeight > 0 {
		src = smartResize(src, opts.MaxWidth, opts.MaxHeight)
	}
	result.Image = src
	result.FinalDimensions = image.Pt(src.Bounds().Dx(), src.Bounds().Dy())

	// Regions were validated in source space; remap their coordinate base so
	// MaxWidth/MaxHeight resizing is accounted for.
	if rs != nil && (preResizeW != src.Bounds().Dx() || preResizeH != src.Bounds().Dy()) {
		rs = rs.remapBase(src.Bounds().Dx(), src.Bounds().Dy())
	}

	// Record an explicitly requested whole-image target; in target-size mode
	// it is enforced together with region thresholds.
	if opts.TargetSSIM > 0 {
		result.TargetSSIM = opts.TargetSSIM
	}

	if err := opts.reportProgress(ctx, StageCompressing, 0.2); err != nil {
		return nil, err
	}

	if opts.TargetSize > 0 {
		return handleTargetSizeMode(ctx, src, rs, opts, result)
	}
	return handleStandardMode(ctx, src, rs, opts, result)
}

// remapBase returns a copy of the region set whose base coordinate space is
// outW x outH. It is used after MaxWidth/MaxHeight resizing so regions follow
// the subject into the resized image.
func (rs *regionSet) remapBase(outW, outH int) *regionSet {
	if rs == nil || rs.baseW == outW && rs.baseH == outH {
		return rs
	}
	next := &regionSet{baseW: outW, baseH: outH, targets: make([]regionTarget, len(rs.targets))}
	for i, t := range rs.targets {
		next.targets[i] = regionTarget{
			request:  t.request,
			oriented: mapRectScale(t.oriented, rs.baseW, rs.baseH, outW, outH),
			minSSIM:  t.minSSIM,
		}
	}
	next.enforce = disjointEnforcement(next.targets)
	return next
}

func handleTargetSizeMode(ctx context.Context, src *image.NRGBA, rs *regionSet, opts Options, result *Result) (*Result, error) {
	sr, err := hitTargetSize(ctx, src, opts.TargetSize, rs, opts)
	if err != nil {
		return nil, fmt.Errorf("fennec: target-size compression: %w", err)
	}

	result.CompressedData = sr.data
	result.Format = sr.format
	result.JPEGQuality = sr.quality
	result.SSIM = sr.ssim
	result.FinalDimensions = image.Pt(sr.finalW, sr.finalH)
	if sr.img != nil {
		result.Image = sr.img
	}
	result.RegionResults = sr.regions
	result.Regions = sr.regions
	result.RegionSSIM = regionSSIMs(sr.regions)
	result.CompressedSize = int64(len(sr.data))
	result.computeStats()
	return result, nil
}

func handleStandardMode(ctx context.Context, src *image.NRGBA, rs *regionSet, opts Options, result *Result) (*Result, error) {
	if opts.Format == Auto {
		opts.Format = analyzeFormat(src)
	}
	result.Format = opts.Format

	if err := opts.reportProgress(ctx, StageOptimizing, 0.3); err != nil {
		return nil, err
	}

	var compressed encodingBuffer
	switch opts.Format {
	case PNG:
		if err := compressPNG(src, &compressed, opts); err != nil {
			return nil, fmt.Errorf("fennec: PNG compression: %w", err)
		}
		result.SSIM = 1.0
		if rs.len() > 0 {
			// PNG is lossless; still report measured per-region quality so
			// callers get the same shape of result as with JPEG.
			_, regions, _ := evaluateRegions(ctx, rs, 0, src, src)
			result.RegionResults = regions
			result.Regions = regions
			result.RegionSSIM = regionSSIMs(regions)
		}
	case JPEG:
		target := opts.Quality.targetSSIM()
		if opts.TargetSSIM > 0 && opts.TargetSSIM <= 1.0 {
			target = opts.TargetSSIM
		}

		// Only attach a gate when regions are configured. Without regions the
		// search behaves exactly as before, with the target SSIM alone.
		var gate *qualityGate
		if rs.len() > 0 {
			gate = newQualityGate(ctx, rs, target, src)
		}
		q, ssim, regions, cachedData, err := compressJPEGOptimal(ctx, src, &compressed, target, gate, opts)
		if err != nil {
			return nil, fmt.Errorf("fennec: JPEG compression: %w", err)
		}
		result.JPEGQuality, result.SSIM = q, ssim
		result.RegionResults = regions
		result.Regions = regions
		result.RegionSSIM = regionSSIMs(regions)
		if cachedData != nil {
			compressed.Reset()
			compressed.Write(cachedData)
		}
	default:
		return nil, ErrUnsupportedFormat
	}

	if err := opts.reportProgress(ctx, StageEncoding, 0.9); err != nil {
		return nil, err
	}
	result.CompressedData = compressed.Bytes()
	result.CompressedSize = int64(compressed.Len())
	result.computeStats()
	return result, nil
}

// encodingBuffer is a bytes.Buffer wrapper that satisfies io.Writer.
// Named to reflect its purpose: buffering encoded image data during compression.
// It is NOT safe for concurrent use.
type encodingBuffer struct {
	bytes.Buffer
}
