package fennec

import (
	"bytes"
	"context"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"io"
)

// compressJPEGOptimal uses binary search to find the lowest JPEG quality
// that still meets the target SSIM. Returns the quality, SSIM, measured
// region results, cached encoded bytes (from the winning iteration), and any
// error.
//
// Traditional approach: pick a quality number and hope for the best.
// Fennec approach: measure actual perceptual quality and optimize precisely.
//
// When gate is non-nil it enforces the full-image target together with any
// region thresholds; a quality is accepted only when every constraint passes.
//
// The cached JPEG bytes avoid the double-encode bug where the final output
// would be re-encoded.
func compressJPEGOptimal(
	ctx context.Context,
	src *image.NRGBA,
	w io.Writer,
	targetSSIM float64,
	gate *qualityGate,
	opts Options,
) (int, float64, []RegionResult, []byte, error) {
	if ctx == nil {
		ctx = context.Background()
	}

	// Guard: if target is 1.0 (Lossless) and format is JPEG, clamp to 0.999
	// since JPEG is inherently lossy and SSIM=1.0 is unreachable.
	if targetSSIM >= 1.0 {
		targetSSIM = 0.999
	}

	// Binary search bounds.
	lo, hi := 1, 100
	bestQuality := hi
	bestSSIM := 1.0
	var bestRegions []RegionResult
	var bestData []byte

	// Fast path: if target is very high, start from higher quality.
	if targetSSIM >= 0.99 {
		lo = 75
	} else if targetSSIM >= 0.97 {
		lo = 50
	} else if targetSSIM >= 0.94 {
		lo = 30
	} else if targetSSIM >= 0.90 {
		lo = 15
	}

	for lo <= hi {
		if err := ctx.Err(); err != nil {
			return 0, 0, nil, nil, err
		}
		mid := (lo + hi) / 2

		// Encode at this quality.
		var buf bytes.Buffer
		if err := encodeJPEG(&buf, src, mid, opts.Subsample); err != nil {
			return 0, 0, nil, nil, err
		}

		// Decode back to measure actual quality.
		decoded, err := jpeg.Decode(bytes.NewReader(buf.Bytes()))
		if err != nil {
			return 0, 0, nil, nil, err
		}
		decodedNRGBA := toNRGBARef(decoded)

		// Compute SSIM between original and compressed, plus region quality.
		ssim, regions, ok := evaluateCandidate(gate, src, targetSSIM, decodedNRGBA)

		if ok {
			// Quality is sufficient — cache this result and try lower quality.
			bestQuality = mid
			bestSSIM = ssim
			bestRegions = regions
			bestData = copyBytes(buf.Bytes())
			hi = mid - 1
		} else {
			// Quality too low — increase quality.
			lo = mid + 1
		}
	}

	// Write the cached best result directly instead of re-encoding.
	if bestData != nil {
		_, err := w.Write(bestData)
		return bestQuality, bestSSIM, bestRegions, bestData, err
	}

	// No quality met every constraint. With region constraints this is an
	// explicit failure rather than silently writing a below-threshold image.
	if gate != nil && gate.rs != nil && gate.rs.len() > 0 {
		// Verify at maximum quality so the error can report the best achieved.
		if err := ctx.Err(); err != nil {
			return 0, 0, nil, nil, err
		}
		var buf bytes.Buffer
		if err := encodeJPEG(&buf, src, 100, opts.Subsample); err != nil {
			return 0, 0, nil, nil, err
		}
		if decoded, derr := jpeg.Decode(bytes.NewReader(buf.Bytes())); derr == nil {
			ssim, regions, _ := evaluateCandidate(gate, src, targetSSIM, toNRGBARef(decoded))
			if regionsMeet(regions) {
				// Only the whole-image target is out of reach; regions are
				// satisfied, so fall back to the best encode (preserving the
				// historical behaviour for unreachable global targets).
				if _, werr := w.Write(buf.Bytes()); werr != nil {
					return 0, 0, nil, nil, werr
				}
				return 100, ssim, regions, copyBytes(buf.Bytes()), nil
			}
			return 100, ssim, regions, nil, &QualityError{
				Target:      targetSSIM,
				Achieved:    ssim,
				JPEGQuality: 100,
				Regions:     regions,
			}
		}
		return 100, 0, nil, nil, &QualityError{Target: targetSSIM, JPEGQuality: 100}
	}

	// Fallback: encode at best quality found. This preserves the historical
	// behaviour for global-only SSIM targets that cannot be reached.
	if err := ctx.Err(); err != nil {
		return 0, 0, nil, nil, err
	}
	if err := encodeJPEG(w, src, bestQuality, opts.Subsample); err != nil {
		return 0, 0, nil, nil, err
	}
	return bestQuality, bestSSIM, bestRegions, nil, nil
}

// evaluateCandidate measures a candidate against the gate when present, or
// against the plain global target otherwise.
func evaluateCandidate(gate *qualityGate, ref *image.NRGBA, targetSSIM float64, candidate *image.NRGBA) (float64, []RegionResult, bool) {
	if gate != nil {
		return gate.evaluate(candidate)
	}
	ssim := SSIMFast(ref, candidate)
	return ssim, nil, targetSSIM <= 0 || ssim >= targetSSIM
}

// compressPNG applies PNG-specific optimizations.
func compressPNG(img *image.NRGBA, w io.Writer, opts Options) error {
	// Check if we can reduce to a palette (indexed color).
	paletted := tryPalettize(img, 256)
	if paletted != nil {
		encoder := png.Encoder{CompressionLevel: png.BestCompression}
		return encoder.Encode(w, paletted)
	}

	// Check if image is grayscale — use Gray format for ~3× savings.
	if isGrayscale(img) {
		gray := toGray(img)
		encoder := png.Encoder{CompressionLevel: png.BestCompression}
		return encoder.Encode(w, gray)
	}

	// Full NRGBA with best compression.
	encoder := png.Encoder{CompressionLevel: png.BestCompression}
	return encoder.Encode(w, img)
}

// tryPalettize attempts to convert the image to an indexed palette.
// Returns nil if the image has too many colors.
func tryPalettize(img *image.NRGBA, maxColors int) *image.Paletted {
	w := img.Bounds().Dx()
	h := img.Bounds().Dy()

	colorMap := make(map[[4]uint8]int)

	for y := 0; y < h; y++ {
		off := y * img.Stride
		for x := 0; x < w; x++ {
			i := off + x*4
			key := [4]uint8{img.Pix[i], img.Pix[i+1], img.Pix[i+2], img.Pix[i+3]}
			colorMap[key]++
			if len(colorMap) > maxColors {
				return nil
			}
		}
	}

	// Build palette.
	palette := make([]color.Color, 0, len(colorMap))
	colorIndex := make(map[[4]uint8]uint8, len(colorMap))

	for c := range colorMap {
		idx := uint8(len(palette))
		colorIndex[c] = idx
		palette = append(palette, color.NRGBA{c[0], c[1], c[2], c[3]})
	}

	// Create paletted image.
	paletted := image.NewPaletted(image.Rect(0, 0, w, h), palette)
	for y := 0; y < h; y++ {
		srcOff := y * img.Stride
		dstOff := y * paletted.Stride
		for x := 0; x < w; x++ {
			i := srcOff + x*4
			key := [4]uint8{img.Pix[i], img.Pix[i+1], img.Pix[i+2], img.Pix[i+3]}
			paletted.Pix[dstOff+x] = colorIndex[key]
		}
	}

	return paletted
}
