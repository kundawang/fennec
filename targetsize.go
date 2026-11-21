package fennec

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"sort"
)

const minJPEGQuality = 20

type sizeResult struct {
	data    []byte
	format  Format
	quality int
	ssim    float64
	finalW  int
	finalH  int
	img     *image.NRGBA
	regions []RegionResult
}

// targetSizeWholeImageSSIM returns the explicit whole-image SSIM target for
// target-size mode. Only TargetSSIM is honored here: the Quality preset is a
// quality-mode setting that target-size mode has never applied. JPEG cannot
// reach SSIM 1.0, so the practical ceiling matches the standard JPEG search.
func targetSizeWholeImageSSIM(opts Options) float64 {
	if opts.TargetSSIM <= 0 || opts.TargetSSIM > 1.0 {
		return 0
	}
	if opts.TargetSSIM >= 1.0 {
		return 0.999
	}
	return opts.TargetSSIM
}

func hitTargetSize(ctx context.Context, original *image.NRGBA, targetBytes int, rs *regionSet, opts Options) (*sizeResult, error) {
	candidates, err := collectSizeCandidates(ctx, original, targetBytes, opts)
	if err != nil {
		return nil, err
	}

	// Cancellation always wins over a partially-completed search.
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	// Without region constraints the historical selection is preserved
	// exactly: no extra quality gate is applied in target-size mode.
	if rs == nil || rs.len() == 0 {
		return selectSizeCandidate(candidates, targetBytes, opts), nil
	}

	// With regions configured, region thresholds always apply. An explicitly
	// requested whole-image TargetSSIM (if any) is enforced alongside them.
	gate := newQualityGate(ctx, rs, targetSizeWholeImageSSIM(opts), original)

	var passing []*sizeResult
	for _, c := range candidates {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if candidateMeetsGate(gate, c) {
			passing = append(passing, c)
		}
	}

	if len(passing) > 0 {
		return selectSizeCandidate(passing, targetBytes, opts), nil
	}

	// No candidate satisfies size and quality together. Distinguish a
	// region threshold that cannot be met even at maximum quality from a
	// genuine size/quality conflict by probing the highest-quality encodes.
	best, probeErr := probeMaxQuality(ctx, original, opts, gate)
	if probeErr != nil {
		return nil, probeErr
	}
	regionsOK := regionsMeet(best.regions)

	qerr := &QualityError{
		Target:      gate.targetSSIM,
		Achieved:    best.ssim,
		JPEGQuality: best.quality,
		Regions:     best.regions,
		TargetSize:  targetBytes,
	}
	if !regionsOK {
		qerr.Reason = fmt.Sprintf("region quality cannot be met within %d bytes (best effort is %d bytes)",
			targetBytes, len(best.data))
	} else if gate.targetSSIM > 0 && best.ssim < gate.targetSSIM {
		qerr.Reason = fmt.Sprintf("whole-image SSIM %.4f is below its %.4f target within %d bytes "+
			"(best effort is %d bytes)", best.ssim, gate.targetSSIM, targetBytes, len(best.data))
	} else {
		qerr.Reason = fmt.Sprintf("target size %d bytes cannot be met together with the quality constraints "+
			"(best effort is %d bytes)", targetBytes, len(best.data))
	}
	return nil, qerr
}

// probeMaxQuality encodes at the highest available quality for the formats the
// options allow, measures each, and returns the attempt that best satisfies the
// quality constraints. It is used only to explain a conflict in an error.
func probeMaxQuality(ctx context.Context, original *image.NRGBA, opts Options, gate *qualityGate) (*sizeResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	w, h := original.Bounds().Dx(), original.Bounds().Dy()

	tryJPEG := opts.Format != PNG && isOpaque(original)
	tryPNG := opts.Format != JPEG

	var best *sizeResult
	consider := func(r *sizeResult) {
		if r == nil || r.img == nil {
			return
		}
		if gate != nil {
			r.ssim, r.regions, _ = gate.evaluate(r.img)
		}
		if best == nil {
			best = r
			return
		}
		bestScore := regionScore(best)
		score := regionScore(r)
		if score > bestScore || (score == bestScore && r.ssim > best.ssim) {
			best = r
		}
	}

	if tryJPEG {
		var buf bytes.Buffer
		if err := encodeJPEG(&buf, original, 100, false); err != nil {
			return nil, err
		}
		img := decodeJPEGFromBytes(buf.Bytes())
		if img == nil {
			img = original
		}
		consider(&sizeResult{
			data: copyBytes(buf.Bytes()), format: JPEG, quality: 100,
			ssim: computeSSIMNRGBA(original, img), finalW: w, finalH: h, img: img,
		})
	}
	if tryPNG {
		var buf bytes.Buffer
		if err := compressPNG(original, &buf, opts); err != nil {
			return nil, err
		}
		consider(&sizeResult{
			data: copyBytes(buf.Bytes()), format: PNG, quality: 0,
			ssim: 1.0, finalW: w, finalH: h, img: original,
		})
	}

	if best == nil {
		return nil, fmt.Errorf("fennec: no encoder available for format %s", opts.Format)
	}
	return best, nil
}

// regionScore counts satisfied region thresholds; used only to rank
// diagnostics when constraints conflict.
func regionScore(r *sizeResult) int {
	score := 0
	for _, rr := range r.regions {
		if rr.Transparent || rr.MinSSIM <= 0 || rr.Meets {
			score++
		}
	}
	return score
}

// collectSizeCandidates runs every applicable strategy and returns all
// candidate encodings.
func collectSizeCandidates(ctx context.Context, original *image.NRGBA, targetBytes int, opts Options) ([]*sizeResult, error) {
	wantPNG := opts.Format == PNG
	wantJPEG := opts.Format == JPEG
	canUseJPEG := !wantPNG && isOpaque(original)

	var candidates []*sizeResult

	if (canUseJPEG || wantJPEG) && ctx.Err() == nil {
		if r, err := jpegQualitySearch(ctx, original, targetBytes); err == nil && r != nil && r.quality >= minJPEGQuality {
			candidates = append(candidates, r)
		}
	}

	if !wantJPEG && ctx.Err() == nil {
		if r, err := quantizeStrategy(ctx, original, targetBytes); err == nil && r != nil {
			candidates = append(candidates, r)
		}
	}

	if (canUseJPEG || wantJPEG) && ctx.Err() == nil {
		if r, err := jpegQualityScaleSearch(ctx, original, targetBytes); err == nil && r != nil {
			candidates = append(candidates, r)
		}
	}

	if len(candidates) == 0 && ctx.Err() == nil {
		format := opts.Format
		if format == Auto {
			format = PNG
			if canUseJPEG {
				format = JPEG
			}
		}
		if r, err := scaleSearch(ctx, original, targetBytes, format); err == nil && r != nil {
			candidates = append(candidates, r)
		}
	}

	if len(candidates) == 0 {
		r, err := fallbackTargetSizeEncode(original, targetBytes, canUseJPEG || wantJPEG, opts)
		if err != nil {
			return nil, err
		}
		candidates = append(candidates, r)
	}
	return candidates, nil
}

// selectSizeCandidate picks the best candidate for the size target.
func selectSizeCandidate(candidates []*sizeResult, targetBytes int, opts Options) *sizeResult {
	var best *sizeResult
	for _, c := range candidates {
		if best == nil || betterFit(c, best, targetBytes) {
			best = c
		}
	}
	return best
}

// candidateMeetsGate decodes an encoded candidate and measures it against the
// original through the quality gate. It records the measured SSIM and region
// results on the candidate.
func candidateMeetsGate(gate *qualityGate, c *sizeResult) bool {
	if gate == nil {
		return true
	}
	decoded := decodeImageFromBytes(c.data)
	if decoded == nil {
		return false
	}
	ssim, regions, ok := gate.evaluate(decoded)
	c.ssim = ssim
	c.regions = regions
	return ok
}

func decodeImageFromBytes(data []byte) *image.NRGBA {
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil
	}
	return toNRGBARef(img)
}

func fallbackTargetSizeEncode(original *image.NRGBA, target int, useJPEG bool, opts Options) (*sizeResult, error) {
	w, h := original.Bounds().Dx(), original.Bounds().Dy()
	var buf bytes.Buffer
	if useJPEG {
		if err := encodeJPEG(&buf, original, 1, false); err != nil {
			return nil, fmt.Errorf("fennec: fallback JPEG encode: %w", err)
		}
		return &sizeResult{data: buf.Bytes(), format: JPEG, quality: 1, ssim: computeSSIMNRGBA(original, original), finalW: w, finalH: h, img: original}, nil
	}
	if err := compressPNG(original, &buf, opts); err != nil {
		return nil, fmt.Errorf("fennec: fallback PNG encode: %w", err)
	}
	return &sizeResult{data: buf.Bytes(), format: PNG, ssim: 1.0, finalW: w, finalH: h, img: original}, nil
}

func betterFit(candidate, current *sizeResult, target int) bool {
	cSize := int64(len(candidate.data))
	bSize := int64(len(current.data))
	t := int64(target)

	cUnder := cSize <= t
	bUnder := bSize <= t

	if cUnder && !bUnder {
		return true
	}
	if !cUnder && bUnder {
		return false
	}
	if cUnder && bUnder {
		if candidate.ssim != current.ssim {
			return candidate.ssim > current.ssim
		}
		return candidate.quality > current.quality
	}
	return cSize < bSize
}

// ── Strategy 1 ──────────────────────────────────────────────────────────────

func jpegQualitySearch(ctx context.Context, src *image.NRGBA, targetBytes int) (*sizeResult, error) {
	return jpegQualitySearchOpt(ctx, src, targetBytes, false)
}

func jpegQualitySearchFast(ctx context.Context, src *image.NRGBA, targetBytes int) (*sizeResult, error) {
	return jpegQualitySearchOpt(ctx, src, targetBytes, true)
}

func jpegQualitySearchOpt(ctx context.Context, src *image.NRGBA, targetBytes int, skipSSIM bool) (*sizeResult, error) {
	w := src.Bounds().Dx()
	h := src.Bounds().Dy()
	pixels := w * h

	targetBPP := float64(targetBytes*8) / float64(pixels)
	lo, hi := 1, 100
	if targetBPP < 0.5 {
		hi = 40
	} else if targetBPP < 1.0 {
		lo, hi = 10, 70
	} else if targetBPP < 2.0 {
		lo, hi = 30, 90
	} else if targetBPP > 4.0 {
		lo = 60
	}

	var bestBuf []byte
	bestQ := 0
	bestSSIM := 0.0

	for lo <= hi {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		mid := (lo + hi) / 2
		var buf bytes.Buffer
		if err := encodeJPEG(&buf, src, mid, false); err != nil {
			return nil, err
		}

		if int64(buf.Len()) <= int64(targetBytes) {
			bestBuf = copyBytes(buf.Bytes())
			bestQ = mid
			if !skipSSIM {
				decoded := decodeJPEGFromBytes(bestBuf)
				if decoded != nil {
					bestSSIM = computeSSIMNRGBA(src, decoded)
				}
			}
			lo = mid + 1
		} else {
			hi = mid - 1
		}
	}

	if bestBuf == nil {
		return nil, nil
	}

	return &sizeResult{
		data: bestBuf, format: JPEG, quality: bestQ,
		ssim: bestSSIM, finalW: w, finalH: h, img: src,
	}, nil
}

// ── Strategy 2 ──────────────────────────────────────────────────────────────

func quantizeStrategy(ctx context.Context, src *image.NRGBA, targetBytes int) (*sizeResult, error) {
	w := src.Bounds().Dx()
	h := src.Bounds().Dy()

	for _, maxColors := range []int{256, 128, 64, 32, 16} {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		palette := medianCut(src, maxColors)
		indexed := applyPalette(src, palette)

		var buf bytes.Buffer
		encoder := png.Encoder{CompressionLevel: png.BestCompression}
		if err := encoder.Encode(&buf, indexed); err != nil {
			continue
		}

		if int64(buf.Len()) <= int64(targetBytes) {
			quantizedNRGBA := palettedToNRGBA(indexed)
			ssim := computeSSIMNRGBA(src, quantizedNRGBA)

			return &sizeResult{
				data: buf.Bytes(), format: PNG, quality: 0,
				ssim: ssim, finalW: w, finalH: h, img: quantizedNRGBA,
			}, nil
		}
	}

	return nil, nil
}

// ── Strategy 3 ──────────────────────────────────────────────────────────────

func jpegQualityScaleSearch(ctx context.Context, src *image.NRGBA, targetBytes int) (*sizeResult, error) {
	origW, origH := src.Bounds().Dx(), src.Bounds().Dy()
	bestCand := findBestScaleBinary(ctx, src, origW, origH, targetBytes)
	bestCand = findBestScaleFixed(ctx, src, origW, origH, targetBytes, bestCand)

	if bestCand == nil {
		return nil, nil
	}

	finalW := int(float64(origW) * bestCand.scale)
	finalH := int(float64(origH) * bestCand.scale)
	finalScaled := lanczosResize(src, finalW, finalH)

	r, err := jpegQualitySearch(ctx, finalScaled, targetBytes)
	if err != nil || r == nil || r.quality < minJPEGQuality {
		return nil, nil
	}

	r.ssim = computeSSIMNRGBA(src, finalScaled)
	r.finalW, r.finalH = finalW, finalH
	r.img = finalScaled
	return r, nil
}

type scaleCandidate struct {
	scale   float64
	quality int
	size    int
}

func findBestScaleBinary(ctx context.Context, src *image.NRGBA, origW, origH, targetBytes int) *scaleCandidate {
	var bestCand *scaleCandidate
	loScale, hiScale := 0.05, 1.0
	for i := 0; i < 10; i++ {
		if ctx.Err() != nil {
			break
		}
		midScale := (loScale + hiScale) / 2
		newW, newH := int(float64(origW)*midScale), int(float64(origH)*midScale)
		if newW < 8 || newH < 8 {
			loScale = midScale
			continue
		}
		r, err := jpegQualitySearchFast(ctx, boxDownsample(src, newW, newH), targetBytes)
		if err == nil && r != nil && int64(len(r.data)) <= int64(targetBytes) && r.quality >= minJPEGQuality {
			bestCand = &scaleCandidate{scale: midScale, quality: r.quality, size: len(r.data)}
			loScale = midScale
		} else {
			hiScale = midScale
		}
	}
	return bestCand
}

func findBestScaleFixed(ctx context.Context, src *image.NRGBA, origW, origH, targetBytes int, best *scaleCandidate) *scaleCandidate {
	for _, scale := range []float64{0.75, 0.50, 0.375, 0.25} {
		if ctx.Err() != nil {
			break
		}
		newW, newH := int(float64(origW)*scale), int(float64(origH)*scale)
		if newW < 8 || newH < 8 {
			continue
		}
		r, err := jpegQualitySearchFast(ctx, boxDownsample(src, newW, newH), targetBytes)
		if err == nil && r != nil && int64(len(r.data)) <= int64(targetBytes) && r.quality >= minJPEGQuality {
			if best == nil || scale > best.scale {
				best = &scaleCandidate{scale: scale, quality: r.quality, size: len(r.data)}
			}
		}
	}
	return best
}

// ── Strategy 4 ──────────────────────────────────────────────────────────────

func scaleSearch(ctx context.Context, src *image.NRGBA, targetBytes int, format Format) (*sizeResult, error) {
	origW, origH := src.Bounds().Dx(), src.Bounds().Dy()
	lo, hi, bestScale, bestQ := 0.05, 1.0, 0.0, 0

	for i := 0; i < 12; i++ {
		if ctx.Err() != nil {
			break
		}
		mid := (lo + hi) / 2
		newW, newH := int(float64(origW)*mid), int(float64(origH)*mid)
		if newW < 1 || newH < 1 {
			lo = mid
			continue
		}

		fits, q := testScaleFits(ctx, boxDownsample(src, newW, newH), targetBytes, format)
		if fits {
			bestScale, bestQ, lo = mid, q, mid
		} else {
			hi = mid
		}
	}

	if bestScale == 0 {
		return nil, nil
	}
	finalW, finalH := int(float64(origW)*bestScale), int(float64(origH)*bestScale)
	return executeFinalScaleEncode(ctx, src, format, bestScale, bestQ, finalW, finalH, targetBytes)
}

func testScaleFits(ctx context.Context, scaled *image.NRGBA, targetBytes int, format Format) (bool, int) {
	if format == JPEG {
		if r, err := jpegQualitySearchFast(ctx, scaled, targetBytes); err == nil && r != nil && int64(len(r.data)) <= int64(targetBytes) && r.quality >= minJPEGQuality {
			return true, r.quality
		}
		return false, 0
	}
	var buf bytes.Buffer
	encoder := png.Encoder{CompressionLevel: png.BestCompression}
	if err := encoder.Encode(&buf, scaled); err == nil && int64(buf.Len()) <= int64(targetBytes) {
		return true, 0
	}
	return false, 0
}

func executeFinalScaleEncode(ctx context.Context, src *image.NRGBA, format Format, scale float64, bestQ, finalW, finalH, targetBytes int) (*sizeResult, error) {
	scaled := lanczosResize(src, finalW, finalH)
	var buf bytes.Buffer
	if format == JPEG {
		r, err := jpegQualitySearchFast(ctx, scaled, targetBytes)
		if err == nil && r != nil {
			return &sizeResult{data: r.data, format: JPEG, quality: r.quality, ssim: computeSSIMNRGBA(src, scaled), finalW: finalW, finalH: finalH, img: scaled}, nil
		}
		if err := encodeJPEG(&buf, scaled, bestQ, false); err != nil {
			return nil, err
		}
	} else {
		encoder := png.Encoder{CompressionLevel: png.BestCompression}
		if err := encoder.Encode(&buf, scaled); err != nil {
			return nil, err
		}
	}
	return &sizeResult{data: buf.Bytes(), format: format, quality: bestQ, ssim: computeSSIMNRGBA(src, scaled), finalW: finalW, finalH: finalH, img: scaled}, nil
}

// ── Median-Cut Color Quantizer ──────────────────────────────────────────────

type colorBox struct {
	pixels     [][3]uint8
	rMin, rMax uint8
	gMin, gMax uint8
	bMin, bMax uint8
}

func newColorBox(pixels [][3]uint8) *colorBox {
	box := &colorBox{
		pixels: pixels,
		rMin:   255, gMin: 255, bMin: 255,
	}
	for _, p := range pixels {
		if p[0] < box.rMin {
			box.rMin = p[0]
		}
		if p[0] > box.rMax {
			box.rMax = p[0]
		}
		if p[1] < box.gMin {
			box.gMin = p[1]
		}
		if p[1] > box.gMax {
			box.gMax = p[1]
		}
		if p[2] < box.bMin {
			box.bMin = p[2]
		}
		if p[2] > box.bMax {
			box.bMax = p[2]
		}
	}
	return box
}

func (b *colorBox) longestAxis() int {
	rRange := int(b.rMax) - int(b.rMin)
	gRange := int(b.gMax) - int(b.gMin)
	bRange := int(b.bMax) - int(b.bMin)
	if rRange >= gRange && rRange >= bRange {
		return 0
	}
	if gRange >= bRange {
		return 1
	}
	return 2
}

func (b *colorBox) average() color.NRGBA {
	if len(b.pixels) == 0 {
		return color.NRGBA{0, 0, 0, 255}
	}
	var rSum, gSum, bSum int64
	for _, p := range b.pixels {
		rSum += int64(p[0])
		gSum += int64(p[1])
		bSum += int64(p[2])
	}
	n := int64(len(b.pixels))
	return color.NRGBA{
		R: uint8(rSum / n), G: uint8(gSum / n), B: uint8(bSum / n), A: 255,
	}
}

func (b *colorBox) volume() int {
	return (int(b.rMax) - int(b.rMin) + 1) *
		(int(b.gMax) - int(b.gMin) + 1) *
		(int(b.bMax) - int(b.bMin) + 1)
}

func medianCut(img *image.NRGBA, maxColors int) color.Palette {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()

	maxSamples := 100000
	step := 1
	if w*h > maxSamples {
		step = (w * h) / maxSamples
		if step < 1 {
			step = 1
		}
	}

	pixels := make([][3]uint8, 0, w*h/step)
	for i := 0; i < w*h; i += step {
		off := i * 4
		if off+3 < len(img.Pix) {
			pixels = append(pixels, [3]uint8{img.Pix[off], img.Pix[off+1], img.Pix[off+2]})
		}
	}

	if len(pixels) == 0 {
		return color.Palette{color.NRGBA{0, 0, 0, 255}}
	}

	boxes := []*colorBox{newColorBox(pixels)}

	for len(boxes) < maxColors {
		bestIdx := -1
		bestScore := -1
		for i, box := range boxes {
			if len(box.pixels) < 2 {
				continue
			}
			score := box.volume() * len(box.pixels)
			if score > bestScore {
				bestScore = score
				bestIdx = i
			}
		}
		if bestIdx == -1 {
			break
		}

		box := boxes[bestIdx]
		axis := box.longestAxis()

		sort.Slice(box.pixels, func(i, j int) bool {
			return box.pixels[i][axis] < box.pixels[j][axis]
		})

		mid := len(box.pixels) / 2
		left := newColorBox(box.pixels[:mid])
		right := newColorBox(box.pixels[mid:])

		boxes[bestIdx] = left
		boxes = append(boxes, right)
	}

	palette := make(color.Palette, len(boxes))
	for i, box := range boxes {
		palette[i] = box.average()
	}
	return palette
}

func applyPalette(src *image.NRGBA, palette color.Palette) *image.Paletted {
	bounds := src.Bounds()
	w, h := bounds.Dx(), bounds.Dy()

	indexed := image.NewPaletted(bounds, palette)

	type cacheKey struct{ r, g, b uint8 }
	cache := make(map[cacheKey]uint8, 256)

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			off := y*src.Stride + x*4
			r, g, b := src.Pix[off], src.Pix[off+1], src.Pix[off+2]

			key := cacheKey{r, g, b}
			if idx, ok := cache[key]; ok {
				indexed.Pix[y*indexed.Stride+x] = idx
				continue
			}

			bestIdx := 0
			bestDist := math.MaxInt32
			for i, c := range palette {
				pr, pg, pb, _ := c.RGBA()
				dr := int(r) - int(pr>>8)
				dg := int(g) - int(pg>>8)
				db := int(b) - int(pb>>8)
				dist := dr*dr + dg*dg + db*db
				if dist < bestDist {
					bestDist = dist
					bestIdx = i
				}
			}

			cache[key] = uint8(bestIdx)
			indexed.Pix[y*indexed.Stride+x] = uint8(bestIdx)
		}
	}
	return indexed
}

func palettedToNRGBA(p *image.Paletted) *image.NRGBA {
	bounds := p.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	dst := image.NewNRGBA(bounds)

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, b, a := p.At(x+bounds.Min.X, y+bounds.Min.Y).RGBA()
			off := y*dst.Stride + x*4
			dst.Pix[off] = uint8(r >> 8)
			dst.Pix[off+1] = uint8(g >> 8)
			dst.Pix[off+2] = uint8(b >> 8)
			dst.Pix[off+3] = uint8(a >> 8)
		}
	}
	return dst
}

// ── Helpers ─────────────────────────────────────────────────────────────────

func copyBytes(src []byte) []byte {
	dst := make([]byte, len(src))
	copy(dst, src)
	return dst
}

func decodeJPEGFromBytes(data []byte) *image.NRGBA {
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil
	}
	return toNRGBARef(img)
}

func computeSSIMNRGBA(a, b *image.NRGBA) float64 {
	if a.Bounds().Dx() != b.Bounds().Dx() || a.Bounds().Dy() != b.Bounds().Dy() {
		b = lanczosResize(b, a.Bounds().Dx(), a.Bounds().Dy())
	}
	return SSIMFast(a, b)
}
