package fennec

import (
	"context"
	"errors"
	"fmt"
	"image"
	"math"
	"sort"
)

// Sentinel errors for region quality constraints.
var (
	// ErrInvalidRegion is returned when a Region has an empty rectangle or an
	// out-of-range MinSSIM threshold.
	ErrInvalidRegion = errors.New("fennec: invalid region")

	// ErrRegionOutOfBounds is returned when a Region lies entirely outside the
	// source image. Regions that only partially overlap the image are clamped
	// to the image instead.
	ErrRegionOutOfBounds = errors.New("fennec: region outside image bounds")

	// ErrQualityTargetUnreachable is returned when no encoding can satisfy the
	// requested quality constraints, either alone or together with a target
	// file size. It is also returned when the requested region quality cannot
	// be reached even at maximum quality.
	ErrQualityTargetUnreachable = errors.New("fennec: quality target cannot be satisfied")

	// ErrImpossibleTargets is an alias of ErrQualityTargetUnreachable for
	// callers that reason in terms of competing size/quality objectives.
	ErrImpossibleTargets = ErrQualityTargetUnreachable
)

// QualityError reports which quality constraints could not be met. It wraps
// ErrQualityTargetUnreachable so callers can use errors.Is.
type QualityError struct {
	// Target is the full-image SSIM target (0 when unset).
	Target float64
	// Achieved is the best full-image SSIM reached.
	Achieved float64
	// JPEGQuality is the quality at which Achieved was measured.
	JPEGQuality int
	// TargetSize is the requested file size in bytes (0 when none).
	TargetSize int
	// Reason adds a human-readable explanation of the conflict.
	Reason string
	// Regions holds per-region measurements at the best attempt.
	Regions []RegionResult
}

func (e *QualityError) Error() string {
	msg := fmt.Sprintf("fennec: quality target %.4f cannot be satisfied (best SSIM %.4f at JPEG Q=%d)",
		e.Target, e.Achieved, e.JPEGQuality)
	for i, r := range e.Regions {
		if r.Transparent || r.MinSSIM <= 0 || r.Meets {
			continue
		}
		msg += fmt.Sprintf("; region %d %v needs SSIM >= %.4f, best %.4f",
			i, r.SourceRect, r.MinSSIM, r.SSIM)
	}
	if e.TargetSize > 0 {
		msg += fmt.Sprintf("; target size %d bytes", e.TargetSize)
	}
	if e.Reason != "" {
		msg += "; " + e.Reason
	}
	return msg
}

// Unwrap allows errors.Is(err, ErrQualityTargetUnreachable).
func (e *QualityError) Unwrap() error { return ErrQualityTargetUnreachable }

// Region describes a rectangular area of an image that must keep at least
// MinSSIM structural similarity after compression. This lets callers protect
// subjects such as eyes, faces, logos, or text while allowing the rest of the
// image (for example a marketing-poster background) to be compressed harder.
//
// Rect coordinates are expressed in the decoded source image's pixel grid
// (the same grid other pixel APIs use). When Options.AutoOrient is true (the
// default) and the file carries EXIF orientation, Fennec maps each rectangle
// through the rotation/flip that is applied to the pixels. Rectangles are then
// mapped through MaxWidth/MaxHeight resizing and through target-size
// downscaling before they are measured, so a rectangle always tracks the same
// subject in the output.
//
// Multiple regions are measured independently. Where two regions overlap, both
// thresholds apply, so the stricter threshold governs the overlapping pixels.
//
// Rects that partially overlap the image are clamped to the image. Rects that
// are empty or lie entirely outside the image produce ErrInvalidRegion or
// ErrRegionOutOfBounds. A region whose source pixels are all fully transparent
// is measured as lossless (SSIM 1) and reported with Transparent set, since a
// lossy threshold is meaningless for invisible pixels.
type Region struct {
	// Rect is the region of interest. Either set Rect directly or use the
	// X/Y/Width/Height (or X/Y/W/H) shorthand below.
	Rect image.Rectangle

	// X, Y, Width, Height are an optional shorthand for Rect. They are used
	// only when Rect is empty and Width/Height (or W/H) are positive.
	X, Y, Width, Height int

	// W and H are aliases for Width and Height.
	W, H int

	// MinSSIM is the minimum acceptable SSIM for this region, in [0.0, 1.0].
	// 0.0 measures and reports the region without constraining it.
	MinSSIM float64
}

// QualityRegion is an alias for Region.
type QualityRegion = Region

// RegionConstraint is an alias for Region.
type RegionConstraint = Region

// ROI is an alias for Region.
type ROI = Region

// rect returns the configured rectangle, honoring the shorthand fields when
// Rect is unset.
func (r Region) rect() image.Rectangle {
	if !r.Rect.Empty() {
		return r.Rect
	}
	w, h := r.Width, r.Height
	if w <= 0 {
		w = r.W
	}
	if h <= 0 {
		h = r.H
	}
	if w > 0 && h > 0 {
		return image.Rect(r.X, r.Y, r.X+w, r.Y+h)
	}
	return r.Rect
}

// RegionResult reports the measured quality of one configured Region.
type RegionResult struct {
	// Rect is the region mapped into final output coordinates, including EXIF
	// orientation, MaxWidth/MaxHeight resizing, and any target-size scaling.
	Rect image.Rectangle

	// SourceRect is the requested region after clamping to the source image,
	// in the source coordinate space (before orientation and scaling).
	SourceRect image.Rectangle

	// MinSSIM is the requested minimum SSIM.
	MinSSIM float64

	// SSIM is the measured structural similarity of the final output against
	// the source for this region.
	SSIM float64

	// ActualSSIM is an alias of SSIM.
	ActualSSIM float64

	// Transparent reports that every source pixel in the region was fully
	// transparent, so no quality threshold was applied.
	Transparent bool

	// Meets reports whether the measured SSIM satisfied MinSSIM.
	Meets bool
}

// RegionQuality is an alias for RegionResult.
type RegionQuality = RegionResult

// regionMeasureMaxDim bounds the resolution at which regions are measured.
// This keeps quality checks predictable for very large images while remaining
// far above what SSIMFast uses internally (512 px).
const regionMeasureMaxDim = 1024

// regionTarget is a validated region in source and oriented coordinates.
type regionTarget struct {
	request  image.Rectangle // clamped input rectangle
	oriented image.Rectangle // rectangle after EXIF orientation
	minSSIM  float64
}

// regionSet holds regions mapped into the oriented coordinate space. It is
// immutable after construction and safe for concurrent use.
type regionSet struct {
	targets []regionTarget
	enforce []enforcedRegion // disjoint cells at their strictest threshold
	baseW   int              // oriented image width
	baseH   int              // oriented image height
}

// mappedRegion is a region mapped to a concrete image size.
type mappedRegion struct {
	request image.Rectangle
	rect    image.Rectangle
	minSSIM float64
}

// enforcedRegion is a rectangle that must meet minSSIM. Space covered by
// several overlapping regions is broken into disjoint enforcement rectangles,
// each at the strictest threshold covering it.
type enforcedRegion struct {
	rect    image.Rectangle
	minSSIM float64
}

// newRegionSet validates and normalizes regions for an image of w x h pixels.
// When autoOrient is true and orient is not normal, regions are mapped through
// the same rotation/flip that will be applied to the pixels.
func newRegionSet(regions []Region, w, h int, orient Orientation, autoOrient bool) (*regionSet, error) {
	if len(regions) == 0 {
		return nil, nil
	}

	bounds := image.Rect(0, 0, w, h)
	outW, outH := w, h
	applyOrient := autoOrient && orient > OrientNormal
	if applyOrient {
		outW, outH = orientedSize(w, h, orient)
	}

	rs := &regionSet{baseW: outW, baseH: outH, targets: make([]regionTarget, 0, len(regions))}
	for i, r := range regions {
		rect := r.rect().Canon()
		if rect.Empty() {
			return nil, fmt.Errorf("%w: regions[%d] has an empty rectangle", ErrInvalidRegion, i)
		}
		if math.IsNaN(r.MinSSIM) || math.IsInf(r.MinSSIM, 0) ||
			r.MinSSIM < 0 || r.MinSSIM > 1 {
			return nil, fmt.Errorf("%w: regions[%d] MinSSIM must be in [0.0, 1.0], got %f",
				ErrInvalidRegion, i, r.MinSSIM)
		}

		clamped := rect.Intersect(bounds)
		if clamped.Empty() {
			return nil, fmt.Errorf("%w: regions[%d] rect %v does not overlap %dx%d image",
				ErrRegionOutOfBounds, i, rect, w, h)
		}

		oriented := clamped
		if applyOrient {
			oriented = mapRectOrientation(clamped, w, h, orient)
		}

		rs.targets = append(rs.targets, regionTarget{
			request:  clamped,
			oriented: oriented,
			minSSIM:  r.MinSSIM,
		})
	}
	rs.enforce = disjointEnforcement(rs.targets)
	return rs, nil
}

// disjointEnforcement converts overlapping regions into a disjoint set of
// rectangles where each piece carries the strictest threshold among the
// regions covering it. This makes an overlap genuinely held to the stricter
// threshold rather than merely measured twice.
func disjointEnforcement(targets []regionTarget) []enforcedRegion {
	// Collect candidate boundaries from every rect, then evaluate each cell's
	// covering set. Rects are few and axis-aligned, so this grid cut is both
	// exact and cheap.
	xs := make([]int, 0, len(targets)*2)
	ys := make([]int, 0, len(targets)*2)
	for _, t := range targets {
		xs = append(xs, t.oriented.Min.X, t.oriented.Max.X)
		ys = append(ys, t.oriented.Min.Y, t.oriented.Max.Y)
	}
	xs = sortedUniqueInts(xs)
	ys = sortedUniqueInts(ys)

	var out []enforcedRegion
	for yi := 0; yi+1 < len(ys); yi++ {
		y0, y1 := ys[yi], ys[yi+1]
		if y1 <= y0 {
			continue
		}
		for xi := 0; xi+1 < len(xs); xi++ {
			x0, x1 := xs[xi], xs[xi+1]
			if x1 <= x0 {
				continue
			}
			cell := image.Rect(x0, y0, x1, y1)
			minSSIM := -1.0
			for _, t := range targets {
				if cell.In(t.oriented) && t.minSSIM > minSSIM {
					minSSIM = t.minSSIM
				}
			}
			if minSSIM < 0 {
				continue // not covered by any region
			}
			out = append(out, enforcedRegion{rect: cell, minSSIM: minSSIM})
		}
	}
	return out
}

// sortedUniqueInts sorts a small int slice in place and returns the
// de-duplicated prefix.
func sortedUniqueInts(v []int) []int {
	sort.Ints(v)
	j := 1
	for i := 1; i < len(v); i++ {
		if v[i] != v[j-1] {
			v[j] = v[i]
			j++
		}
	}
	if len(v) == 0 {
		return v
	}
	return v[:j]
}

// len returns the number of configured regions.
func (rs *regionSet) len() int {
	if rs == nil {
		return 0
	}
	return len(rs.targets)
}

// mappedFor maps every region into an image of outW x outH pixels. If the set
// is empty it returns nil.
func (rs *regionSet) mappedFor(outW, outH int) []mappedRegion {
	if rs == nil || len(rs.targets) == 0 || rs.baseW <= 0 || rs.baseH <= 0 {
		return nil
	}
	if outW <= 0 || outH <= 0 {
		return nil
	}
	out := make([]mappedRegion, 0, len(rs.targets))
	for _, t := range rs.targets {
		rect := mapRectScale(t.oriented, rs.baseW, rs.baseH, outW, outH)
		out = append(out, mappedRegion{
			request: t.request,
			rect:    rect,
			minSSIM: t.minSSIM,
		})
	}
	return out
}

// mappedEnforcement maps the disjoint enforcement cells into an image of
// outW x outH pixels.
func (rs *regionSet) mappedEnforcement(outW, outH int) []enforcedRegion {
	if rs == nil || rs.baseW <= 0 || rs.baseH <= 0 || outW <= 0 || outH <= 0 {
		return nil
	}
	out := make([]enforcedRegion, 0, len(rs.enforce))
	for _, e := range rs.enforce {
		out = append(out, enforcedRegion{
			rect:    mapRectScale(e.rect, rs.baseW, rs.baseH, outW, outH),
			minSSIM: e.minSSIM,
		})
	}
	return out
}

// orientedSize returns the displayed dimensions for an oriented image.
func orientedSize(w, h int, orient Orientation) (int, int) {
	switch orient {
	case OrientTranspose, OrientRotate90CW, OrientTransverse, OrientRotate270CW:
		return h, w
	default:
		return w, h
	}
}

// mapRectOrientation maps a rectangle from stored pixel coordinates to
// displayed (EXIF-oriented) coordinates.
func mapRectOrientation(r image.Rectangle, w, h int, orient Orientation) image.Rectangle {
	switch orient {
	case OrientFlipH:
		return image.Rect(w-r.Max.X, r.Min.Y, w-r.Min.X, r.Max.Y)
	case OrientRotate180:
		return image.Rect(w-r.Max.X, h-r.Max.Y, w-r.Min.X, h-r.Min.Y)
	case OrientFlipV:
		return image.Rect(r.Min.X, h-r.Max.Y, r.Max.X, h-r.Min.Y)
	case OrientTranspose:
		// Rotate 270° CW, then flip horizontally. Output is h x w.
		return image.Rect(h-r.Max.Y, w-r.Max.X, h-r.Min.Y, w-r.Min.X)
	case OrientRotate90CW:
		return image.Rect(h-r.Max.Y, r.Min.X, h-r.Min.Y, r.Max.X)
	case OrientTransverse:
		// Rotate 90° CW, then flip horizontally. Output is h x w.
		return image.Rect(r.Min.Y, r.Min.X, r.Max.Y, r.Max.X)
	case OrientRotate270CW:
		return image.Rect(r.Min.Y, w-r.Max.X, r.Max.Y, w-r.Min.X)
	default:
		return r
	}
}

// mapRectScale maps a rectangle from inW x inH coordinates to outW x outH,
// rounding outward so the mapped region covers every source pixel. The result
// is clamped to the output and is always at least one pixel wide/high when the
// source rectangle was non-empty and the output is non-empty.
func mapRectScale(r image.Rectangle, inW, inH, outW, outH int) image.Rectangle {
	if inW <= 0 || inH <= 0 || outW <= 0 || outH <= 0 || r.Empty() {
		return image.Rectangle{}
	}
	sx := float64(outW) / float64(inW)
	sy := float64(outH) / float64(inH)

	x0 := int(math.Floor(float64(r.Min.X)*sx + 1e-9))
	y0 := int(math.Floor(float64(r.Min.Y)*sy + 1e-9))
	x1 := int(math.Ceil(float64(r.Max.X)*sx - 1e-9))
	y1 := int(math.Ceil(float64(r.Max.Y)*sy - 1e-9))

	if x0 < 0 {
		x0 = 0
	}
	if y0 < 0 {
		y0 = 0
	}
	if x1 > outW {
		x1 = outW
	}
	if y1 > outH {
		y1 = outH
	}
	if x1 <= x0 {
		x1 = x0 + 1
	}
	if y1 <= y0 {
		y1 = y0 + 1
	}
	if x1 > outW {
		x1 = outW
		x0 = outW - 1
	}
	if y1 > outH {
		y1 = outH
		y0 = outH - 1
	}
	return image.Rect(x0, y0, x1, y1)
}

// qualityGate combines the global SSIM target with any region thresholds.
// Every candidate is compared against the canonical reference image (the
// image after EXIF orientation and MaxWidth/MaxHeight resizing). Candidates
// that were downscaled further are upscaled back to the measurement size
// before comparison, so quality lost to scaling is included.
type qualityGate struct {
	ctx        context.Context
	ref        *image.NRGBA
	measRef    *image.NRGBA
	measW      int
	measH      int
	targetSSIM float64 // <= 0 disables the global threshold
	rs         *regionSet
	measure    []mappedRegion   // regions in measurement coordinates
	enforce    []enforcedRegion // disjoint enforcement cells in measurement space
}

// newQualityGate builds a gate for candidates compared against ref.
func newQualityGate(ctx context.Context, rs *regionSet, targetSSIM float64, ref *image.NRGBA) *qualityGate {
	if (rs == nil || rs.len() == 0) && targetSSIM <= 0 {
		return nil
	}
	if ctx == nil {
		ctx = context.Background()
	}

	refW, refH := ref.Bounds().Dx(), ref.Bounds().Dy()
	measW, measH := refW, refH
	measRef := ref
	if maxDim := maxInt(refW, refH); maxDim > regionMeasureMaxDim {
		scale := float64(regionMeasureMaxDim) / float64(maxDim)
		measW = maxInt(1, int(math.Round(float64(refW)*scale)))
		measH = maxInt(1, int(math.Round(float64(refH)*scale)))
		measRef = boxDownsample(ref, measW, measH)
	}

	g := &qualityGate{
		ctx:        ctx,
		ref:        ref,
		measRef:    measRef,
		measW:      measW,
		measH:      measH,
		targetSSIM: targetSSIM,
		rs:         rs,
	}
	if rs != nil {
		g.measure = rs.mappedFor(measW, measH)
		g.enforce = rs.mappedEnforcement(measW, measH)
	}
	return g
}

// evaluate measures the candidate and reports whether it satisfies the global
// and region thresholds. The candidate may have different dimensions than the
// reference (target-size downscaling); RegionResult.Rect uses the candidate's
// own coordinates. A cancelled context yields ok == false with no error, so
// callers should check ctx.Err().
func (g *qualityGate) evaluate(candidate *image.NRGBA) (float64, []RegionResult, bool) {
	if g == nil {
		return 1.0, nil, true
	}
	if g.ctx != nil && g.ctx.Err() != nil {
		return 0, nil, false
	}

	nativeW := candidate.Bounds().Dx()
	nativeH := candidate.Bounds().Dy()
	refW := g.ref.Bounds().Dx()
	refH := g.ref.Bounds().Dy()

	var scaled *image.NRGBA
	ssim := 1.0
	if nativeW == refW && nativeH == refH {
		scaled = candidate
		ssim = SSIMFast(g.ref, candidate)
	} else {
		scaled = resizeForMeasure(candidate, g.measW, g.measH)
		ssim = SSIMFast(g.measRef, scaled)
	}

	ok := g.targetSSIM <= 0 || ssim >= g.targetSSIM

	if g.rs == nil || len(g.measure) == 0 {
		return ssim, nil, ok
	}

	// Region crops are always compared in the bounded measurement space, so
	// both crops have identical pixel dimensions.
	measOut := scaled
	if scaled.Bounds().Dx() != g.measW || scaled.Bounds().Dy() != g.measH {
		measOut = resizeForMeasure(scaled, g.measW, g.measH)
	}

	// Report each configured region independently: its own rectangle at its
	// own threshold.
	results := make([]RegionResult, 0, len(g.measure))
	reportRects := g.rs.mappedFor(nativeW, nativeH)
	regionsOK := true
	for i, m := range g.measure {
		if g.ctx != nil && g.ctx.Err() != nil {
			return 0, nil, false
		}

		refCrop := cropNRGBA(g.measRef, m.rect)
		transparent := isTransparent(refCrop)

		regionSSIM := 1.0
		if !transparent {
			outCrop := cropNRGBA(measOut, m.rect)
			if refCrop.Bounds().Dx() != outCrop.Bounds().Dx() ||
				refCrop.Bounds().Dy() != outCrop.Bounds().Dy() {
				outCrop = lanczosResize(outCrop, refCrop.Bounds().Dx(), refCrop.Bounds().Dy())
			}
			regionSSIM = SSIMFast(refCrop, outCrop)
		}

		meets := transparent || m.minSSIM <= 0 || regionSSIM >= m.minSSIM
		if !meets {
			regionsOK = false
		}

		reportRect := m.rect
		if i < len(reportRects) {
			reportRect = reportRects[i].rect
		}
		results = append(results, RegionResult{
			Rect:        reportRect,
			SourceRect:  m.request,
			MinSSIM:     m.minSSIM,
			SSIM:        regionSSIM,
			ActualSSIM:  regionSSIM,
			Transparent: transparent,
			Meets:       meets,
		})
	}

	// Enforce the disjoint overlap cells at their strictest threshold. This
	// is what makes overlapping regions genuinely held to the stricter value
	// on the shared pixels, not just on each region's average.
	regionsOK = regionsOK && g.enforcementOK(measOut)
	return ssim, results, ok && regionsOK
}

// enforcementOK checks the disjoint enforcement cells.
func (g *qualityGate) enforcementOK(measOut *image.NRGBA) bool {
	for _, e := range g.enforce {
		if g.ctx != nil && g.ctx.Err() != nil {
			return false
		}
		if e.minSSIM <= 0 {
			continue
		}
		refCrop := cropNRGBA(g.measRef, e.rect)
		if isTransparent(refCrop) {
			continue
		}
		outCrop := cropNRGBA(measOut, e.rect)
		if refCrop.Bounds().Dx() != outCrop.Bounds().Dx() ||
			refCrop.Bounds().Dy() != outCrop.Bounds().Dy() {
			outCrop = lanczosResize(outCrop, refCrop.Bounds().Dx(), refCrop.Bounds().Dy())
		}
		if SSIMFast(refCrop, outCrop) < e.minSSIM {
			return false
		}
	}
	return true
}

// evaluateRegions measures an already-encoded output against the canonical
// reference. It is used for the final result report.
func evaluateRegions(ctx context.Context, rs *regionSet, targetSSIM float64, ref, candidate *image.NRGBA) (float64, []RegionResult, bool) {
	return newQualityGate(ctx, rs, targetSSIM, ref).evaluate(candidate)
}

// regionsMeet reports whether every non-transparent region that requested a
// positive threshold met it.
func regionsMeet(results []RegionResult) bool {
	for _, r := range results {
		if r.Transparent || r.MinSSIM <= 0 {
			continue
		}
		if !r.Meets {
			return false
		}
	}
	return true
}

// regionSSIMs extracts the measured SSIM values in configuration order.
func regionSSIMs(results []RegionResult) []float64 {
	if len(results) == 0 {
		return nil
	}
	out := make([]float64, len(results))
	for i, r := range results {
		out[i] = r.SSIM
	}
	return out
}

// resizeForMeasure resizes img for quality measurement: box filter when
// shrinking, Lanczos when growing.
func resizeForMeasure(img *image.NRGBA, w, h int) *image.NRGBA {
	if img.Bounds().Dx() == w && img.Bounds().Dy() == h {
		return img
	}
	if w <= img.Bounds().Dx() && h <= img.Bounds().Dy() {
		return boxDownsample(img, w, h)
	}
	return lanczosResize(img, w, h)
}

// cropNRGBA returns the pixels of img inside r. If r covers the whole image
// the original image is returned (callers only read the result).
func cropNRGBA(img *image.NRGBA, r image.Rectangle) *image.NRGBA {
	bounds := img.Bounds()
	r = r.Intersect(bounds)
	if r.Empty() {
		return image.NewNRGBA(image.Rect(0, 0, 0, 0))
	}
	if r == bounds {
		return img
	}

	w, h := r.Dx(), r.Dy()
	dst := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		srcOff := (r.Min.Y+y)*img.Stride + r.Min.X*4
		dstOff := y * dst.Stride
		copy(dst.Pix[dstOff:dstOff+w*4], img.Pix[srcOff:srcOff+w*4])
	}
	return dst
}

// isTransparent reports whether every pixel has zero alpha.
func isTransparent(img *image.NRGBA) bool {
	for i := 3; i < len(img.Pix); i += 4 {
		if img.Pix[i] != 0 {
			return false
		}
	}
	return true
}

// optionsRegionList merges the supported region option fields. Regions is the
// primary field; QualityRegions and RegionConstraints are accepted aliases.
func (o *Options) optionsRegionList() []Region {
	if len(o.QualityRegions) == 0 && len(o.RegionConstraints) == 0 {
		return o.Regions
	}
	out := make([]Region, 0, len(o.Regions)+len(o.QualityRegions)+len(o.RegionConstraints))
	out = append(out, o.Regions...)
	out = append(out, o.QualityRegions...)
	out = append(out, o.RegionConstraints...)
	return out
}

// AddRegion appends a region constraint to the options.
func (o *Options) AddRegion(rect image.Rectangle, minSSIM float64) {
	o.Regions = append(o.Regions, Region{Rect: rect, MinSSIM: minSSIM})
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// RegionResultList returns the per-region measurements, preferring
// RegionResults and falling back to the Regions alias.
func (r *Result) RegionResultList() []RegionResult {
	if len(r.RegionResults) > 0 {
		return r.RegionResults
	}
	return r.Regions
}
