package fennec

import (
	"bytes"
	"context"
	"errors"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// ── Region fixtures ─────────────────────────────────────────────────────────

// makeEdgeSubjectImage builds an image whose high-detail subject (synthetic
// "eyes/text") sits at the right edge, over a smooth gradient background that
// compresses very well. This mirrors the poster use case: the background can
// be softened, the subject cannot.
func makeEdgeSubjectImage(w, h int) *image.NRGBA {
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			off := y*img.Stride + x*4
			img.Pix[off] = uint8(30 + 40*x/w)
			img.Pix[off+1] = uint8(60 + 30*y/h)
			img.Pix[off+2] = uint8(90)
			img.Pix[off+3] = 0xff
		}
	}
	// Subject: dense checkerboard + diagonal "glyphs" near the right edge.
	// The block size scales with the image so the subject survives the
	// resolution cap used for region measurement.
	block := w / 40
	if block < 2 {
		block = 2
	}
	sx, sy := w*3/4, h/2
	for y := sy - h/6; y < sy+h/6; y++ {
		for x := sx - w/8; x < sx+w/8; x++ {
			if x < 0 || y < 0 || x >= w || y >= h {
				continue
			}
			off := y*img.Stride + x*4
			if ((x/block)+(y/block))%2 == 0 {
				img.Pix[off], img.Pix[off+1], img.Pix[off+2] = 8, 8, 8
			} else {
				img.Pix[off], img.Pix[off+1], img.Pix[off+2] = 245, 245, 245
			}
		}
	}
	return img
}

func subjectRegion(w, h int) image.Rectangle {
	return image.Rect(w*3/4-w/8, h/2-h/6, w*3/4+w/8, h/2+h/6)
}

func encodeTestJPEG(t *testing.T, img *image.NRGBA, q int) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: q}); err != nil {
		t.Fatalf("jpeg encode: %v", err)
	}
	return buf.Bytes()
}

func encodeWithEXIFOrientation(t *testing.T, img image.Image, orient uint16) []byte {
	t.Helper()
	var jbuf bytes.Buffer
	if err := jpeg.Encode(&jbuf, img, &jpeg.Options{Quality: 100}); err != nil {
		t.Fatalf("jpeg encode: %v", err)
	}
	jpegData := jbuf.Bytes()

	// Minimal little-endian TIFF with one IFD entry: Orientation (0x0112).
	tiff := []byte{
		'I', 'I', 42, 0, // byte order + magic
		8, 0, 0, 0, // IFD offset
		1, 0, // entry count
		0x12, 0x01, // tag 0x0112
		3, 0, // SHORT
		1, 0, 0, 0, // count
		byte(orient), byte(orient >> 8), 0, 0, // value
		0, 0, 0, 0, // next IFD
	}
	exif := append([]byte("Exif\x00\x00"), tiff...)
	segLen := len(exif) + 2

	var out bytes.Buffer
	out.Write(jpegData[:2]) // SOI
	out.Write([]byte{0xFF, 0xE1, byte(segLen >> 8), byte(segLen)})
	out.Write(exif)
	out.Write(jpegData[2:])
	return out.Bytes()
}

// ── Region mapping tests ────────────────────────────────────────────────────

func TestMapRectOrientation(t *testing.T) {
	// Source 100x50. The subject rectangle (10,5)-(20,15) must land at the
	// same visible subject after each orientation.
	r := image.Rect(10, 5, 20, 15)
	w, h := 100, 50

	tests := []struct {
		orient Orientation
		want   image.Rectangle
	}{
		{OrientNormal, image.Rect(10, 5, 20, 15)},
		{OrientFlipH, image.Rect(80, 5, 90, 15)},
		{OrientRotate180, image.Rect(80, 35, 90, 45)},
		{OrientFlipV, image.Rect(10, 35, 20, 45)},
		{OrientRotate90CW, image.Rect(35, 10, 45, 20)},
		{OrientRotate270CW, image.Rect(5, 80, 15, 90)},
	}
	for _, tt := range tests {
		got := mapRectOrientation(r, w, h, tt.orient)
		if got != tt.want {
			t.Fatalf("%v: mapRectOrientation(%v) = %v, want %v", tt.orient, r, got, tt.want)
		}
	}
}

func TestMapRectOrientationMatchesPixels(t *testing.T) {
	// Paint a single distinctive pixel, apply the orientation to the image and
	// to the rectangle independently, and check they agree.
	w, h := 40, 30
	marker := image.Pt(7, 11)
	src := image.NewNRGBA(image.Rect(0, 0, w, h))
	for i := 0; i < len(src.Pix); i += 4 {
		src.Pix[i+3] = 0xff
	}
	off := marker.Y*src.Stride + marker.X*4
	src.Pix[off], src.Pix[off+1], src.Pix[off+2] = 255, 0, 0

	orients := []Orientation{
		OrientFlipH, OrientRotate180, OrientFlipV, OrientTranspose,
		OrientRotate90CW, OrientTransverse, OrientRotate270CW,
	}
	for _, o := range orients {
		oriented := ApplyOrientation(src, o)
		// ApplyOrientation may return the original for Normal only.
		rect := mapRectOrientation(image.Rect(marker.X, marker.Y, marker.X+1, marker.Y+1), w, h, o)

		var markerAt image.Point
		b := oriented.Bounds()
		for y := b.Min.Y; y < b.Max.Y; y++ {
			for x := b.Min.X; x < b.Max.X; x++ {
				if oriented.Pix[y*oriented.Stride+x*4] == 255 &&
					oriented.Pix[y*oriented.Stride+x*4+1] == 0 {
					if markerAt != (image.Point{}) {
						t.Fatalf("%v: multiple markers", o)
					}
					markerAt = image.Pt(x, y)
				}
			}
		}
		if markerAt == (image.Point{}) && !image.Pt(0, 0).Eq(markerAt) {
			t.Fatalf("%v: marker pixel lost after orientation", o)
		}
		if rect != image.Rect(markerAt.X, markerAt.Y, markerAt.X+1, markerAt.Y+1) {
			t.Fatalf("%v: mapped rect %v does not equal marker cell %v", o, rect, markerAt)
		}
	}
}

func TestMapRectScale(t *testing.T) {
	r := image.Rect(50, 25, 100, 75) // quarter of a 200x100 image
	got := mapRectScale(r, 200, 100, 100, 50)
	want := image.Rect(25, 12, 50, 38)
	if got != want {
		t.Fatalf("mapRectScale = %v, want %v", got, want)
	}

	// A tiny region must not collapse to nothing when downscaled.
	tiny := image.Rect(3, 3, 4, 4)
	got = mapRectScale(tiny, 1000, 1000, 10, 10)
	if got.Empty() || got.Dx() < 1 || got.Dy() < 1 {
		t.Fatalf("tiny region collapsed: %v", got)
	}
}

func TestRegionSetValidation(t *testing.T) {
	t.Run("empty_rect", func(t *testing.T) {
		_, err := newRegionSet([]Region{{Rect: image.Rect(5, 5, 5, 10), MinSSIM: 0.9}}, 100, 100, OrientNormal, true)
		if !errors.Is(err, ErrInvalidRegion) {
			t.Fatalf("expected ErrInvalidRegion, got %v", err)
		}
	})

	t.Run("bad_ssim", func(t *testing.T) {
		_, err := newRegionSet([]Region{{Rect: image.Rect(0, 0, 10, 10), MinSSIM: 1.5}}, 100, 100, OrientNormal, true)
		if !errors.Is(err, ErrInvalidRegion) {
			t.Fatalf("expected ErrInvalidRegion, got %v", err)
		}
		_, err = newRegionSet([]Region{{Rect: image.Rect(0, 0, 10, 10), MinSSIM: -0.1}}, 100, 100, OrientNormal, true)
		if !errors.Is(err, ErrInvalidRegion) {
			t.Fatalf("expected ErrInvalidRegion for negative SSIM, got %v", err)
		}
	})

	t.Run("out_of_bounds", func(t *testing.T) {
		_, err := newRegionSet([]Region{{Rect: image.Rect(200, 200, 300, 300), MinSSIM: 0.9}}, 100, 100, OrientNormal, true)
		if !errors.Is(err, ErrRegionOutOfBounds) {
			t.Fatalf("expected ErrRegionOutOfBounds, got %v", err)
		}
	})

	t.Run("partial_clamped", func(t *testing.T) {
		rs, err := newRegionSet([]Region{{Rect: image.Rect(90, 90, 300, 300), MinSSIM: 0.9}}, 100, 100, OrientNormal, true)
		if err != nil {
			t.Fatalf("partial region should be clamped, got %v", err)
		}
		if got := rs.targets[0].request; got != image.Rect(90, 90, 100, 100) {
			t.Fatalf("clamped rect = %v", got)
		}
	})

	t.Run("shorthand", func(t *testing.T) {
		rs, err := newRegionSet([]Region{{X: 10, Y: 20, Width: 30, Height: 40, MinSSIM: 0.9}}, 100, 100, OrientNormal, true)
		if err != nil {
			t.Fatal(err)
		}
		if got := rs.targets[0].request; got != image.Rect(10, 20, 40, 60) {
			t.Fatalf("shorthand rect = %v", got)
		}
	})
}

// ── Compression tests ───────────────────────────────────────────────────────

func TestRegionsUnconfiguredPreservesOutput(t *testing.T) {
	img := makeTestImage(200, 200)

	opts := DefaultOptions()
	opts.Format = JPEG
	without, err := CompressImage(ctx(), img, opts)
	if err != nil {
		t.Fatal(err)
	}

	opts.Regions = nil
	again, err := CompressImage(ctx(), img, opts)
	if err != nil {
		t.Fatal(err)
	}

	if without.SSIM != again.SSIM || without.JPEGQuality != again.JPEGQuality ||
		!bytes.Equal(without.CompressedData, again.CompressedData) {
		t.Fatal("unconfigured regions must not change output")
	}
	if without.RegionResults != nil || without.RegionSSIM != nil {
		t.Fatal("unconfigured regions should report no region data")
	}
}

func TestCompressRegionQualityEdgeSubject(t *testing.T) {
	const w, h = 400, 300
	img := makeEdgeSubjectImage(w, h)
	region := subjectRegion(w, h)

	// Baseline: a low whole-image target lets the edge subject soften.
	baseOpts := DefaultOptions()
	baseOpts.Format = JPEG
	baseOpts.TargetSSIM = 0.80
	base, err := CompressImage(ctx(), img, baseOpts)
	if err != nil {
		t.Fatal(err)
	}
	baseDecoded := decodeImageFromBytes(base.CompressedData)
	baseRegionSSIM := SSIMFast(cropNRGBA(img, region), cropNRGBA(baseDecoded, region))
	if baseRegionSSIM >= 0.95 {
		t.Skipf("fixture does not degrade at low quality (region SSIM %.4f)", baseRegionSSIM)
	}

	// With a region constraint the subject must stay sharp even though the
	// whole-image budget is loose.
	opts := DefaultOptions()
	opts.Format = JPEG
	opts.TargetSSIM = 0.80
	opts.Regions = []Region{{Rect: region, MinSSIM: 0.95}}

	res, err := CompressImage(ctx(), img, opts)
	if err != nil {
		t.Fatalf("CompressImage with region: %v", err)
	}

	if len(res.RegionResults) != 1 {
		t.Fatalf("expected 1 region result, got %d", len(res.RegionResults))
	}
	rr := res.RegionResults[0]
	if rr.MinSSIM != 0.95 {
		t.Fatalf("MinSSIM = %f", rr.MinSSIM)
	}
	if rr.SSIM < 0.95 {
		t.Fatalf("region SSIM %.4f below requested 0.95", rr.SSIM)
	}
	if !rr.Meets {
		t.Fatal("region should be reported as meeting its threshold")
	}
	// The mapped rect must stay on the subject (right edge), not the frame.
	if rr.Rect.Min.X < w/2 {
		t.Fatalf("region mapped to the wrong side: %v", rr.Rect)
	}
	if len(res.RegionSSIM) != 1 || math.Abs(res.RegionSSIM[0]-rr.SSIM) > 1e-9 {
		t.Fatalf("RegionSSIM mismatch: %v vs %v", res.RegionSSIM, rr.SSIM)
	}
	if res.CompressedSize != int64(len(res.CompressedData)) || res.CompressedSize == 0 {
		t.Fatal("result must report the actual file size")
	}
	if res.SSIM < 0.80 {
		t.Fatalf("whole-image SSIM %.4f below its 0.80 target", res.SSIM)
	}

	// Verify against an independent decode of the written bytes.
	decoded := decodeImageFromBytes(res.CompressedData)
	if decoded == nil {
		t.Fatal("output is not a decodable image")
	}
	measured := SSIMFast(cropNRGBA(img, region), cropNRGBA(decoded, region))
	if measured < 0.95 {
		t.Fatalf("independently measured region SSIM %.4f below target", measured)
	}
	if measured <= baseRegionSSIM {
		t.Fatalf("constrained region SSIM %.4f should beat baseline %.4f", measured, baseRegionSSIM)
	}
	t.Logf("baseline region SSIM %.4f (%d B) vs constrained %.4f (%d B)",
		baseRegionSSIM, len(base.CompressedData), measured, len(res.CompressedData))

	// Background quality: the constrained run may spend more bits overall,
	// but the constraint must be what drives it — the subject improves while
	// the smooth background remains visually intact.
	bg := image.Rect(0, 0, w/2, h)
	bgSSIM := SSIMFast(cropNRGBA(img, bg), cropNRGBA(decoded, bg))
	if bgSSIM < 0.90 {
		t.Fatalf("background SSIM %.4f degraded too far", bgSSIM)
	}
	t.Logf("background SSIM around the edge subject: %.4f", bgSSIM)
}

func TestQualityGateCancellation(t *testing.T) {
	const w, h = 300, 200
	img := makeEdgeSubjectImage(w, h)
	rs, err := newRegionSet([]Region{{Rect: subjectRegion(w, h), MinSSIM: 0.9}},
		w, h, OrientNormal, true)
	if err != nil {
		t.Fatal(err)
	}

	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	gate := newQualityGate(cancelled, rs, 0.9, img)

	_, _, ok := gate.evaluate(img)
	if ok {
		t.Fatal("cancelled gate must not accept a candidate")
	}
}

func TestCompressMultipleRegionsAndOverlap(t *testing.T) {
	const w, h = 320, 240
	img := makeEdgeSubjectImage(w, h)
	subject := subjectRegion(w, h)

	// Region A covers the subject loosely, region B covers a nested strip with
	// a stricter threshold. Both must be enforced.
	strict := image.Rect(subject.Min.X+10, subject.Min.Y+10, subject.Max.X-10, subject.Max.Y-10)

	opts := DefaultOptions()
	opts.Format = JPEG
	opts.Regions = []Region{
		{Rect: subject, MinSSIM: 0.75},
		{Rect: strict, MinSSIM: 0.92},
	}
	res, err := CompressImage(ctx(), img, opts)
	if err != nil {
		t.Fatalf("CompressImage: %v", err)
	}
	if len(res.RegionResults) != 2 {
		t.Fatalf("expected 2 region results, got %d", len(res.RegionResults))
	}
	for i, rr := range res.RegionResults {
		if rr.SSIM < rr.MinSSIM {
			t.Fatalf("region %d SSIM %.4f < %.4f", i, rr.SSIM, rr.MinSSIM)
		}
		if !rr.Meets {
			t.Fatalf("region %d should meet threshold", i)
		}
	}
	if res.RegionSSIM[1] < 0.92 {
		t.Fatalf("strict nested region SSIM %.4f below 0.92", res.RegionSSIM[1])
	}
}

func TestCompressRegionQualityUnreachable(t *testing.T) {
	img := makeEdgeSubjectImage(400, 300)

	// SSIM 1.0 on JPEG is unreachable; a lossy region must fail explicitly
	// instead of silently writing a below-threshold image.
	opts := DefaultOptions()
	opts.Format = JPEG
	opts.Regions = []Region{{Rect: subjectRegion(400, 300), MinSSIM: 1.0}}

	_, err := CompressImage(ctx(), img, opts)
	if err == nil {
		t.Fatal("expected unreachable quality error")
	}
	if !errors.Is(err, ErrQualityTargetUnreachable) {
		t.Fatalf("expected ErrQualityTargetUnreachable, got %v", err)
	}
	var qerr *QualityError
	if !errors.As(err, &qerr) {
		t.Fatalf("expected *QualityError, got %T", err)
	}
	if len(qerr.Regions) == 0 {
		t.Fatal("QualityError should report best region measurements")
	}
}

func TestCompressRegionAutoOrient(t *testing.T) {
	// Stored pixel grid is w x h; EXIF says it must be rotated 90° CW for
	// display, so the displayed image is h x w.
	const w, h = 240, 160
	stored := makeEdgeSubjectImage(w, h)
	subjectStored := subjectRegion(w, h)

	data := encodeWithEXIFOrientation(t, stored, uint16(OrientRotate90CW))
	tmpDir := t.TempDir()
	srcPath := filepath.Join(tmpDir, "rotated.jpg")
	if err := os.WriteFile(srcPath, data, 0644); err != nil {
		t.Fatal(err)
	}

	// Orient 1: default auto-orient, region in stored coordinates. Fennec must
	// rotate it into display space automatically.
	opts := DefaultOptions()
	opts.Format = JPEG
	opts.Regions = []Region{{Rect: subjectStored, MinSSIM: 0.88}}
	res, err := CompressFile(ctx(), srcPath, filepath.Join(tmpDir, "out.jpg"), opts)
	if err != nil {
		t.Fatalf("CompressFile: %v", err)
	}
	if res.FinalDimensions != image.Pt(h, w) {
		t.Fatalf("expected displayed %dx%d, got %v", h, w, res.FinalDimensions)
	}
	rr := res.RegionResults[0]
	if rr.Rect.Max.X > h || rr.Rect.Max.Y > w {
		t.Fatalf("rotated region out of bounds: %v in %dx%d", rr.Rect, h, w)
	}
	// The subject sits at the right edge of stored space; 90° CW rotation
	// moves it to the bottom edge of display space.
	if rr.Rect.Min.Y < w/2 {
		t.Fatalf("rotated region should be near the bottom: %v", rr.Rect)
	}
	if rr.SSIM < rr.MinSSIM {
		t.Fatalf("rotated region SSIM %.4f < %.4f", rr.SSIM, rr.MinSSIM)
	}

	// Orient 2: disable auto-orient. The same stored-space rectangle must be
	// used as-is and the output must not be rotated.
	opts2 := DefaultOptions()
	opts2.Format = JPEG
	opts2.AutoOrient = false
	opts2.Regions = []Region{{Rect: subjectStored, MinSSIM: 0.88}}
	res2, err := CompressFile(ctx(), srcPath, filepath.Join(tmpDir, "out_raw.jpg"), opts2)
	if err != nil {
		t.Fatalf("CompressBytes no-orient: %v", err)
	}
	if res2.FinalDimensions != image.Pt(w, h) {
		t.Fatalf("expected raw %dx%d, got %v", w, h, res2.FinalDimensions)
	}
	if res2.RegionResults[0].Rect != subjectStored {
		t.Fatalf("no-orient region changed: %v", res2.RegionResults[0].Rect)
	}
}

func TestCompressRegionTransparent(t *testing.T) {
	// A region covering only fully transparent pixels cannot be meaningfully
	// measured; it is reported as transparent and does not fail the encode.
	img := makeSolidImage(200, 200, color.NRGBA{0, 0, 0, 0})
	opts := DefaultOptions()
	opts.Format = PNG
	opts.Regions = []Region{{Rect: image.Rect(0, 0, 100, 100), MinSSIM: 0.95}}

	res, err := CompressImage(ctx(), img, opts)
	if err != nil {
		t.Fatalf("CompressImage: %v", err)
	}
	if len(res.RegionResults) != 1 {
		t.Fatalf("expected 1 region result, got %d", len(res.RegionResults))
	}
	rr := res.RegionResults[0]
	if !rr.Transparent {
		t.Fatal("region should be reported as transparent")
	}
	if !rr.Meets {
		t.Fatal("transparent region should not fail the constraint")
	}
}

func TestCompressRegionWithMaxDimensions(t *testing.T) {
	const w, h = 800, 600
	img := makeEdgeSubjectImage(w, h)
	subject := subjectRegion(w, h)

	opts := DefaultOptions()
	opts.Format = JPEG
	opts.MaxWidth, opts.MaxHeight = 400, 400
	opts.Regions = []Region{{Rect: subject, MinSSIM: 0.90}}

	res, err := CompressImage(ctx(), img, opts)
	if err != nil {
		t.Fatalf("CompressImage: %v", err)
	}
	if res.FinalDimensions.X > 400 || res.FinalDimensions.Y > 400 {
		t.Fatalf("resize not applied: %v", res.FinalDimensions)
	}
	if len(res.RegionResults) != 1 {
		t.Fatalf("expected 1 region result")
	}
	rr := res.RegionResults[0]
	// Region must be scaled into the 400x300 output.
	if rr.Rect.Max.X > 400 || rr.Rect.Max.Y > 300 {
		t.Fatalf("region not scaled with image: %v", rr.Rect)
	}
	if rr.SSIM < 0.90 {
		t.Fatalf("region SSIM %.4f < 0.90 after resize", rr.SSIM)
	}
}

func TestCompressRegionTargetSize(t *testing.T) {
	const w, h = 400, 300
	img := makeEdgeSubjectImage(w, h)
	region := subjectRegion(w, h)

	opts := DefaultOptions()
	opts.Format = JPEG
	opts.TargetSize = 24 * 1024
	opts.Regions = []Region{{Rect: region, MinSSIM: 0.90}}

	res, err := CompressImage(ctx(), img, opts)
	if err != nil {
		t.Fatalf("CompressImage: %v", err)
	}
	if res.CompressedSize > int64(opts.TargetSize) {
		t.Fatalf("size %d exceeds target %d", res.CompressedSize, opts.TargetSize)
	}
	if res.CompressedSize != int64(len(res.CompressedData)) {
		t.Fatal("reported size must match data")
	}
	if len(res.RegionResults) != 1 {
		t.Fatalf("expected 1 region result, got %d", len(res.RegionResults))
	}
	if res.RegionResults[0].SSIM < 0.90 {
		t.Fatalf("region SSIM %.4f < 0.90", res.RegionResults[0].SSIM)
	}

	decoded := decodeImageFromBytes(res.CompressedData)
	if decoded == nil {
		t.Fatal("output not decodable")
	}
	if got := len(res.CompressedData); got == 0 {
		t.Fatal("empty output")
	}
	// Verify independently on the final size.
	refCrop := cropNRGBA(img, region)
	outCrop := cropNRGBA(decoded, mapRectScale(region, w, h, decoded.Bounds().Dx(), decoded.Bounds().Dy()))
	if outCrop.Bounds().Dx() != refCrop.Bounds().Dx() {
		outCrop = lanczosResize(outCrop, refCrop.Bounds().Dx(), refCrop.Bounds().Dy())
	}
	if s := SSIMFast(refCrop, outCrop); s < 0.88 {
		t.Fatalf("independently measured region SSIM %.4f", s)
	}
}

func TestCompressRegionImpossibleTargets(t *testing.T) {
	const w, h = 400, 300
	img := makeEdgeSubjectImage(w, h)

	// A subject that cannot survive JPEG compression at 0.99 within 1 KB.
	opts := DefaultOptions()
	opts.Format = JPEG
	opts.TargetSize = 1024
	opts.Regions = []Region{{Rect: subjectRegion(w, h), MinSSIM: 0.99}}

	_, err := CompressImage(ctx(), img, opts)
	if err == nil {
		t.Fatal("expected impossible target error")
	}
	if !errors.Is(err, ErrImpossibleTargets) {
		t.Fatalf("expected ErrImpossibleTargets, got %v", err)
	}
	var qerr *QualityError
	if !errors.As(err, &qerr) {
		t.Fatalf("expected *QualityError, got %T", err)
	}
	if qerr.TargetSize != 1024 {
		t.Fatalf("error should report target size, got %d", qerr.TargetSize)
	}
	if qerr.Reason == "" {
		t.Fatal("error should explain the conflict")
	}
}

func TestCompressRegionTargetSizeQualityConflict(t *testing.T) {
	const w, h = 400, 300
	img := makeEdgeSubjectImage(w, h)

	// Regions are satisfiable at high quality, but not within 8 KB when the
	// whole-image SSIM target is also demanding.
	opts := DefaultOptions()
	opts.Format = JPEG
	opts.TargetSize = 3 * 1024
	opts.TargetSSIM = 0.99
	opts.Regions = []Region{{Rect: subjectRegion(w, h), MinSSIM: 0.95}}

	_, err := CompressImage(ctx(), img, opts)
	if !errors.Is(err, ErrImpossibleTargets) {
		t.Fatalf("expected ErrImpossibleTargets, got %v", err)
	}
	var qerr *QualityError
	if !errors.As(err, &qerr) {
		t.Fatalf("expected *QualityError, got %T", err)
	}
	if qerr.Achieved <= 0 {
		t.Fatal("error should report the best measured SSIM")
	}
	if len(qerr.Regions) != 1 {
		t.Fatalf("error should report region measurements, got %d", len(qerr.Regions))
	}
}

func TestCompressRegionInvalidReturnsErrors(t *testing.T) {
	img := makeTestImage(100, 100)

	t.Run("empty", func(t *testing.T) {
		opts := DefaultOptions()
		opts.Format = JPEG
		opts.Regions = []Region{{Rect: image.Rect(10, 10, 10, 20), MinSSIM: 0.9}}
		_, err := CompressImage(ctx(), img, opts)
		if !errors.Is(err, ErrInvalidRegion) {
			t.Fatalf("expected ErrInvalidRegion, got %v", err)
		}
	})

	t.Run("out_of_bounds", func(t *testing.T) {
		opts := DefaultOptions()
		opts.Format = JPEG
		opts.Regions = []Region{{Rect: image.Rect(500, 500, 600, 600), MinSSIM: 0.9}}
		_, err := CompressImage(ctx(), img, opts)
		if !errors.Is(err, ErrRegionOutOfBounds) {
			t.Fatalf("expected ErrRegionOutOfBounds, got %v", err)
		}
	})

	t.Run("validate", func(t *testing.T) {
		opts := DefaultOptions()
		opts.Regions = []Region{{Rect: image.Rect(0, 0, 0, 0), MinSSIM: 0.9}}
		if err := opts.Validate(); !errors.Is(err, ErrInvalidRegion) {
			t.Fatalf("Validate should reject empty region, got %v", err)
		}
	})
}

func TestCompressRegionLargeImage(t *testing.T) {
	// Larger than the internal measurement cap (1024 px). The region must
	// still be measured and enforced without panicking or timing out.
	const w, h = 1600, 1200
	img := makeEdgeSubjectImage(w, h)
	region := subjectRegion(w, h)

	opts := DefaultOptions()
	opts.Format = JPEG
	opts.Quality = Aggressive
	opts.Regions = []Region{{Rect: region, MinSSIM: 0.80}}

	res, err := CompressImage(ctx(), img, opts)
	if err != nil {
		t.Fatalf("CompressImage: %v", err)
	}
	if len(res.RegionResults) != 1 {
		t.Fatalf("expected 1 region result")
	}
	if res.RegionResults[0].SSIM < 0.80 {
		t.Fatalf("region SSIM %.4f < 0.80", res.RegionResults[0].SSIM)
	}
}

func TestCompressRegionCancelled(t *testing.T) {
	img := makeEdgeSubjectImage(1200, 900)
	opts := DefaultOptions()
	opts.Format = JPEG
	opts.Regions = []Region{{Rect: subjectRegion(1200, 900), MinSSIM: 0.99}}

	cancelled, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := CompressImage(cancelled, img, opts)
	if err == nil {
		t.Fatal("expected cancellation error")
	}
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
}

func TestCompressRegionCancelledTargetSize(t *testing.T) {
	img := makeEdgeSubjectImage(1200, 900)
	opts := DefaultOptions()
	opts.Format = JPEG
	opts.TargetSize = 20 * 1024
	opts.Regions = []Region{{Rect: subjectRegion(1200, 900), MinSSIM: 0.9}}

	cancelled, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := CompressImage(cancelled, img, opts)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
}

func TestEncodeRegionConstraints(t *testing.T) {
	img := makeEdgeSubjectImage(300, 200)
	opts := DefaultOptions()
	opts.Regions = []Region{{Rect: subjectRegion(300, 200), MinSSIM: 1.0}}

	var buf bytes.Buffer
	err := Encode(&buf, img, JPEG, opts)
	if !errors.Is(err, ErrQualityTargetUnreachable) {
		t.Fatalf("expected ErrQualityTargetUnreachable from Encode, got %v", err)
	}
}

func TestCompressFileRegionConstraint(t *testing.T) {
	tmpDir := t.TempDir()
	const w, h = 400, 300
	img := makeEdgeSubjectImage(w, h)
	srcPath := filepath.Join(tmpDir, "poster.jpg")
	if err := os.WriteFile(srcPath, encodeTestJPEG(t, img, 98), 0644); err != nil {
		t.Fatal(err)
	}
	dstPath := filepath.Join(tmpDir, "poster_out.jpg")

	opts := DefaultOptions()
	opts.Format = JPEG
	opts.Regions = []Region{{Rect: subjectRegion(w, h), MinSSIM: 0.9}}

	res, err := CompressFile(context.Background(), srcPath, dstPath, opts)
	if err != nil {
		t.Fatalf("CompressFile: %v", err)
	}
	info, err := os.Stat(dstPath)
	if err != nil {
		t.Fatal(err)
	}
	if info.Size() != res.CompressedSize {
		t.Fatalf("file size %d != reported %d", info.Size(), res.CompressedSize)
	}

	out, err := Open(dstPath)
	if err != nil {
		t.Fatal(err)
	}
	decoded := toNRGBARef(out)
	refCrop := cropNRGBA(toNRGBARef(img), subjectRegion(w, h))
	outCrop := cropNRGBA(decoded, subjectRegion(w, h))
	if s := SSIMFast(refCrop, outCrop); s < 0.9 {
		t.Fatalf("file region SSIM %.4f < 0.9", s)
	}
	if res.String() == "" || !bytes.Contains([]byte(res.String()), []byte("Regions")) {
		t.Fatalf("Result.String should include region report: %s", res.String())
	}
}

func TestCompressPNGRegionsReportLossless(t *testing.T) {
	img := makeEdgeSubjectImage(200, 160)
	opts := DefaultOptions()
	opts.Format = PNG
	opts.Regions = []Region{{Rect: subjectRegion(200, 160), MinSSIM: 0.99}}

	res, err := CompressImage(ctx(), img, opts)
	if err != nil {
		t.Fatalf("CompressImage PNG: %v", err)
	}
	if len(res.RegionResults) != 1 {
		t.Fatalf("expected region results for PNG, got %d", len(res.RegionResults))
	}
	if res.RegionResults[0].SSIM < 0.999 {
		t.Fatalf("PNG region SSIM should be ~1.0, got %f", res.RegionResults[0].SSIM)
	}
	var buf bytes.Buffer
	if _, err := png.Decode(bytes.NewReader(res.CompressedData)); err != nil {
		t.Fatalf("PNG output invalid: %v", err)
	}
	_ = buf
}

func TestDisjointEnforcement(t *testing.T) {
	regions := []Region{
		{Rect: image.Rect(0, 0, 100, 100), MinSSIM: 0.90},
		{Rect: image.Rect(50, 50, 150, 150), MinSSIM: 0.97},
	}
	rs, err := newRegionSet(regions, 200, 200, OrientNormal, true)
	if err != nil {
		t.Fatal(err)
	}

	// Cells inside the overlap must use 0.97; cells covered only by the first
	// region must use 0.90; cells covered only by the second use 0.97.
	overlap := image.Rect(50, 50, 100, 100)
	var sawOverlap, sawFirstOnly, sawSecondOnly bool
	for _, e := range rs.enforce {
		switch {
		case e.rect.In(overlap):
			sawOverlap = true
			if e.minSSIM != 0.97 {
				t.Fatalf("overlap cell %v enforced at %.2f, want 0.97", e.rect, e.minSSIM)
			}
		case e.rect.In(regions[0].Rect):
			sawFirstOnly = true
			if e.minSSIM != 0.90 {
				t.Fatalf("first-region cell %v enforced at %.2f, want 0.90", e.rect, e.minSSIM)
			}
		case e.rect.In(regions[1].Rect):
			sawSecondOnly = true
			if e.minSSIM != 0.97 {
				t.Fatalf("second-region cell %v enforced at %.2f, want 0.97", e.rect, e.minSSIM)
			}
		default:
			t.Fatalf("enforcement cell %v outside all regions", e.rect)
		}
	}
	if !sawOverlap || !sawFirstOnly || !sawSecondOnly {
		t.Fatalf("missing enforcement cells: overlap=%v first=%v second=%v cells=%v",
			sawOverlap, sawFirstOnly, sawSecondOnly, rs.enforce)
	}
}

func TestOverlapStrictestThresholdRejects(t *testing.T) {
	// Two regions overlap on a high-detail area. The overlap's strict
	// threshold must gate the search even though each region's average could
	// pass on its own.
	const w, h = 320, 240
	img := makeEdgeSubjectImage(w, h)
	sub := subjectRegion(w, h)

	loose := image.Rect(sub.Min.X-40, sub.Min.Y-40, sub.Max.X, sub.Max.Y)
	strict := image.Rect(sub.Min.X, sub.Min.Y, sub.Max.X+40, sub.Max.Y+40)

	opts := DefaultOptions()
	opts.Format = JPEG
	opts.Regions = []Region{
		{Rect: loose, MinSSIM: 0.60},
		{Rect: strict, MinSSIM: 0.985},
	}
	res, err := CompressImage(ctx(), img, opts)
	if err != nil {
		t.Fatalf("CompressImage: %v", err)
	}
	// Both reported regions must satisfy their own thresholds, and the
	// overlap cell must satisfy the stricter one.
	decoded := decodeImageFromBytes(res.CompressedData)
	rs, err := newRegionSet(opts.Regions, w, h, OrientNormal, true)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range rs.enforce {
		if e.minSSIM < 0.985 {
			continue
		}
		got := SSIMFast(cropNRGBA(img, e.rect), cropNRGBA(decoded, e.rect))
		if got < e.minSSIM-0.01 {
			t.Fatalf("overlap cell %v SSIM %.4f below strict %.3f", e.rect, got, e.minSSIM)
		}
	}
}

func TestRegionsUnconfiguredPreservesTargetSizeOutput(t *testing.T) {
	img := makeEdgeSubjectImage(320, 240)

	opts := DefaultOptions()
	opts.Format = JPEG
	opts.TargetSize = 12 * 1024
	without, err := CompressImage(ctx(), img, opts)
	if err != nil {
		t.Fatal(err)
	}

	opts.Regions = []Region{}
	again, err := CompressImage(ctx(), img, opts)
	if err != nil {
		t.Fatal(err)
	}
	if without.CompressedSize != again.CompressedSize || without.SSIM != again.SSIM ||
		!bytes.Equal(without.CompressedData, again.CompressedData) {
		t.Fatalf("empty region list changed target-size output: %d/%d bytes", without.CompressedSize, again.CompressedSize)
	}
}

func TestCompressBatchRegionConstraints(t *testing.T) {
	dir := t.TempDir()
	const w, h = 240, 180
	img := makeEdgeSubjectImage(w, h)
	src := filepath.Join(dir, "poster.jpg")
	if err := os.WriteFile(src, encodeTestJPEG(t, img, 97), 0644); err != nil {
		t.Fatal(err)
	}

	opts := DefaultOptions()
	opts.Format = JPEG
	opts.Regions = []Region{{Rect: subjectRegion(w, h), MinSSIM: 0.90}}

	results := CompressBatch(ctx(), []BatchItem{{
		Src:  src,
		Dst:  filepath.Join(dir, "out.jpg"),
		Opts: &opts,
	}}, BatchOptions{Workers: 1, DefaultOpts: opts})

	if results[0].Err != nil {
		t.Fatalf("batch failed: %v", results[0].Err)
	}
	if len(results[0].Result.RegionResultList()) != 1 {
		t.Fatal("batch result should carry region reporting")
	}
}
