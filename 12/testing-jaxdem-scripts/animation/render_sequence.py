import json
import os
import sys

# Try to import paraview.simple
try:
    from paraview.simple import *  # noqa: F401,F403
    from paraview.simple import _DisableFirstRenderCameraReset
except ModuleNotFoundError:
    sys.path.append("/usr/lib/python3/dist-packages")
    from paraview.simple import *  # noqa: F401,F403
    from paraview.simple import _DisableFirstRenderCameraReset


def _apply_view_and_box(view, dim: int, box_size, base_pixels: int, box_src=None, box_disp=None):
    if dim == 2:
        xmin, xmax = 0.0, float(box_size[0])
        ymin, ymax = 0.0, float(box_size[1])
        Lx = xmax - xmin
        Ly = ymax - ymin
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)

        view.InteractionMode = "2D"
        view.CameraParallelProjection = 1
        view.CameraParallelScale = 0.5 * Ly
        view.CameraPosition = [cx, cy, 10000.0]
        view.CameraFocalPoint = [cx, cy, 0.0]

        aspect = Lx / Ly
        if aspect >= 1.0:
            width_px = base_pixels
            height_px = int(round(base_pixels / aspect))
        else:
            height_px = base_pixels
            width_px = int(round(base_pixels * aspect))

        view.ViewSize = [width_px, height_px]
        return

    # 3D
    xmin, xmax = 0.0, float(box_size[0])
    ymin, ymax = 0.0, float(box_size[1])
    zmin, zmax = 0.0, float(box_size[2])

    Lx = xmax - xmin
    Ly = ymax - ymin
    Lz = zmax - zmin
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    cz = 0.5 * (zmin + zmax)

    if box_src is not None:
        box_src.Center = [cx, cy, cz]
        box_src.XLength = Lx
        box_src.YLength = Ly
        box_src.ZLength = Lz

    max_extent = max(Lx, Ly, Lz)
    dist = 2.0 * max_extent
    view.CameraFocalPoint = [cx, cy, cz]
    view.CameraPosition = [cx + dist, cy + dist, cz + dist]
    view.CameraViewUp = [0.0, 0.0, 1.0]

    view.ViewSize = [base_pixels, base_pixels]


def main():
    if len(sys.argv) < 2:
        print("Usage: pvbatch render_sequence.py manifest.json", file=sys.stderr)
        raise SystemExit(2)

    manifest_path = sys.argv[1]
    if not os.path.exists(manifest_path):
        print(f"Error: manifest not found: {manifest_path}", file=sys.stderr)
        raise SystemExit(2)

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    dim = int(manifest["dim"])
    base_pixels = int(manifest.get("base_pixels", 1000))
    frames = manifest["frames"]
    if not frames:
        print("Error: manifest contains 0 frames.", file=sys.stderr)
        raise SystemExit(3)

    _DisableFirstRenderCameraReset()

    # 1. Read first frame
    first_vtp = frames[0]["vtp"]
    src = XMLPolyDataReader(registrationName="data.vtp", FileName=[first_vtp])
    src.PointArrayStatus = ["Diameter", "Radius", "ParticleID"]
    src.TimeArray = "None"
    src.UpdatePipeline()

    # 2. View setup (match single-frame script behavior)
    view = GetActiveViewOrCreate("RenderView")

    try:
        view.UseColorPaletteForBackground = 0
    except AttributeError:
        pass
    view.Background = [1.0, 1.0, 1.0]
    try:
        view.Background2 = [1.0, 1.0, 1.0]
    except AttributeError:
        pass
    try:
        view.GradientBackground = 0
    except AttributeError:
        pass
    try:
        view.TexturedBackground = 0
    except AttributeError:
        pass
    view.OrientationAxesVisibility = 0

    if dim == 2:
        view.UseLight = 0
        try:
            view.LightSwitch = 0
        except AttributeError:
            pass

    # 3. Glyph (created once)
    glyph = Glyph(registrationName="Glyph1", Input=src, GlyphType="Sphere")
    glyph.GlyphMode = "All Points"
    glyph.ScaleArray = ["POINTS", "Radius"]
    glyph.ScaleFactor = 2.0
    glyph.GlyphType.ThetaResolution = 32
    glyph.GlyphType.PhiResolution = 32
    glyph.UpdatePipeline()

    glyph_disp = Show(glyph, view, "GeometryRepresentation")
    glyph_disp.Representation = "Surface"
    ColorBy(glyph_disp, ("POINTS", "ParticleID"))
    lut = GetColorTransferFunction("ParticleID")
    lut.ScalarRangeInitialized = 1.0
    glyph_disp.SetScalarBarVisibility(view, False)
    Hide(src, view)

    box_src = None
    box_disp = None
    if dim != 2:
        box_src = Box(registrationName="Box1")
        box_disp = Show(box_src, view, "GeometryRepresentation")
        box_disp.Representation = "Wireframe"
        box_disp.ColorArrayName = ["POINTS", ""]
        box_disp.LookupTable = None
        box_disp.DiffuseColor = [0.0, 0.0, 0.0]
        box_disp.AmbientColor = [0.0, 0.0, 0.0]
        box_disp.LineWidth = 1.0

    # Render loop (update reader filename; keep pipeline alive)
    n = len(frames)
    try:
        from tqdm import tqdm  # type: ignore

        iterator = tqdm(frames, total=n, desc="pvbatch render")
    except ModuleNotFoundError:
        iterator = frames

    for i, fr in enumerate(iterator):
        vtp = fr["vtp"]
        png = fr["png"]
        box = fr["box"]

        src.FileName = [vtp]
        src.UpdatePipeline()
        glyph.UpdatePipeline()

        _apply_view_and_box(view, dim=dim, box_size=box, base_pixels=base_pixels, box_src=box_src, box_disp=box_disp)
        Render(view)
        SaveScreenshot(png, view, ImageResolution=list(view.ViewSize))

        if not os.path.exists(png):
            print(f"Error: screenshot was not created: {png}", file=sys.stderr)
            raise SystemExit(4)

        # Minimal progress if tqdm isn't available
        if iterator is frames and (i == 0 or (i + 1) % 25 == 0 or (i + 1) == n):
            print(f"Rendered {i+1}/{n}", file=sys.stderr)


if __name__ == "__main__":
    main()


