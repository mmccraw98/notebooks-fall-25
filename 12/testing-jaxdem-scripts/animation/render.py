import sys
import os

# Try to import paraview.simple
try:
    from paraview.simple import *
    from paraview.simple import _DisableFirstRenderCameraReset
except ModuleNotFoundError:
    sys.path.append("/usr/lib/python3/dist-packages")
    from paraview.simple import *
    from paraview.simple import _DisableFirstRenderCameraReset


def render_scene(vtp_file, output_file, dim, box_size, base_pixels=1000):
    # Disable first render camera reset
    _DisableFirstRenderCameraReset()

    # Define box bounds based on box_size
    if dim == 2:
        xmin, xmax = 0.0, box_size[0]
        ymin, ymax = 0.0, box_size[1]
        zmin, zmax = 0.0, 0.0
    else:
        xmin, xmax = 0.0, box_size[0]
        ymin, ymax = 0.0, box_size[1]
        zmin, zmax = 0.0, box_size[2]

    # 1. Read data
    src = XMLPolyDataReader(
        registrationName="data.vtp",
        FileName=[vtp_file],
    )
    src.PointArrayStatus = ["Diameter", "Radius", "ParticleID"]
    src.TimeArray = "None"
    src.UpdatePipeline()

    # 2. Create View
    view = GetActiveViewOrCreate("RenderView")

    # White background
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

    view.OrientationAxesVisibility = 0  # Hide axes

    # Lighting
    if dim == 2:
        view.UseLight = 0
        try:
            view.LightSwitch = 0
        except AttributeError:
            pass

    # 3. Glyph
    glyph = Glyph(registrationName="Glyph1", Input=src, GlyphType="Sphere")
    glyph.GlyphMode = "All Points"
    glyph.ScaleArray = ["POINTS", "Radius"]
    glyph.ScaleFactor = 2.0
    glyph.GlyphType.ThetaResolution = 32
    glyph.GlyphType.PhiResolution = 32
    glyph.UpdatePipeline()

    # 4. Show Glyphs
    glyph_disp = Show(glyph, view, "GeometryRepresentation")
    glyph_disp.Representation = "Surface"
    ColorBy(glyph_disp, ("POINTS", "ParticleID"))
    lut = GetColorTransferFunction("ParticleID")
    lut.ScalarRangeInitialized = 1.0
    glyph_disp.SetScalarBarVisibility(view, False)

    Hide(src, view)

    # 5. Box / Camera Logic
    if dim == 2:
        # 2D Logic
        view.InteractionMode = "2D"
        view.CameraParallelProjection = 1

        Lx = xmax - xmin
        Ly = ymax - ymin
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)

        view.CameraParallelScale = 0.5 * Ly
        view.CameraPosition = [cx, cy, 10000.0]
        view.CameraFocalPoint = [cx, cy, 0.0]

        # Image Resolution based on aspect ratio
        aspect = Lx / Ly
        if aspect >= 1.0:
            width_px = base_pixels
            height_px = int(round(base_pixels / aspect))
        else:
            height_px = base_pixels
            width_px = int(round(base_pixels * aspect))

    else:
        # 3D Logic
        Lx = xmax - xmin
        Ly = ymax - ymin
        Lz = zmax - zmin
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        cz = 0.5 * (zmin + zmax)

        # Add Wireframe Box
        box = Box(registrationName="Box1")
        box.Center = [cx, cy, cz]
        box.XLength = Lx
        box.YLength = Ly
        box.ZLength = Lz

        box_disp = Show(box, view, "GeometryRepresentation")
        box_disp.Representation = "Wireframe"
        box_disp.ColorArrayName = ["POINTS", ""]
        box_disp.LookupTable = None
        box_disp.DiffuseColor = [0.0, 0.0, 0.0]
        box_disp.AmbientColor = [0.0, 0.0, 0.0]
        box_disp.LineWidth = 1.0

        # Camera
        max_extent = max(Lx, Ly, Lz)
        dist = 2.0 * max_extent
        view.CameraFocalPoint = [cx, cy, cz]
        view.CameraPosition = [cx + dist, cy + dist, cz + dist]
        view.CameraViewUp = [0.0, 0.0, 1.0]

        width_px = base_pixels
        height_px = base_pixels

    # Set View Size
    view.ViewSize = [width_px, height_px]

    Render(view)

    SaveScreenshot(output_file, view, ImageResolution=[width_px, height_px])
    print(f"Saved rendering to {output_file}")


def main():
    if len(sys.argv) < 5:
        print("Usage: pvbatch render.py input.vtp output.png dim box_size_elements...", file=sys.stderr)
        sys.exit(1)

    vtp_path = sys.argv[1]
    output_path = sys.argv[2]
    dim = int(sys.argv[3])

    # Parse box size elements from remaining arguments (up to dim count, but check remaining)
    # We expect args: script.py input.vtp output.png dim box_x box_y [box_z] [base_pixels]
    box_args = sys.argv[4 : 4 + dim]
    box_size = [float(x) for x in box_args]

    # Optional base_pixels might be after box args
    if len(sys.argv) > 4 + dim:
        base_pixels = int(sys.argv[4 + dim])
    else:
        base_pixels = 1000

    if not os.path.exists(vtp_path):
        print(f"Error: Input file {vtp_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    render_scene(vtp_path, output_path, dim, box_size, base_pixels)


if __name__ == "__main__":
    main()


