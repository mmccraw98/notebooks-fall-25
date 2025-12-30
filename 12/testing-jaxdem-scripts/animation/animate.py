import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import List

from preprocess_animation import (
    compute_sampled_frame_indices,
    get_total_frames,
    write_frame_vtp,
)


def _resolve_pvbatch() -> str:
    """
    Match the behavior of the single-frame run_render.sh:
    - Allow PVBATCH override
    - Allow PATH lookup
    - On macOS, try the ParaView.app default path
    """
    pvbatch = os.environ.get("PVBATCH", "pvbatch")
    if os.path.isfile(pvbatch) and os.access(pvbatch, os.X_OK):
        return pvbatch

    found = shutil.which(pvbatch)
    if found:
        return found

    mac_default = "/Applications/ParaView.app/Contents/bin/pvbatch"
    if os.path.isfile(mac_default) and os.access(mac_default, os.X_OK):
        return mac_default

    raise FileNotFoundError(
        "pvbatch not found. Install ParaView or set PVBATCH=/path/to/pvbatch."
    )


def _pvbatch_cmd(pvbatch: str) -> List[str]:
    # Headless Linux support:
    # - If DISPLAY is unset and xvfb-run is available, wrap pvbatch in xvfb-run.
    # - Otherwise, run pvbatch directly.
    if platform.system() == "Linux":
        if os.environ.get("DISPLAY", "") == "":
            xvfb = shutil.which("xvfb-run")
            if not xvfb:
                raise FileNotFoundError(
                    "DISPLAY is unset (headless Linux), but xvfb-run was not found. "
                    "Install it (e.g. apt-get install xvfb)."
                )
            return [xvfb, "-s", "-screen 0 1600x1600x24", pvbatch]
    return [pvbatch]


def _write_gif(png_paths: List[str], output_gif: str, fps: float) -> None:
    if not png_paths:
        raise ValueError("No PNG frames to write to GIF.")
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")

    duration = 1.0 / fps

    # Preferred: imageio
    try:
        import imageio.v2 as imageio  # type: ignore

        frames = [imageio.imread(p) for p in png_paths]
        imageio.mimsave(output_gif, frames, duration=duration, loop=0)
        return
    except ModuleNotFoundError:
        pass

    # Fallback: Pillow
    try:
        from PIL import Image  # type: ignore

        imgs = [Image.open(p).convert("RGBA") for p in png_paths]
        ms = int(round(duration * 1000.0))
        imgs[0].save(
            output_gif,
            save_all=True,
            append_images=imgs[1:],
            duration=ms,
            loop=0,
            disposal=2,
        )
        return
    except ModuleNotFoundError:
        pass

    raise ModuleNotFoundError(
        "Could not write GIF because neither imageio nor Pillow is installed. "
        "Install one of them, e.g. `pip install imageio` or `pip install pillow`."
    )


def _progress_iter(it, total: int, desc: str):
    try:
        from tqdm import tqdm  # type: ignore

        # NOTE: this function contains a `yield` in the fallback branch, so it must
        # always yield (not `return`) to avoid immediately terminating iteration.
        yield from tqdm(it, total=total, desc=desc)
    except ModuleNotFoundError:
        # Minimal fallback
        for i, x in enumerate(it, start=1):
            print(f"{desc}: {i}/{total}", file=sys.stderr)
            yield x


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render an animated GIF from an H5 trajectory using ParaView pvbatch."
    )
    parser.add_argument("input_h5", help="Path to input .h5 file")
    parser.add_argument("output_gif", help="Path to output .gif file")
    parser.add_argument("--num-frames", type=int, required=True, help="Frames to render")
    parser.add_argument("--base-pixels", type=int, default=1000, help="Base pixel size")
    parser.add_argument("--fps", type=float, default=15.0, help="GIF frames per second")
    args = parser.parse_args()

    if not os.path.exists(args.input_h5):
        print(f"Error: input H5 does not exist: {args.input_h5}", file=sys.stderr)
        return 2

    total_frames = get_total_frames(args.input_h5)
    frame_indices = compute_sampled_frame_indices(total_frames, args.num_frames)
    if frame_indices.size == 0:
        print("Error: no frames found to render.", file=sys.stderr)
        return 3

    pvbatch = _resolve_pvbatch()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    render_sequence_py = os.path.join(script_dir, "render_sequence.py")

    with tempfile.TemporaryDirectory(prefix="jaxdem_anim_") as tmpdir:
        vtps_dir = os.path.join(tmpdir, "vtp")
        pngs_dir = os.path.join(tmpdir, "png")
        os.makedirs(vtps_dir, exist_ok=True)
        os.makedirs(pngs_dir, exist_ok=True)

        png_paths: List[str] = []
        cmd_prefix = _pvbatch_cmd(pvbatch)

        # 1) Preprocess frames (H5 -> VTP), with a progress bar
        frames_manifest = []
        dim0 = None
        for out_i, t in enumerate(_progress_iter(frame_indices.tolist(), total=int(frame_indices.size), desc="Preprocess")):
            vtp_path = os.path.join(vtps_dir, f"frame_{out_i:05d}_t{t:07d}.vtp")
            png_path = os.path.join(pngs_dir, f"frame_{out_i:05d}.png")

            dim, box = write_frame_vtp(args.input_h5, int(t), vtp_path)
            if dim0 is None:
                dim0 = int(dim)
            elif int(dim) != int(dim0):
                print(f"Error: dim changed across frames ({dim0} -> {dim}).", file=sys.stderr)
                return 4

            frames_manifest.append(
                {
                    "vtp": vtp_path,
                    "png": png_path,
                    "box": box.tolist(),
                    "t": int(t),
                }
            )
            png_paths.append(png_path)

        # 2) Render all frames in ONE pvbatch session (massive speedup vs spawning pvbatch per frame)
        manifest_path = os.path.join(tmpdir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(
                {
                    "dim": int(dim0) if dim0 is not None else 2,
                    "base_pixels": int(args.base_pixels),
                    "frames": frames_manifest,
                },
                f,
            )

        cmd = cmd_prefix + [render_sequence_py, manifest_path]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            print("pvbatch failed.", file=sys.stderr)
            print("Command:", " ".join(cmd), file=sys.stderr)
            if proc.stdout:
                print("stdout:\n" + proc.stdout, file=sys.stderr)
            if proc.stderr:
                print("stderr:\n" + proc.stderr, file=sys.stderr)
            return proc.returncode

        # Ensure outputs exist
        missing = [p for p in png_paths if not os.path.exists(p)]
        if missing:
            print(f"Error: pvbatch completed but {len(missing)} PNGs are missing.", file=sys.stderr)
            print("First missing:", missing[0], file=sys.stderr)
            return 5

        _write_gif(png_paths, args.output_gif, fps=float(args.fps))

    print(f"Saved GIF to {args.output_gif}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


