from pydpmd.data import load
from pydpmd.plot import draw_particles_frame, create_animation, downsample
import matplotlib.pyplot as plt

if __name__ == "__main__":

    data = load('/home/mmccraw/dev/data/10-15-25/animations-large/data', location=['final', 'init'], load_trajectory=True, load_full=True)

    for sid in range(data.n_systems()):
        desired_frames = 1000
        steps_to_animate = downsample(data, desired_frames)

        # Define the output path
        output_path = f"animations-large/{sid}.gif"

        # Create the animation using the downsampled steps
        create_animation(
            update_func=draw_particles_frame,
            frames=steps_to_animate,
            filename=output_path,
            fps=15,  # 15 fps for smooth but not too fast animation
            dpi=300,  # Higher resolution
            bitrate=3000,  # Higher bitrate for better quality
            # Keyword arguments passed to draw_particles_frame
            data=data,
            system_id=sid,
            use_pbc=True,
            which='vertex',
            cmap_name='grey',
            id_scale=1.1
        )