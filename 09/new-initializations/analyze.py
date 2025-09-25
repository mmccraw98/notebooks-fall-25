import numpy as np
from pydpmd.data import RigidBumpy, load
from pydpmd.plot import draw_particles_frame, create_animation, downsample
import matplotlib.pyplot as plt
if __name__ == "__main__":
    new_rb_path = "/home/mmccraw/dev/data/09-09-25/new-initializations/rb_cell"
    rb = load(new_rb_path, location=["final", "init"], load_trajectory=True)
    te_total = rb.trajectory.pe_total + rb.trajectory.ke_total
    print(np.mean(np.std(te_total, axis=0) / np.mean(te_total, axis=0)))

    # draw_particles_frame(-1, plt.gca(), rb, system_id=0, which='vertex', cmap_name='viridis', location='final')
    # plt.savefig('dynamics.png')
    # plt.close()

    desired_frames = 100
    steps_to_animate = downsample(rb, desired_frames)

    # Define the output path
    output_path = "test.gif"

    # Create the animation using the downsampled steps
    create_animation(
        update_func=draw_particles_frame,
        frames=steps_to_animate,
        filename=output_path,
        fps=15,  # 15 fps for smooth but not too fast animation
        dpi=150,  # Higher resolution
        bitrate=3000,  # Higher bitrate for better quality
        # Keyword arguments passed to draw_particles_frame
        data=rb,
        system_id=0,
        use_pbc=True,
        which='vertex',
        cmap_name='viridis'
    )

    for i in range(rb.n_systems()):
        plt.plot(te_total[:, i])
        plt.plot(rb.trajectory.ke_total[:, i])
        plt.plot(rb.trajectory.pe_total[:, i])
    plt.savefig('te_total.png')
    plt.close()

    # np.save("vertex_positions.npy", rb.trajectory.vertex_pos)
    vertex_pos = np.load("vertex_positions.npy")
    # plt.plot(np.linalg.norm(rb.trajectory.pos - pos, axis=-1))
    # plt.yscale('log')
    # plt.savefig('positions.png')
    # plt.close()