import pydpmd as md
import numpy as np
import os
from system_building_resources import *
import time
import shutil
from correlation_functions import compute_shear_modulus

if __name__ == "__main__":
    for i in range(1):
        root = f"/home/mmccraw/dev/data/10-02-25/disk-shear-modulus-small/trial-{i}/"  # run again at 1k particles
        if not os.path.exists(root):
            os.makedirs(root)

        radii = generate_bidisperse_radii(100, 0.5, 1.4)
        which = 'small'
        packing_fraction = 0.8
        phi_increment = 1e-3
        temperature = 1e-6
        n_steps = 1e5
        save_freq = 1e0
        pressure_target = 1e-3
        packing_fraction_target = 0.85
        dt = 2e-2

        # build the initial data and equilibrate it, ideally to a 0-overlap state

        n_duplicates = 10
        
        temp_path = tempfile.mkdtemp()
        # build the initial data and equilibrate it, ideally to a 0-overlap state
        disk = join_systems([build_disk_system_from_radii(radii, which, packing_fraction, int(np.random.randint(0, 1e9))) for _ in range(n_duplicates)])
        disk.set_neighbor_method(NeighborMethod.Cell)
        set_standard_cell_list_parameters(disk, 0.3)
        disk.save(temp_path)
        subprocess.run([
            os.path.join("/home/mmccraw/dev/dpmd/build/", "disk_equilibrate_pbc"),
            temp_path,
            temp_path,
        ], check=True)

        disk = load(temp_path, location=["final", "init"])
        shutil.rmtree(temp_path)

        init_path = os.path.join(root, "init")
        disk.set_neighbor_method(NeighborMethod.Cell)
        set_standard_cell_list_parameters(disk, 0.3)
        disk.save(init_path)
        
        # run the iterative compress-dynamics protocol
        file_index = 0
        while True:
            # compress the data, incrementing the packing fraction by phi_increment and maintaining the temperature
            disk = load(init_path, location=["final", "init"])
            shutil.rmtree(init_path)
            
            compression_path = os.path.join(root, f"compression_{file_index}")
            disk.set_velocities(temperature, np.random.randint(0, 1e9))
            disk.set_neighbor_method(NeighborMethod.Cell)
            set_standard_cell_list_parameters(disk, 0.3)
            disk.save(compression_path)
            subprocess.run([
                os.path.join("/home/mmccraw/dev/dpmd/build/", "nvt_rescale_disk_pbc_compress"),
                compression_path,
                compression_path,
                str(int(max(int(n_steps / 10), 1e3) / 2)),  # total number of steps - half are used for compression, the other half are used for equilibration
                str(phi_increment),
                str(temperature),
                str(dt),
            ], check=True)

            # run nve dynamics from the output of the compression data
            disk = load(compression_path, location=["final", "init"])
            dynamics_data_path = os.path.join(root, f"dynamics_{file_index}")
            disk.set_velocities(temperature, np.random.randint(0, 1e9))
            disk.set_neighbor_method(NeighborMethod.Cell)
            set_standard_cell_list_parameters(disk, 0.3)
            disk.save(dynamics_data_path)
            subprocess.run([
                os.path.join("/home/mmccraw/dev/dpmd/build/", "nve_disk_pbc_final"),
                dynamics_data_path,
                dynamics_data_path,
                str(n_steps),  # total number of dynamics steps
                str(save_freq),  # save frequency
                str(dt),
            ], check=True)

            # # calculate the average overlap in each system
            # disk = load(dynamics_data_path, location=["final", "init"], load_trajectory=True, load_full=False)
            # overlap_path = os.path.join(root, f"overlap_{file_index}.npz")
            # first column stores the overlap for vertex i, second column stores the number of overlaps for vertex i
            # mean_overlaps = np.sum(  # sum over all frames
            #     np.array(
            #         [
            #             np.add.reduceat(  # sum over all vertices within each system
            #                 disk.trajectory[i].overlaps, disk.system_offset[:-1]
            #             , axis=0)
            #             for i in range(disk.trajectory.num_frames())
            #         ]
            # ), axis=0)
            # divide the total number of overlaps by the number of overlapping vertices
            # repeat for all systems
            # mean_overlaps = mean_overlaps[:, 0] / mean_overlaps[:, 1]
            # np.savez(
            #     overlap_path,
            #     overlap=mean_overlaps,
            #     packing_fraction=disk.init.packing_fraction,
            # )

            # calculate the shear modulus and save it with packing_fraction, and temperature
            disk = load(dynamics_data_path, location=["final", "init"], load_trajectory=True, load_full=False)
            shear_modulus_path = os.path.join(root, f"shear_modulus_{file_index}.npz")
            shear_modulus, t = compute_shear_modulus(disk, None)
            np.savez(
                shear_modulus_path,
                shear_modulus=shear_modulus,
                t=t,
                packing_fraction=disk.init.packing_fraction,
            )

            # load the data, split into separate systems, recombining only those that have a packing fraction less than the target
            disk = load(dynamics_data_path, location=["final", "init"], load_trajectory=True, load_full=False)
            remaining_systems = [
                _ for i, _ in enumerate(split_systems(disk))
                if disk.init.packing_fraction[i] < packing_fraction_target
            ]

            # mean_pressure = 0.5 * np.mean([disk.trajectory[i].stress_tensor_total_x[:, 0] + disk.trajectory[i].stress_tensor_total_y[:, 0] for i in range(disk.trajectory.num_frames())], axis=0)
            # remaining_systems = [
            #     _ for i, _ in enumerate(split_systems(disk))
            #     if mean_pressure[i] < pressure_target
            # ]
            if len(remaining_systems) == 0:
                break
            disk = join_systems(remaining_systems)
            disk.save(init_path)

            # delete the compression path
            shutil.rmtree(compression_path)
            shutil.rmtree(dynamics_data_path)

            # increment the file index
            file_index += 1