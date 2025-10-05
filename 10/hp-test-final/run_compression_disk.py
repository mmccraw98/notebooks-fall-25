from statistics import correlation
import pydpmd as md
import numpy as np
import os
from system_building_resources import *
import time
import shutil
from correlation_functions import compute_msd, compute_pair_correlation_function

if __name__ == "__main__":
    n_steps_base = 1e7
    save_freq_base = 1e4
    temp_base = 1e-6

    radii = generate_bidisperse_radii(100, 0.5, 1.4)
    which = 'small'
    packing_fraction = 0.73
    phi_increment = 1e-2
    packing_fraction_target = 0.85
    dt = 5e-2

    for temp in [1e-4, 1e-5, 1e-6]:
        root = f"/home/mmccraw/dev/data/10-03-25/disk_compression_T_{temp:.3e}/"
        if not os.path.exists(root):
            os.makedirs(root)

        # build the initial data and equilibrate it, ideally to a 0-overlap state
        temp_path = tempfile.mkdtemp()
        # build the initial data and equilibrate it, ideally to a 0-overlap state
        disk = build_disk_system_from_radii(radii, which, packing_fraction, int(np.random.randint(0, 1e9)))
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

            n_steps = int(np.ceil(n_steps_base * np.sqrt(temp_base / temp) / 1e4) * 1e4)
            save_freq = int(np.ceil(save_freq_base * np.sqrt(temp_base / temp) / 1e3) * 1e3)
            
            compression_path = os.path.join(root, f"compression_{file_index}")
            disk.set_velocities(temp, np.random.randint(0, 1e9))
            disk.set_neighbor_method(NeighborMethod.Cell)
            set_standard_cell_list_parameters(disk, 0.3)
            disk.save(compression_path)
            subprocess.run([
                os.path.join("/home/mmccraw/dev/dpmd/build/", "nvt_rescale_disk_pbc_compress"),
                compression_path,
                compression_path,
                str(int(max(int(n_steps / 10), 1e3) / 2)),  # total number of steps - half are used for compression, the other half are used for equilibration
                str(phi_increment),
                str(temp),
                str(dt),
            ], check=True)

            # run nve dynamics from the output of the compression data
            disk = load(compression_path, location=["final", "init"])
            dynamics_data_path = os.path.join(root, f"dynamics_{file_index}")
            disk.set_velocities(temp, np.random.randint(0, 1e9))
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

            disk = load(dynamics_data_path, location=["final", "init"], load_trajectory=True, load_full=False)
            
            # calculate the average overlap in each system
            # first column stores the overlap for vertex i, second column stores the number of overlaps for vertex i
            mean_overlaps = np.sum(  # sum over all frames
                np.array(
                    [
                        np.add.reduceat(  # sum over all vertices within each system
                            disk.trajectory[i].overlaps, disk.system_offset[:-1]
                        , axis=0)
                        for i in range(disk.trajectory.num_frames())
                    ]
            ), axis=0)
            # divide the total number of overlaps by the number of overlapping vertices
            # repeat for all systems
            mean_overlaps = mean_overlaps[:, 0] / mean_overlaps[:, 1]
            # scale by the small particle diameter
            dimless_overlaps = mean_overlaps / (2.0 * disk.rad[disk.system_offset[:-1]])

            # calculate the average temperature in each system
            mean_temp = np.mean([disk.trajectory[i].temperature for i in range(disk.trajectory.num_frames())], axis=0)

            # calculate the average pressure in each system
            mean_p = np.mean([disk.trajectory[i].pressure for i in range(disk.trajectory.num_frames())], axis=0)

            # calculate the msd
            msd, t = compute_msd(disk)

            # calculate the pair correlation function
            g, r = compute_pair_correlation_function(disk)

            # save the data
            correlation_path = dynamics_data_path.rstrip("/") + '_results.npz'
            np.savez(
                correlation_path,
                overlap=mean_overlaps,
                dimless_overlap=dimless_overlaps,
                temp=mean_temp,
                pressure=mean_p,
                msd=msd,
                t=t,
                g=g,
                r=r,
                packing_fraction=disk.init.packing_fraction,
            )


            # load the data, split into separate systems, recombining only those that have a packing fraction less than the target
            disk = load(dynamics_data_path, location=["final", "init"], load_trajectory=True, load_full=False)
            remaining_systems = [
                _ for i, _ in enumerate(split_systems(disk))
                if disk.init.packing_fraction[i] < packing_fraction_target
            ]
            if len(remaining_systems) == 0:
                break
            disk = join_systems(remaining_systems)
            disk.save(init_path)

            # delete the compression path
            shutil.rmtree(compression_path)
            shutil.rmtree(dynamics_data_path)

            # increment the file index
            file_index += 1
