import pydpmd as md
import numpy as np
import os
from system_building_resources import *
import time
import shutil
from correlation_functions import compute_shear_modulus

if __name__ == "__main__":
    for i in range(1):
        # root = f"/home/mmccraw/dev/data/10-01-25/calculate-shear-modulus-final/trial-{i}/"
        # root = f"/home/mmccraw/dev/data/10-01-25/calculate-shear-modulus-small/trial-{i}/"
        # root = f"/home/mmccraw/dev/data/10-01-25/calculate-shear-modulus-small-lower-temp/trial-{i}/"
        # root = f"/home/mmccraw/dev/data/10-01-25/calculate-shear-modulus-small-lower-temp-extra/trial-{i}/"
        # root = f"/home/mmccraw/dev/data/10-01-25/short-test/trial-{i}/"
        # root = f"/home/mmccraw/dev/data/10-01-25/short-test-3/trial-{i}/"
        # root = f"/home/mmccraw/dev/data/10-01-25/calculate-shear-modulus-fine-range/trial-{i}/"
        root = f"/home/mmccraw/dev/data/10-01-25/calculate-shear-modulus-fine-range-2/trial-{i}/"
        if not os.path.exists(root):
            os.makedirs(root)

        radii = generate_bidisperse_radii(100, 0.5, 1.4)
        which = 'small'
        packing_fraction = 0.75
        phi_increment = 1e-3
        temperature = 1e-6
        n_steps = 1e5
        save_freq = 1e0
        pressure_target = 1e-3
        packing_fraction_target = 0.85
        dt = 2e-2

        # build the initial data and equilibrate it, ideally to a 0-overlap state
        mu_effs = []
        nvs = []

        nvs = [
            3, 3, 3,
            6, 6, 6,
            10, 10, 10,
            20, 20, 20,
            30, 30, 30,
        ]
        mu_effs = [
            0.2, 0.3, 0.35,
            0.55, 0.65, 0.85,
            0.55, 0.65, 0.85,
            0.45, 0.55, 0.65,
            0.45, 0.55, 0.65,
        ]
        # for mu_eff in [0.01, 0.05, 0.1, 0.5, 1.0]:
        #     for nv in [3, 6, 10, 20, 30]:
        #         mu_effs.append(mu_eff)
        #         nvs.append(nv)
        
        n_duplicates = len(mu_effs)
        cap_nv = 3
        add_core = True
        rb = build_rigid_bumpy_system_from_radii(radii, which, mu_effs, nvs, packing_fraction, add_core, cap_nv, 'uniform', n_duplicates)
        rb.calculate_mu_eff()
        print(rb.mu_eff[rb.system_offset[:-1]])

        init_path = os.path.join(root, "init")
        rb.set_neighbor_method(NeighborMethod.Cell)
        set_standard_cell_list_parameters(rb, 0.3)
        rb.save(init_path)

        # compress each system individually until they are at the initial packing fraction
        merged = []
        for i, _rb in enumerate(split_systems(rb)):
            if _rb.packing_fraction[0] < packing_fraction - phi_increment:
                print(f"Compressing system {i} from {_rb.packing_fraction[0]:.3f} to {packing_fraction - phi_increment:.3f}")
                interm_path = tempfile.mktemp()
                while _rb.packing_fraction[0] < packing_fraction - phi_increment:
                    _rb.set_velocities(temperature, np.random.randint(0, 1e9))
                    _rb.set_neighbor_method(NeighborMethod.Cell)
                    set_standard_cell_list_parameters(_rb, 0.3)
                    _rb.save(interm_path)
                    subprocess.run([
                        os.path.join("/home/mmccraw/dev/dpmd/build/", "nvt_rescale_rigid_bumpy_pbc_compress"),
                        interm_path,
                        interm_path,
                        str(1e3),
                        str(phi_increment),
                        str(temperature),
                        str(dt),
                    ], check=True)
                    _rb = load(interm_path, location=["final", "init"])
                shutil.rmtree(interm_path)
                print(f"System {i} compressed to {_rb.packing_fraction[0]:.3f}")
            else:  # have to run a no-op compression so that join_systems wont complain about the different init structures
                interm_path = tempfile.mktemp()
                _rb.set_velocities(temperature, np.random.randint(0, 1e9))
                _rb.set_neighbor_method(NeighborMethod.Cell)
                set_standard_cell_list_parameters(_rb, 0.3)
                _rb.save(interm_path)
                subprocess.run([
                    os.path.join("/home/mmccraw/dev/dpmd/build/", "nvt_rescale_rigid_bumpy_pbc_compress"),
                    interm_path,
                    interm_path,
                    str(1e3),
                    str(0),
                    str(temperature),
                    str(dt),
                ], check=True)
                _rb = load(interm_path, location=["final", "init"])
                shutil.rmtree(interm_path)
            merged.append(_rb)
        rb = join_systems(merged)
        rb = join_systems([rb for i in range(10)])
        rb.save(init_path)
        
        # run the iterative compress-dynamics protocol
        file_index = 0
        while True:
            # compress the data, incrementing the packing fraction by phi_increment and maintaining the temperature
            rb = load(init_path, location=["final", "init"])
            shutil.rmtree(init_path)
            
            compression_path = os.path.join(root, f"compression_{file_index}")
            rb.set_velocities(temperature, np.random.randint(0, 1e9))
            rb.set_neighbor_method(NeighborMethod.Cell)
            set_standard_cell_list_parameters(rb, 0.3)
            rb.save(compression_path)
            subprocess.run([
                os.path.join("/home/mmccraw/dev/dpmd/build/", "nvt_rescale_rigid_bumpy_pbc_compress"),
                compression_path,
                compression_path,
                str(int(max(int(n_steps / 10), 1e3) / 2)),  # total number of steps - half are used for compression, the other half are used for equilibration
                str(phi_increment),
                str(temperature),
                str(dt),
            ], check=True)

            # run nve dynamics from the output of the compression data
            rb = load(compression_path, location=["final", "init"])
            dynamics_data_path = os.path.join(root, f"dynamics_{file_index}")
            rb.set_velocities(temperature, np.random.randint(0, 1e9))
            rb.set_neighbor_method(NeighborMethod.Cell)
            set_standard_cell_list_parameters(rb, 0.3)
            rb.save(dynamics_data_path)
            subprocess.run([
                os.path.join("/home/mmccraw/dev/dpmd/build/", "nve_rigid_bumpy_pbc_final"),
                dynamics_data_path,
                dynamics_data_path,
                str(n_steps),  # total number of dynamics steps
                str(save_freq),  # save frequency
                str(dt),
            ], check=True)

            # # calculate the average overlap in each system
            # rb = load(dynamics_data_path, location=["final", "init"], load_trajectory=True, load_full=False)
            # rb.calculate_mu_eff()
            # overlap_path = os.path.join(root, f"overlap_{file_index}.npz")
            # first column stores the overlap for vertex i, second column stores the number of overlaps for vertex i
            # mean_overlaps = np.sum(  # sum over all frames
            #     np.array(
            #         [
            #             np.add.reduceat(  # sum over all vertices within each system
            #                 rb.trajectory[i].overlaps, rb.system_offset[:-1]
            #             , axis=0)
            #             for i in range(rb.trajectory.num_frames())
            #         ]
            # ), axis=0)
            # divide the total number of overlaps by the number of overlapping vertices
            # repeat for all systems
            # mean_overlaps = mean_overlaps[:, 0] / mean_overlaps[:, 1]
            # np.savez(
            #     overlap_path,
            #     overlap=mean_overlaps,
            #     mu_eff=rb.mu_eff[rb.system_offset[:-1]],
            #     nv=rb.n_vertices_per_particle[rb.system_offset[:-1]],
            #     packing_fraction=rb.init.packing_fraction,
            # )

            # calculate the shear modulus and save it with mu_eff, nv, packing_fraction, and temperature
            rb = load(dynamics_data_path, location=["final", "init"], load_trajectory=True, load_full=False)
            rb.calculate_mu_eff()
            shear_modulus_path = os.path.join(root, f"shear_modulus_{file_index}.npz")
            shear_modulus, t = compute_shear_modulus(rb, None)
            np.savez(
                shear_modulus_path,
                shear_modulus=shear_modulus,
                t=t,
                mu_eff=rb.mu_eff[rb.system_offset[:-1]],
                nv=rb.n_vertices_per_particle[rb.system_offset[:-1]],
                packing_fraction=rb.init.packing_fraction,
            )


            # load the data, split into separate systems, recombining only those that have a packing fraction less than the target
            rb = load(dynamics_data_path, location=["final", "init"], load_trajectory=True, load_full=False)
            remaining_systems = [
                _ for i, _ in enumerate(split_systems(rb))
                if rb.init.packing_fraction[i] < packing_fraction_target
            ]

            # mean_pressure = 0.5 * np.mean([rb.trajectory[i].stress_tensor_total_x[:, 0] + rb.trajectory[i].stress_tensor_total_y[:, 0] for i in range(rb.trajectory.num_frames())], axis=0)
            # remaining_systems = [
            #     _ for i, _ in enumerate(split_systems(rb))
            #     if mean_pressure[i] < pressure_target
            # ]
            if len(remaining_systems) == 0:
                break
            rb = join_systems(remaining_systems)
            rb.save(init_path)

            # delete the compression path
            shutil.rmtree(compression_path)
            shutil.rmtree(dynamics_data_path)

            # increment the file index
            file_index += 1