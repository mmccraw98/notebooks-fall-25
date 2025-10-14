import pydpmd as md
import numpy as np
import os
from system_building_resources import *
import time
import shutil
from correlation_functions import compute_shear_modulus

# we want to see the behavior of:
# the shear modulus as a function of mu_eff, nv, phi
# the shear viscosity as a function of mu_eff, nv, phi
# we want to see if there is a regime where either begins to plateau upon increasing phi if mu_eff is large enough
# prior work shows that viscosity should diverge at increasingly smaller phi for increasingly large mu_eff - 
# however, it seems that there may be a plateau regime where the particles begin to act as if they were soft
# here, we will continuously minimize the energy after each step, stopping further calculations for each system if -
# it is beyond its maximal jamming density
# this will allow us to disentangle the possible plateau (effectively soft) regime from true soft regime (where overlaps exist)

if __name__ == "__main__":
    for i in range(10):
        root = f"/home/mmccraw/dev/data/10-15-25/new-shear-modulus-dilute/trial-{i}/"
        if not os.path.exists(root):
            os.makedirs(root)

        radii = generate_bidisperse_radii(100, 0.5, 1.4)
        which = 'small'
        packing_fraction = 0.7
        phi_increment = 1e-3
        temperature = 1e-6
        n_steps = 1e5
        save_freq = 1e0
        dt = 1e-2
        pe_tol = 1e-14
        n_duplicates_per_system = 10

        # build the initial data and equilibrate it, ideally to a 0-overlap state
        mu_effs = []
        nvs = []
        for mu_eff in [0.01, 0.05, 0.1, 0.5, 1.0]:
            for nv in [3, 6, 10, 20, 30]:
                mu_effs.append(mu_eff)
                nvs.append(nv)
        assert len(mu_effs) == len(nvs)

        cap_nv = 3
        add_core = True
        # create the systems at a 10% lower density than initially desired so that the duplicate systems have a chance to become sufficiently dissimilar
        # this way, we avoid having to create 10x duplicates in the below function which is very slow
        rb = build_rigid_bumpy_system_from_radii(radii, which, mu_effs, nvs, packing_fraction - 0.1, add_core, cap_nv, 'uniform', len(mu_effs))
        rb.calculate_mu_eff()

        init_path = os.path.join(root, "init")
        rb.set_neighbor_method(NeighborMethod.Cell)
        set_standard_cell_list_parameters(rb, 0.3)
        rb.save(init_path)

        # compress each system individually, creating n_duplicates_per_system copies of each system, and compressing each set until they are at the initial packing fraction
        merged = []
        for i, _rb in enumerate(split_systems(rb)):
            _rb = join_systems([_rb for _ in range(n_duplicates_per_system)])
            initial_phi_increment = 1e-2
            if _rb.packing_fraction[0] < packing_fraction - initial_phi_increment:
                print(f"Compressing system {i} from {_rb.packing_fraction[0]:.3f} to {packing_fraction - initial_phi_increment:.3f}")
                interm_path = tempfile.mktemp()
                while _rb.packing_fraction[0] < packing_fraction - initial_phi_increment:
                    _rb.set_velocities(temperature, np.random.randint(0, 1e9))
                    _rb.set_neighbor_method(NeighborMethod.Cell)
                    set_standard_cell_list_parameters(_rb, 0.3)
                    _rb.save(interm_path)
                    subprocess.run([
                        os.path.join("/home/mmccraw/dev/dpmd/build/", "nvt_rescale_rigid_bumpy_pbc_compress"),
                        interm_path,
                        interm_path,
                        str(1e3),
                        str(initial_phi_increment),
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
        rb.set_velocities(temperature, np.random.randint(0, 1e9))
        rb.set_neighbor_method(NeighborMethod.Cell)
        set_standard_cell_list_parameters(rb, 0.3)
        rb.save(init_path)
        
        # relax the systems in nvt
        subprocess.run([
            os.path.join("/home/mmccraw/dev/dpmd/build/", "nvt_rescale_rigid_bumpy_pbc_compress"),
            init_path,
            init_path,
            str(1_000_000 // 2),  # total number of steps - half are used for compression, the other half are used for equilibration
            str(phi_increment),
            str(temperature),
            str(dt),
        ], check=True)

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

            # load the data and minimize it to the nearest energy minimum
            rb = load(dynamics_data_path, location=["final", "init"])
            minimization_data_path = os.path.join(root, f"minimization_{file_index}")
            rb.set_neighbor_method(NeighborMethod.Cell)
            set_standard_cell_list_parameters(rb, 0.3)
            rb.save(minimization_data_path)
            subprocess.run([
                os.path.join("/home/mmccraw/dev/dpmd/build/", "rigid_bumpy_equilibrate_pbc"),
                minimization_data_path,
                minimization_data_path,
            ], check=True)

            # load the data, split into separate systems, recombining only those that still correspond to a hard-particle system
            rb = load(minimization_data_path, location=["final", "init"])
            remaining_systems = [
                _ for i, _ in enumerate(split_systems(rb))
                if rb.final.pe_total[i] / rb.system_size[i] < pe_tol
            ]

            if len(remaining_systems) == 0:
                break
            rb = join_systems(remaining_systems)
            rb.save(init_path)

            # delete the compression path
            shutil.rmtree(compression_path)
            shutil.rmtree(dynamics_data_path)

            # increment the file index
            file_index += 1
    
    print(f"Done running {i}")