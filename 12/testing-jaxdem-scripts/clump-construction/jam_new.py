import jax.numpy as jnp
import jax

import jaxdem as jd
from jaxdem.utils.geometricAsperityCreation import generate_ga_clump_state

import numpy as np

jax.config.update("jax_enable_x64", True)

if __name__ == "__main__":
    phi = 0.4
    dim = 2
    
    N = 50
    asperity_radius = 0.1
    min_nv = 10
    max_nv = 14

    particle_radii = jd.utils.dispersity.get_polydisperse_radii(N)
    vertex_counts = np.ones_like(particle_radii).astype(int) * min_nv
    vertex_counts[particle_radii == max(particle_radii)] = max_nv

    state, box_size = generate_ga_clump_state(
        particle_radii,
        vertex_counts,
        phi,
        dim,
        asperity_radius,
        aspect_ratio=5.0,
        use_uniform_mesh=True,
        add_core=False
    )

    e_int = 1.0
    dt = 1e-2

    mats = [jd.Material.create("elastic", young=e_int, poisson=0.5, density=1.0)]
    matcher = jd.MaterialMatchmaker.create("harmonic")
    mat_table = jd.MaterialTable.from_materials(mats, matcher=matcher)

    system = jd.System.create(
        state_shape=state.shape,
        dt=dt,
        linear_integrator_type="linearfire",
        rotation_integrator_type="rotationfire",
        domain_type="periodic",
        force_model_type="spring",
        collider_type="naive",
        # collider_type="celllist",
        # collider_kw=dict(state=state),
        mat_table=mat_table,
        domain_kw=dict(
            box_size=box_size,
        ),
    )



    # exit()



    state, system, phi, pe = jd.utils.bisection_jam(state, system, n_minimization_steps=1_000_00, n_jamming_steps=1_000_000, packing_fraction_increment=1e-2)

    jd.utils.h5.save(state, 'jammed_state.h5')
    jd.utils.h5.save(system, 'jammed_system.h5')

    import subprocess
    from pathlib import Path
    import h5py
    with h5py.File('config.h5', 'w') as f:
        f.create_dataset("pos", data=np.asarray(state.pos))
        f.create_dataset("rad", data=np.asarray(state.rad))
        f.create_dataset("ID",  data=np.asarray(state.ID))
        f.create_dataset("box_size", data=np.asarray(system.domain.box_size))
    script_dir = Path(__file__).resolve().parent
    run_render = script_dir.parent / "rigid-particle-creation" / "run_render.sh"
    subprocess.run([
        str(run_render),
        "config.h5",
        "jammed.png",
        "1000",
    ], check=True)

    # make script to create any initial system of ga particles in 2d/3d
    # make script to jam any initial system of ga particles in 2d/3d

    # add to jaxdem

    # add nvt compression to jaxdem