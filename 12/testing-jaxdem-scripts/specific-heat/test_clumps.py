import jax
import jax.numpy as jnp
import dataclasses
from dataclasses import replace
from functools import partial

import jaxdem as jd

jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    dt = 1e-2
    e_int = 1.0

    arr = np.load("clump.npz")
    pos_c = jnp.asarray(arr["pos_c"])
    pos_p = jnp.asarray(arr["pos_p"])
    vel = jnp.asarray(arr["vel"])
    angVel = jnp.asarray(arr["angVel"])
    rad = jnp.asarray(arr["rad"])
    mass = jnp.asarray(arr["mass"])
    inertia = jnp.asarray(arr["inertia"])
    theta = jnp.asarray(arr["angle"])
    w = jnp.cos(theta / 2.0)[..., None]
    xyz = jnp.zeros((theta.shape[0], 3))
    xyz = xyz.at[:, 2].set(jnp.sin(theta / 2.0))
    q = jd.utils.Quaternion.create(w=w, xyz=xyz)
    ID = jnp.asarray(arr["ID"])
    box_size = jnp.asarray(arr["box_size"])

    state = jd.State(
        pos_c=pos_c,
        pos_p=pos_p,
        vel=vel,
        force=jnp.zeros_like(pos_c),
        q=q,
        angVel=angVel,
        torque=jnp.zeros_like(angVel),
        rad=rad,
        mass=mass,
        inertia=inertia,
        ID=ID,
        mat_id=jnp.zeros(ID.shape, int),
        species_id=jnp.zeros(ID.shape, int),
        fixed=jnp.zeros(ID.shape, bool),
    )
    assert state.is_valid

    mats = [jd.Material.create("elastic", young=e_int, poisson=0.5, density=1.0)]
    matcher = jd.MaterialMatchmaker.create("harmonic")
    mat_table = jd.MaterialTable.from_materials(mats, matcher=matcher)

    system = jd.System.create(
        state_shape=state.shape,
        dt=dt,
        linear_integrator_type="verlet",
        rotation_integrator_type="spiral",
        # rotation_integrator_type="",  # to turn rotations off
        domain_type="periodic",
        force_model_type="spring",
        collider_type="naive",
        mat_table=mat_table,
        domain_kw=dict(
            box_size=box_size,
        ),
    )

    n_repeats = 100
    ke = np.zeros(n_repeats)
    pe = np.zeros(n_repeats)
    particle_offset = jnp.concatenate((jnp.zeros(1), jnp.cumsum(jnp.bincount(state.ID)))).astype(jnp.int32)
    _, indices = jnp.unique(state.ID, return_index=True)
    for i in range(n_repeats):
        n_steps = 100
        state, system = system.step(state, system, n=n_steps)
        ke_t = 0.5 * (state.mass * jnp.sum(state.vel ** 2, axis=-1))
        ke_r = 0.5 * (state.inertia * state.angVel ** 2).squeeze()
        ke[i] = jnp.sum((ke_t + ke_r)[indices])
        pe[i] = jnp.sum(system.collider.compute_potential_energy(state, system))

    plt.plot(ke + pe / 2, label='TE')
    plt.plot(ke, label='KE')
    plt.plot(pe, label='PE')
    plt.legend()
    plt.savefig('energy_conservation.png')
    plt.close()
