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
    N = 100
    dim = 3
    e_int = 1.0
    dt = 1e-2
    phi = 0.4

    # assign bidisperse radii
    rad = jnp.ones(N)
    rad = rad.at[: N // 2].set(0.5)
    rad = rad.at[N // 2:].set(0.7)

    # set the box size for the packing fraction and the radii
    volume = (jnp.pi ** (dim / 2) / jax.scipy.special.gamma(dim / 2 + 1)) * rad ** dim
    L = (jnp.sum(volume) / phi) ** (1 / dim)
    box_size = jnp.ones(dim) * L

    # create microstate
    key = jax.random.PRNGKey(np.random.randint(0, 1000000))
    pos = jax.random.uniform(key, (N, dim), minval=0.0, maxval=L)
    mass = jnp.ones(N)
    mats = [jd.Material.create("elastic", young=e_int, poisson=0.5, density=1.0)]
    matcher = jd.MaterialMatchmaker.create("harmonic")
    mat_table = jd.MaterialTable.from_materials(mats, matcher=matcher)

    # create system and state
    state = jd.State.create(pos=pos, rad=rad, mass=mass)
    system = jd.System.create(
        state_shape=state.shape,
        dt=dt,
        linear_integrator_type="linearfire",
        domain_type="periodic",
        force_model_type="spring",
        collider_type="naive",
        mat_table=mat_table,
        domain_kw=dict(
            box_size=box_size,
        ),
    )

    # minimize initially
    state, system, _, _ = jd.minimizers.minimize(state, system)

    # run dynamics
    target_temperature = 1e-4
    seed = np.random.randint(0, 1000000)

    key = jax.random.PRNGKey(seed)
    state.vel = jax.random.normal(key, state.vel.shape)
    state.vel -= jnp.mean(state.vel, axis=0)
    temperature = 2 * jnp.sum(0.5 * state.mass * jnp.sum(state.vel ** 2, axis=-1)) / (state.dim * state.N)
    scale = jnp.sqrt(target_temperature / temperature)
    state.vel *= scale

    system = jd.System.create(
        state_shape=state.shape,
        dt=dt,
        linear_integrator_type="verlet",
        domain_type="periodic",
        force_model_type="spring",
        collider_type="naive",
        # collider_type="celllist",
        # collider_kw=dict(state=state),
        mat_table=system.mat_table,
        domain_kw=dict(
            box_size=system.domain.box_size,
        ),
    )

    n_repeats = 100
    ke = np.zeros(n_repeats)
    pe = np.zeros(n_repeats)
    for i in range(n_repeats):
        n_steps = 100
        state, system = system.step(state, system, n=n_steps)
        ke_t = 0.5 * (state.mass * jnp.sum(state.vel ** 2, axis=-1))
        ke[i] = jnp.sum(ke_t)
        pe[i] = jnp.sum(system.collider.compute_potential_energy(state, system))

    plt.plot(ke + pe / 2, label='TE')
    plt.plot(ke, label='KE')
    plt.plot(pe, label='PE')
    plt.legend()
    plt.savefig('energy_conservation.png')
    plt.close()