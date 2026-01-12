import jax.numpy as jnp
import jax

import jaxdem as jd
from jaxdem.utils.geometricAsperityCreation import generate_ga_clump_state

import numpy as np
from scipy.optimize import minimize_scalar, brentq

jax.config.update("jax_enable_x64", True)

def calc_mu_eff(vertex_radius, outer_radius, num_vertices):
    return 1 / np.sqrt(((2 * vertex_radius) / ((outer_radius - vertex_radius) * np.sin(np.pi / num_vertices))) ** 2 - 1)

def find_num_vertices_for_target_mu_eff(
    target_mu_eff: float,
    vertex_radius: float,
    outer_radius: float,
    num_vertices_min: int = 3,
    num_vertices_max: int = 100):
    best_nv = None
    best_mu = np.nan
    best_err = np.inf
    for nv in range(int(num_vertices_min), int(num_vertices_max) + 1):
        try:
            mu = float(calc_mu_eff(vertex_radius, outer_radius, nv))
        except (ValueError, ZeroDivisionError, FloatingPointError, OverflowError, TypeError):
            continue
        if not np.isfinite(mu):
            continue
        err = abs(mu - target_mu_eff)
        if err < best_err:
            best_nv, best_mu, best_err = nv, mu, err
    return best_nv, best_mu, best_err

def get_closest_vertex_radius_for_mu_eff(mu_eff, outer_radius, num_vertices):
    # Calculate mathematically valid bounds
    sin_term = np.sin(np.pi / num_vertices)
    min_vertex_radius = outer_radius * sin_term / (2 + sin_term) + 1e-12
    max_vertex_radius = outer_radius - 1e-12
    
    # Check if target mu_eff is achievable
    max_mu_eff = calc_mu_eff(min_vertex_radius, outer_radius, num_vertices)
    min_mu_eff = calc_mu_eff(max_vertex_radius, outer_radius, num_vertices)
    
    if mu_eff > max_mu_eff or mu_eff < min_mu_eff:
        # Target mu_eff is outside achievable range
        return np.nan
    try:
        # Use root finding since we want calc_mu_eff(vertex_radius) = mu_eff
        def objective(vertex_radius):
            return calc_mu_eff(vertex_radius, outer_radius, num_vertices) - mu_eff
        
        # Brent's method is robust for this monotonic function
        result = brentq(objective, min_vertex_radius, max_vertex_radius, xtol=1e-12)
        return result
        
    except (ValueError, RuntimeError, ZeroDivisionError):
        # Fallback to bounded scalar minimization if root finding fails
        def obj_squared(vertex_radius):
            try:
                return (calc_mu_eff(vertex_radius, outer_radius, num_vertices) - mu_eff) ** 2
            except (ValueError, RuntimeError, ZeroDivisionError):
                return np.inf
        
        result = minimize_scalar(obj_squared, bounds=(min_vertex_radius, max_vertex_radius), method='bounded')
        return result.x if result.success else np.nan

import os
data_root = '/home/mmccraw/dev/data/01-01-26/specific-heat'

if __name__ == "__main__":
    phi = 0.4
    dim = 2
    min_nv = 20
    N = 50

    jamming_root = os.path.join(data_root, 'jamming')
    if not os.path.exists(jamming_root):
        os.makedirs(jamming_root)

    for mu_eff in [0.01, 0.1, 1.0]:
        for aspect_ratio in [1.0, 1.5, 2.0]:
            particle_name = f'mu-{mu_eff}-alpha-{aspect_ratio}'
            particle_root = os.path.join(jamming_root, particle_name)

            if not os.path.exists(particle_root):
                os.makedirs(particle_root)
            else:
                continue
            
            particle_radii = jd.utils.dispersity.get_polydisperse_radii(N)
            asperity_radius = get_closest_vertex_radius_for_mu_eff(mu_eff, min(particle_radii), min_nv)
            max_nv, max_mu_eff, err = find_num_vertices_for_target_mu_eff(mu_eff, asperity_radius, max(particle_radii))
            vertex_counts = np.ones_like(particle_radii).astype(int) * min_nv
            vertex_counts[particle_radii == max(particle_radii)] = max_nv

            state, box_size = generate_ga_clump_state(
                particle_radii,
                vertex_counts,
                phi,
                dim,
                asperity_radius,
                aspect_ratio=aspect_ratio,
                use_uniform_mesh=True,
                add_core=True
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

            state, system, phi, pe = jd.utils.bisection_jam(state, system, n_minimization_steps=1_000_00, n_jamming_steps=1_000_000, packing_fraction_increment=1e-2)

            jd.utils.h5.save(state, os.path.join(particle_root, 'state.h5'))
            jd.utils.h5.save(system, os.path.join(particle_root, 'system.h5'))

            # import subprocess
            # from pathlib import Path
            # import h5py
            # with h5py.File('config.h5', 'w') as f:
            #     f.create_dataset("pos", data=np.asarray(state.pos))
            #     f.create_dataset("rad", data=np.asarray(state.rad))
            #     f.create_dataset("ID",  data=np.asarray(state.ID))
            #     f.create_dataset("box_size", data=np.asarray(system.domain.box_size))
            # script_dir = Path(__file__).resolve().parent
            # run_render = script_dir.parent / "rigid-particle-creation" / "run_render.sh"
            # subprocess.run([
            #     str(run_render),
            #     "config.h5",
            #     f"figures/jamming/{particle_name}.png",
            #     "1000",
            # ], check=True)
            # os.remove("config.h5")

            # # make script to create any initial system of ga particles in 2d/3d
            # # make script to jam any initial system of ga particles in 2d/3d

            # # add to jaxdem

            # # add nvt compression to jaxdem