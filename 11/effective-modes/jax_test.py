
from tkinter import N
import numpy as np
import time

import pydpmd as dp


temp = 1e-6
delta_phi = 1e-4
dt = 1e-2
n_steps = int(1e5)

# path = '/home/mmccraw/dev/data/11-15-25/jamming/disk/jamming_9'
path = "/home/mmccraw/dev/data/11-20-25/test/"
data = dp.data.load(path, location=['final', 'init'])
data.scale_to_packing_fraction(data.packing_fraction - delta_phi)
data.set_velocities(temp, 0)


# import jax
# import jaxdem as jd
# import jax.numpy as jnp
# jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
# pos, vel, mass, rad, box_size, e_int, ids = [], [], [], [], [], [], []
# for i in range(data.n_systems()):
#     beg = data.system_offset[i]
#     end = data.system_offset[i + 1]
#     pos.append(data.pos[beg:end])
#     vel.append(data.vel[beg:end])
#     mass.append(data.mass[beg:end])
#     rad.append(data.rad[beg:end])
#     box_size.append(data.box_size[i])
#     e_int.append(data.e_interaction[i])
#     ids.append(np.arange(end - beg, dtype=int))

# pos = jnp.asarray(pos)
# vel = jnp.asarray(vel)
# mass = jnp.asarray(mass)
# rad = jnp.asarray(rad)
# box_size = jnp.asarray(box_size)
# e_int = jnp.asarray(e_int)
# ids = jnp.asarray(ids)


# def create(sid):
#     mats = [jd.Material.create("elastic", young=e_int[sid], poisson=0.2)]
#     matcher = jd.MaterialMatchmaker.create("harmonic")
#     mat_table = jd.MaterialTable.from_materials(mats, matcher=matcher)
    
#     state = jd.State.create(
#         pos=pos[sid],
#         vel=vel[sid],
#         rad=rad[sid],
#         mass=mass[sid],
#         mat_id=ids[sid]
#     )
#     system = jd.System.create(
#         state_shape=state.shape,
#         dt=dt,
#         linear_integrator_type="verlet",
#         domain_type="periodic",
#         force_model_type="spring",
#         # collider_type="naive",
#         collider_type="celllist",
#         collider_kw=dict(state=state),
#         mat_table=mat_table,
#         domain_kw=dict(
#             box_size=box_size[sid],
#             anchor=box_size[sid] * 0.0,
#         ),
#     )
#     return state, system

# state, system = jax.vmap(create)(jnp.arange(data.n_systems()))
# # state, system = system.step(state, system, n=n_steps, batched=True)
# state, system = system.step(state, system, n=n_steps)
# print('starting')

# start = time.time()
# # state, system = system.step(state, system, n=n_steps, batched=True)
# state, system = system.step(state, system, n=n_steps)
# jax.block_until_ready(state)
# print(time.time() - start)


import subprocess
import os
from system_building_resources import *
path = '/home/mmccraw/dev/data/11-15-25/test/'
# data.set_neighbor_method(NeighborMethod.Naive)
data.set_neighbor_method(NeighborMethod.Cell)
set_standard_cell_list_parameters(data, 0.3)
data.save(path)
start = time.time()
subprocess.run([
    os.path.join("/home/mmccraw/dev/dpmd/build/", "nve_disk_pbc_final"),
    path,
    path,
    str(n_steps),
    str(1e5),
    str(dt)
], check=True)
print(time.time() - start)