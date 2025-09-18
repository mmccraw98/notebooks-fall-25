import numpy as np
from pydpmd.data import BaseParticle, load

def clamp_domains_to_box(domain_pos, domain_size, box_size, system_id):
    box_repeat = np.concatenate([box_size[sid][None].repeat(domain_size[i], axis=0) for i, sid in enumerate(system_id)])
    return np.clip(domain_pos, np.zeros_like(box_repeat), box_repeat)

def tri_area(a, b, c):
    return 0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))

def get_domain_areas(domain_pos, domain_centroid):
    areas = []
    size = domain_pos.shape[0]
    for i in range(0, size):
        areas.append(tri_area(domain_centroid, domain_pos[i], domain_pos[(i + 1) % size]))
    areas = np.array(areas)
    total_area = np.sum(areas)
    return np.cumsum(areas) / total_area, total_area

def build_domains(data: BaseParticle, domain_style: str = "square", pos: np.ndarray = None, domain_particle_id: np.ndarray = None, clamp_to_box: bool = False, domain_kwargs: dict = {}):
    """
    Build polygonal domains, typically centered around particles, typically used for sampling.

    Args:
        data (BaseParticle): The system data.
        domain_style (str): The style of domain to build (square, )
        pos (np.ndarray): The positions of the particles to build domains for.  If not specified, the positions from the data are used.
        domain_particle_id (np.ndarray): The particle ids to build domains for.  If not specified, one is built for each particle.
        clamp_to_box (bool): Whether to clamp the domain to the box.  Set to True for sampling in a walled container.
        domain_kwargs (dict): Additional keyword arguments for the domain.
    """
    if pos is None:
        pos = data.pos
    assert pos.shape == data.pos.shape
    if domain_particle_id is None:  # if not specified, build one for each particle
        domain_particle_id = np.arange(len(pos))

    if domain_style == "square":
        domain_length = domain_kwargs.get("domain_length")
        domain_stencil_dimless = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) / 2
        domain_pos = np.concatenate([domain_stencil_dimless * domain_length[i] + pos[i] for i in domain_particle_id])
        domain_size = np.ones(domain_particle_id.shape[0]).astype(int) * domain_stencil_dimless.shape[0]
    else:
        raise ValueError(f"Domain style {domain_style} not supported")

    domain_offset = np.concatenate([[0], np.cumsum(domain_size)])
    if clamp_to_box:
        domain_pos = clamp_domains_to_box(domain_pos, domain_size, data.box_size, data.system_id)
    domain_centroid = np.array([np.mean(domain_pos[domain_offset[i]:domain_offset[i + 1]], axis=0) for i in range(len(domain_offset) - 1)])

    # divide the domain into triangles, calculate the cumulative fractional area in each, and the total area
    domain_fractional_area, domain_area = zip(*[
        get_domain_areas(
            domain_pos[domain_offset[i]:domain_offset[i + 1]],
            domain_centroid[i]
        ) for i in range(len(domain_offset) - 1)])

    domain_fractional_area = np.concatenate(domain_fractional_area)
    domain_area = np.array(domain_area)

    # add domains to the data
    data.add_array(domain_pos, "domain_pos", ignore_missing_index_space=True)
    data.add_array(domain_centroid, "domain_centroid", ignore_missing_index_space=True)
    data.add_array(domain_offset, "domain_offset", ignore_missing_index_space=True)
    data.add_array(domain_fractional_area, "domain_fractional_area", ignore_missing_index_space=True)
    data.add_array(domain_area, "domain_area", ignore_missing_index_space=True)
    data.add_array(domain_particle_id, "domain_particle_id", ignore_missing_index_space=True)