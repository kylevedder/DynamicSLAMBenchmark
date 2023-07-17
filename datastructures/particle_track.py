from typing import List
import numpy as np
import open3d as o3d


class ParticlePosition():

    def __init__(self, x: float, y: float, z: float, occluded: bool = False):
        self.x = x
        self.y = y
        self.z = z
        self.occluded = occluded

    def __repr__(self):
        return f"ParticlePosition(x={self.x}, y={self.y}, z={self.z}, occluded={self.occluded})"


class UninitializedParticle(ParticlePosition):
    """
    Subclass of ParticlePosition where the particle is uninitialized in its position (e.g. off camera).
    """

    def __init__(self):
        pass

    def __repr__(self):
        return "UninitializedParticle()"


class ParticleFrame():

    def __init__(self, particle_positions: np.array,
                 is_occluded: np.ndarray[bool],
                 is_initialized: np.ndarray[bool]):
        assert len(particle_positions) == len(is_occluded), \
            f"particle_positions and is_obstructed have different lengths, {len(particle_positions)} != {len(is_occluded)}"

        self.is_occluded = is_occluded
        self.is_initialized = is_initialized
        self.particle_positions = np.array([
            ParticlePosition(*p, o) if i else UninitializedParticle()
            for p, o, i in zip(particle_positions, is_occluded, is_initialized)
        ])

    def __getitem__(self, idx):
        assert idx < len(self.particle_positions), \
            f"idx {idx} is out of bounds, {idx} >= {len(self.particle_positions)}"
        assert idx >= 0, \
            f"idx {idx} is out of bounds, {idx} < 0"
        return self.particle_positions[idx]

    def __len__(self):
        return len(self.particle_positions)

    def get_unoccluded_particles(self) -> np.ndarray[ParticlePosition]:
        return self.particle_positions[~self.is_occluded & self.is_initialized]

    def set_unoccluded_particles(self, new_positions: List[ParticlePosition]):
        assert len(new_positions) == len(self.get_unoccluded_particles()), \
            f"new_positions and unoccluded_particles have different lengths, {len(new_positions)} != {len(self.get_unoccluded_particles())}"
        self.particle_positions[~self.is_occluded
                                & self.is_initialized] = new_positions

    def to_o3d(self) -> List[o3d.geometry.TriangleMesh]:

        def make_sphere_at_location(particle_position: ParticlePosition):
            return o3d.geometry.TriangleMesh.create_sphere(
                radius=0.05).translate(
                    np.array([
                        particle_position.x, particle_position.y,
                        particle_position.z
                    ])).paint_uniform_color([1, 0, 0])

        # Make a sphere for each particle
        return [
            make_sphere_at_location(p)
            for p in self.get_unoccluded_particles()
        ]
