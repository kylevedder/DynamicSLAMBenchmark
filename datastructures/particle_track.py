from typing import List
import numpy as np
import open3d as o3d


class ParticlePosition():

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class ObstructedParticle(ParticlePosition):
    """
    Subclass of ParticlePosition where the particle is obstructed by an object.
    """

    def __init__(self):
        pass


class ParticleFrame():

    def __init__(self, particle_positions: np.array, is_occluded: np.array):
        assert len(particle_positions) == len(is_occluded), \
            f"particle_positions and is_obstructed have different lengths, {len(particle_positions)} != {len(is_occluded)}"

        self.is_occluded = is_occluded
        self.particle_positions = np.array([
            ParticlePosition(*p) if not o else ObstructedParticle()
            for p, o in zip(particle_positions, is_occluded)
        ])

    def __getitem__(self, idx):
        assert idx < len(self.particle_positions), \
            f"idx {idx} is out of bounds, {idx} >= {len(self.particle_positions)}"
        assert idx >= 0, \
            f"idx {idx} is out of bounds, {idx} < 0"
        return self.particle_positions[idx]

    def __len__(self):
        return len(self.particle_positions)

    def get_unobstructed_particles(self) -> np.array:
        return self.particle_positions[~self.is_occluded]

    def set_unobstructed_particles(self,
                                   new_positions: List[ParticlePosition]):
        self.particle_positions[~self.is_occluded] = new_positions

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
            for p in self.get_unobstructed_particles()
        ]
