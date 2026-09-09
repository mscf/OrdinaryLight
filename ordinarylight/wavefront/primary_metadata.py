"""Device-independent scene-local identity and base roughness table."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PreparedPrimaryMetadata:
    data: bytes
    triangle_count: int


def prepare_primary_metadata(scene):
    """Pack uvec4(material ID, instance ID, roughness bits, reserved) per triangle.

    Reprepare after topology, material or visibility changes. IDs are stable only
    within the source scene. Roughness is the base material value, not a textured
    or custom-program evaluation.
    """
    materials = scene.triangle_material_ids()
    instances = scene.triangle_instance_ids()
    roughness = (
        np.concatenate(
            [
                np.full(len(mesh.indices), mesh.material.roughness, dtype=np.float32)
                for mesh in scene.render_meshes
                if len(mesh.indices)
            ]
        )
        if len(materials)
        else np.empty(0, np.float32)
    )
    words = np.zeros((len(materials), 4), dtype=np.uint32)
    words[:, 0], words[:, 1], words[:, 2] = (
        materials,
        instances,
        roughness.view(np.uint32),
    )
    return PreparedPrimaryMetadata(words.tobytes(), len(words))
