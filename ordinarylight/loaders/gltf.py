"""glTF 2.0 scene loading for :mod:`ordinarylight.loaders`."""

from base64 import b64decode
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote

import numpy as np
from PIL import Image
from pygltflib import GLTF2

from ..animations import AnimationClip, AnimationTrack, MorphTarget, Skin
from ..materials import unlit_material
from ..scene import Material, Scene, Texture, TextureTransform, Transform


_COMPONENT_DTYPES = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
_TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}
_SUPPORTED_EXTENSIONS = frozenset({
    "KHR_lights_punctual",
    "KHR_materials_emissive_strength",
    "KHR_materials_ior",
    "KHR_materials_transmission",
    "KHR_materials_clearcoat",
    "KHR_materials_sheen",
    "KHR_materials_anisotropy",
    "KHR_materials_unlit",
    "KHR_materials_volume",
    "KHR_texture_transform",
    # Metadata does not change scene interpretation.
    "KHR_xmp_json_ld",
})


def _buffer_bytes(gltf, index, path):
    buffer = gltf.buffers[index]
    if buffer.uri is None:
        return gltf.binary_blob()
    if buffer.uri.startswith("data:"):
        return b64decode(buffer.uri.split(",", 1)[1])
    return (path.parent / unquote(buffer.uri)).read_bytes()


def _accessor_array(gltf, accessor_index, buffers):
    accessor = gltf.accessors[accessor_index]
    dtype = np.dtype(_COMPONENT_DTYPES[accessor.componentType]).newbyteorder("<")
    components = _TYPE_COMPONENTS[accessor.type]

    def read_view(view_index, count, value_dtype, width, byte_offset=0):
        view = gltf.bufferViews[view_index]
        offset = (view.byteOffset or 0) + (byte_offset or 0)
        packed_stride = value_dtype.itemsize * width
        stride = view.byteStride or packed_stride
        payload = buffers[view.buffer]
        if stride == packed_stride:
            return np.frombuffer(
                payload, dtype=value_dtype, count=count * width, offset=offset
            ).reshape((count, width))
        return np.ndarray(
            shape=(count, width), dtype=value_dtype, buffer=payload,
            offset=offset, strides=(stride, value_dtype.itemsize),
        )

    if accessor.bufferView is None:
        result = np.zeros((accessor.count, components), dtype=dtype)
    else:
        result = read_view(
            accessor.bufferView, accessor.count, dtype, components,
            accessor.byteOffset,
        )
    result = np.array(result, copy=True)
    sparse = accessor.sparse
    if sparse is not None and sparse.count:
        index_dtype = np.dtype(
            _COMPONENT_DTYPES[sparse.indices.componentType]
        ).newbyteorder("<")
        if sparse.indices.componentType not in (5121, 5123, 5125):
            raise ValueError("glTF sparse indices must be unsigned integers")
        indices = read_view(
            sparse.indices.bufferView, sparse.count, index_dtype, 1,
            sparse.indices.byteOffset,
        ).reshape(-1).astype(np.int64)
        if np.any(indices >= accessor.count):
            raise ValueError("glTF sparse accessor index is out of range")
        values = read_view(
            sparse.values.bufferView, sparse.count, dtype, components,
            sparse.values.byteOffset,
        )
        result[indices] = values
    if accessor.normalized and accessor.componentType != 5126:
        if np.issubdtype(result.dtype, np.signedinteger):
            maximum = float(np.iinfo(result.dtype).max)
            result = np.maximum(result.astype(np.float32) / maximum, -1.0)
        else:
            result = result.astype(np.float32) / float(np.iinfo(result.dtype).max)
    return result


def _node_matrix(node):
    if node.matrix:
        return np.asarray(node.matrix, dtype=np.float64).reshape((4, 4), order="F")
    translation = np.eye(4)
    translation[:3, 3] = node.translation or (0.0, 0.0, 0.0)
    scale = np.eye(4)
    scale[np.diag_indices(3)] = node.scale or (1.0, 1.0, 1.0)
    x, y, z, w = node.rotation or (0.0, 0.0, 0.0, 1.0)
    rotation = np.array((
        (1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w, 0),
        (2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w, 0),
        (2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y, 0),
        (0, 0, 0, 1),
    ))
    return translation @ rotation @ scale


def _image_bytes(gltf, index, buffers, path):
    image = gltf.images[index]
    if image.uri is not None:
        if image.uri.startswith("data:"):
            return b64decode(image.uri.split(",", 1)[1])
        return (path.parent / unquote(image.uri)).read_bytes()
    view = gltf.bufferViews[image.bufferView]
    start = view.byteOffset or 0
    return buffers[view.buffer][start:start + view.byteLength]


def _textures(gltf, buffers, path):
    decoded_images = []
    for index in range(len(gltf.images)):
        with Image.open(BytesIO(_image_bytes(gltf, index, buffers, path))) as image:
            decoded_images.append(np.asarray(image.convert("RGBA"), dtype=np.uint8))
    wrap = {10497: "repeat", 33071: "clamp", 33648: "mirror"}
    result = []
    for source in gltf.textures:
        sampler = gltf.samplers[source.sampler] if source.sampler is not None else None
        min_filter = sampler.minFilter if sampler is not None else None
        mag_filter = sampler.magFilter if sampler is not None else None
        result.append(Texture(
            decoded_images[source.source],
            wrap_s=wrap.get(sampler.wrapS if sampler else None, "repeat"),
            wrap_t=wrap.get(sampler.wrapT if sampler else None, "repeat"),
            linear_filter=(mag_filter != 9728 and min_filter not in (9728, 9984, 9986)),
        ))
    return result


def _texture_reference(info, textures):
    if info is None:
        return None, TextureTransform()
    if isinstance(info, dict):
        extension = (info.get("extensions") or {}).get("KHR_texture_transform", {})
        texcoord = extension.get("texCoord", info.get("texCoord", 0))
        texture_index = info["index"]
    else:
        extension = (info.extensions or {}).get("KHR_texture_transform", {})
        texcoord = extension.get("texCoord", info.texCoord or 0)
        texture_index = info.index
    if texcoord not in (0, 1):
        raise ValueError("Only texture coordinate sets TEXCOORD_0 and TEXCOORD_1 are supported")
    transform = TextureTransform(
        offset=tuple(extension.get("offset", (0.0, 0.0))),
        scale=tuple(extension.get("scale", (1.0, 1.0))),
        rotation=float(extension.get("rotation", 0.0)),
        texcoord_set=int(texcoord),
    )
    return textures[texture_index], transform


def _material(gltf, index, textures):
    if index is None:
        return Material()
    source = gltf.materials[index]
    pbr = source.pbrMetallicRoughness
    base = tuple(((pbr.baseColorFactor if pbr is not None else None)
                  or (1.0, 1.0, 1.0, 1.0))[:3])
    opacity = float(((pbr.baseColorFactor if pbr is not None else None)
                     or (1.0, 1.0, 1.0, 1.0))[3])
    emissive = tuple(source.emissiveFactor or (0.0, 0.0, 0.0))
    extensions = source.extensions or {}
    transmission = extensions.get("KHR_materials_transmission", {})
    volume = extensions.get("KHR_materials_volume", {})
    ior = extensions.get("KHR_materials_ior", {})
    clearcoat = extensions.get("KHR_materials_clearcoat", {})
    sheen = extensions.get("KHR_materials_sheen", {})
    anisotropy = extensions.get("KHR_materials_anisotropy", {})
    attenuation_distance = volume.get("attenuationDistance", float("inf"))
    emissive_strength = extensions.get(
        "KHR_materials_emissive_strength", {}
    ).get("emissiveStrength", 1.0)
    emissive = tuple(float(component) * float(emissive_strength)
                     for component in emissive)
    base_texture, base_transform = _texture_reference(
        pbr.baseColorTexture if pbr is not None else None, textures
    )
    metallic_texture, metallic_transform = _texture_reference(
        pbr.metallicRoughnessTexture if pbr is not None else None, textures
    )
    emissive_texture, emissive_transform = _texture_reference(
        source.emissiveTexture, textures
    )
    normal_texture, normal_transform = _texture_reference(
        source.normalTexture, textures
    )
    occlusion_texture, occlusion_transform = _texture_reference(
        source.occlusionTexture, textures
    )
    transmission_texture, transmission_transform = _texture_reference(
        transmission.get("transmissionTexture"), textures
    )
    thickness_texture, thickness_transform = _texture_reference(
        volume.get("thicknessTexture"), textures
    )
    clearcoat_texture, _ = _texture_reference(
        clearcoat.get("clearcoatTexture"), textures
    )
    sheen_texture, _ = _texture_reference(
        sheen.get("sheenColorTexture"), textures
    )
    anisotropy_texture, _ = _texture_reference(
        anisotropy.get("anisotropyTexture"), textures
    )
    return Material(
        base_color=base,
        emission=emissive,
        metallic=(pbr.metallicFactor if pbr is not None
                  and pbr.metallicFactor is not None else 1.0),
        roughness=(pbr.roughnessFactor if pbr is not None
                   and pbr.roughnessFactor is not None else 1.0),
        transmission=transmission.get("transmissionFactor", 0.0),
        clearcoat=clearcoat.get("clearcoatFactor", 0.0),
        clearcoat_roughness=clearcoat.get("clearcoatRoughnessFactor", 0.1),
        sheen_color=tuple(sheen.get("sheenColorFactor", (0.0, 0.0, 0.0))),
        sheen_roughness=sheen.get("sheenRoughnessFactor", 0.5),
        anisotropy=anisotropy.get("anisotropyStrength", 0.0),
        ior=ior.get("ior", 1.5),
        attenuation_color=tuple(volume.get("attenuationColor", (1.0, 1.0, 1.0))),
        attenuation_distance=attenuation_distance,
        thickness=volume.get("thicknessFactor", 0.0),
        opacity=opacity,
        alpha_mode=(source.alphaMode or "OPAQUE").lower(),
        alpha_cutoff=(source.alphaCutoff if source.alphaCutoff is not None else 0.5),
        emission_two_sided=bool(source.doubleSided),
        base_color_texture=base_texture,
        base_color_transform=base_transform,
        metallic_roughness_texture=metallic_texture,
        metallic_roughness_transform=metallic_transform,
        emissive_texture=emissive_texture,
        emissive_transform=emissive_transform,
        normal_texture=normal_texture,
        normal_transform=normal_transform,
        normal_scale=(
            source.normalTexture.scale
            if source.normalTexture is not None
            and source.normalTexture.scale is not None else 1.0
        ),
        occlusion_texture=occlusion_texture,
        occlusion_transform=occlusion_transform,
        occlusion_strength=(
            source.occlusionTexture.strength
            if source.occlusionTexture is not None
            and source.occlusionTexture.strength is not None else 1.0
        ),
        transmission_texture=transmission_texture,
        transmission_transform=transmission_transform,
        thickness_texture=thickness_texture,
        thickness_transform=thickness_transform,
        clearcoat_texture=clearcoat_texture,
        sheen_texture=sheen_texture,
        anisotropy_texture=anisotropy_texture,
        program=(unlit_material if "KHR_materials_unlit" in extensions else None),
    )


def load(path):
    """Load the active glTF scene as shared primitives and node instances."""
    path = Path(path)
    gltf = GLTF2().load(str(path))
    required_extensions = set(gltf.extensionsRequired or ())
    unsupported_required = sorted(required_extensions - _SUPPORTED_EXTENSIONS)
    if unsupported_required:
        raise ValueError(
            "glTF requires unsupported extensions: "
            + ", ".join(unsupported_required)
        )
    buffers = [_buffer_bytes(gltf, index, path) for index in range(len(gltf.buffers))]
    textures = _textures(gltf, buffers, path)
    materials = [_material(gltf, index, textures) for index in range(len(gltf.materials))]
    used_extensions = set(gltf.extensionsUsed or ())
    scene = Scene(metadata={
        "source": "gltf",
        "path": str(path),
        "extensions_used": sorted(used_extensions),
        "extensions_required": sorted(required_extensions),
        "unsupported_optional_extensions": sorted(
            used_extensions - _SUPPORTED_EXTENSIONS
        ),
    })
    primitive_resources = {}
    primitive_morph_targets = {}
    primitive_skin_attributes = {}
    scene_nodes = {}
    node_instances = {}
    skinned_meshes = {
        node.mesh for node in gltf.nodes
        if node.mesh is not None and node.skin is not None
    }
    root_extensions = gltf.extensions or {}
    punctual_lights = root_extensions.get("KHR_lights_punctual", {}).get(
        "lights", ()
    )

    def primitive_resource(mesh_index, primitive_index):
        """Decode one glTF primitive once, even when many nodes place it."""
        key = (mesh_index, primitive_index)
        cached = primitive_resources.get(key)
        if cached is not None:
            return cached
        primitive = gltf.meshes[mesh_index].primitives[primitive_index]
        if primitive.mode not in (None, 4):
            return None
        position_index = primitive.attributes.POSITION
        if position_index is None:
            return None
        vertices = _accessor_array(
            gltf, position_index, buffers
        ).astype(np.float32)
        normals = None
        normal_index = primitive.attributes.NORMAL
        if normal_index is not None:
            normals = _accessor_array(
                gltf, normal_index, buffers
            ).astype(np.float32)
        texcoords = None
        texcoord_index = primitive.attributes.TEXCOORD_0
        if texcoord_index is not None:
            texcoords = _accessor_array(
                gltf, texcoord_index, buffers
            ).astype(np.float32)
        texcoords1 = None
        texcoord1_index = primitive.attributes.TEXCOORD_1
        if texcoord1_index is not None:
            texcoords1 = _accessor_array(
                gltf, texcoord1_index, buffers
            ).astype(np.float32)
        tangents = None
        tangent_index = primitive.attributes.TANGENT
        if tangent_index is not None:
            tangents = _accessor_array(
                gltf, tangent_index, buffers
            ).astype(np.float32)
        joint_indices = None
        joint_weights = None
        joints_index = primitive.attributes.JOINTS_0
        weights_index = primitive.attributes.WEIGHTS_0
        if joints_index is not None or weights_index is not None:
            if joints_index is None or weights_index is None:
                raise ValueError("glTF skin primitives require JOINTS_0 and WEIGHTS_0")
            joint_indices = _accessor_array(
                gltf, joints_index, buffers
            ).astype(np.int32)
            joint_weights = _accessor_array(
                gltf, weights_index, buffers
            ).astype(np.float32)
            if joint_indices.shape != (len(vertices), 4):
                raise ValueError("glTF JOINTS_0 must be VEC4")
            if joint_weights.shape != (len(vertices), 4):
                raise ValueError("glTF WEIGHTS_0 must be VEC4")
        if primitive.indices is None:
            indices = np.arange(len(vertices), dtype=np.uint32)
        else:
            indices = _accessor_array(
                gltf, primitive.indices, buffers
            ).reshape(-1)
        if len(indices) % 3:
            raise ValueError(
                "Triangle primitive index count is not divisible by three"
            )
        material = (
            materials[primitive.material]
            if primitive.material is not None else Material()
        )
        morph_targets = []
        for target_index, target in enumerate(primitive.targets or ()):
            position_target = (
                target.get("POSITION") if isinstance(target, dict)
                else target.POSITION
            )
            if position_target is None:
                position_deltas = np.zeros_like(vertices)
            else:
                position_deltas = _accessor_array(
                    gltf, position_target, buffers
                ).astype(np.float32)
            normal_target = (
                target.get("NORMAL") if isinstance(target, dict)
                else target.NORMAL
            )
            normal_deltas = None if normal_target is None else _accessor_array(
                gltf, normal_target, buffers
            ).astype(np.float32)
            morph_targets.append(MorphTarget(
                position_deltas, normal_deltas,
                name=f"morph-{mesh_index}-{primitive_index}-{target_index}",
            ))
        resource = scene.create_mesh(
            vertices, indices.astype(np.uint32).reshape((-1, 3)), material,
            normals=normals, texcoords=texcoords,
            texcoords1=texcoords1, tangents=tangents,
            deformable=bool(morph_targets) or mesh_index in skinned_meshes, name=(
                f"{gltf.meshes[mesh_index].name or 'mesh'}:{primitive_index}"
            ),
            metadata={
                "source": "gltf",
                "mesh_index": mesh_index,
                "primitive_index": primitive_index,
            },
        )
        primitive_resources[key] = resource
        primitive_morph_targets[key] = tuple(morph_targets)
        primitive_skin_attributes[key] = (joint_indices, joint_weights)
        return resource

    def visit(node_index, parent=None, parent_node_index=None):
        node = gltf.nodes[node_index]
        metadata = {"source": "gltf", "node_index": node_index}
        if node.extras is not None:
            metadata["extras"] = node.extras
        scene_node = scene.add_node(
            transform=Transform(_node_matrix(node)), parent=parent,
            name=node.name, metadata=metadata,
        )
        scene_nodes[node_index] = scene_node
        world = scene_node.world_matrix
        node_extensions = node.extensions or {}
        punctual = node_extensions.get("KHR_lights_punctual")
        if punctual is not None:
            source = punctual_lights[punctual["light"]]
            light_type = source.get("type")
            color = tuple(source.get("color", (1.0, 1.0, 1.0)))
            intensity = float(source.get("intensity", 1.0))
            position = tuple(world[:3, 3])
            direction = world[:3, :3] @ np.asarray((0.0, 0.0, -1.0))
            direction /= np.linalg.norm(direction)
            if light_type == "point":
                scene.add_point_light(
                    position, color, intensity, range=source.get("range")
                )
            elif light_type == "directional":
                scene.add_directional_light(tuple(direction), color, intensity)
            elif light_type == "spot":
                spot = source.get("spot") or {}
                scene.add_spot_light(
                    position, tuple(direction), color, intensity,
                    inner_cone_angle=float(spot.get("innerConeAngle", 0.0)),
                    outer_cone_angle=float(
                        spot.get("outerConeAngle", np.pi / 4.0)
                    ),
                    range=source.get("range"),
                )
            else:
                raise ValueError(f"Unsupported glTF punctual light type: {light_type!r}")
        if node.mesh is not None:
            for primitive_index, _primitive in enumerate(
                gltf.meshes[node.mesh].primitives
            ):
                resource = primitive_resource(node.mesh, primitive_index)
                if resource is not None:
                    metadata = {
                        "source": "gltf",
                        "node_index": node_index,
                        "parent_node_index": parent_node_index,
                        "mesh_index": node.mesh,
                        "primitive_index": primitive_index,
                    }
                    if node.extras is not None:
                        metadata["extras"] = node.extras
                    instance = scene.add_instance(
                        resource, node=scene_node, name=node.name,
                        metadata=metadata,
                    )
                    node_instances.setdefault(node_index, []).append(instance)
                    morph_targets = primitive_morph_targets.get(
                        (node.mesh, primitive_index), ()
                    )
                    if morph_targets:
                        initial_weights = (
                            node.weights
                            if node.weights is not None
                            else gltf.meshes[node.mesh].weights
                        )
                        scene.bind_morph_targets(
                            instance, morph_targets, initial_weights
                        )
        for child in node.children or ():
            visit(child, scene_node, node_index)

    scene_index = gltf.scene if gltf.scene is not None else 0
    for root in gltf.scenes[scene_index].nodes or ():
        visit(root)
    skins = []
    for skin_index, source_skin in enumerate(gltf.skins or ()):
        try:
            joints = tuple(scene_nodes[index] for index in source_skin.joints)
        except KeyError as exc:
            raise ValueError(
                f"glTF skin {skin_index} references a joint outside the active scene"
            ) from exc
        if source_skin.inverseBindMatrices is None:
            inverse_bind_matrices = np.repeat(
                np.eye(4, dtype=np.float32)[None, ...], len(joints), axis=0
            )
        else:
            packed = _accessor_array(
                gltf, source_skin.inverseBindMatrices, buffers
            ).astype(np.float32)
            if packed.shape != (len(joints), 16):
                raise ValueError("glTF inverse bind matrices must be MAT4")
            inverse_bind_matrices = packed.reshape((-1, 4, 4)).transpose(0, 2, 1)
        skins.append(Skin(
            joints, inverse_bind_matrices,
            name=source_skin.name or f"skin-{skin_index}",
        ))
    for node_index, node in enumerate(gltf.nodes):
        if node.skin is None or node.mesh is None:
            continue
        instances = node_instances.get(node_index, ())
        for instance in instances:
            primitive_index = instance.metadata["primitive_index"]
            joint_indices, joint_weights = primitive_skin_attributes[
                (node.mesh, primitive_index)
            ]
            if joint_indices is None:
                raise ValueError(
                    f"glTF skinned node {node_index} primitive {primitive_index} "
                    "has no joint attributes"
                )
            scene.bind_skin(
                instance, skins[node.skin], joint_indices, joint_weights,
                mesh_node=scene_nodes[node_index],
            )
    interpolation_names = {
        "LINEAR": "linear", "STEP": "step", "CUBICSPLINE": "cubic",
    }
    for animation_index, animation in enumerate(gltf.animations or ()):
        tracks = []
        for channel in animation.channels:
            sampler = animation.samplers[channel.sampler]
            node_index = channel.target.node
            target = scene_nodes.get(node_index)
            if target is None:
                continue
            times = _accessor_array(gltf, sampler.input, buffers).reshape(-1)
            values = _accessor_array(gltf, sampler.output, buffers)
            interpolation = interpolation_names.get(
                sampler.interpolation or "LINEAR"
            )
            if interpolation is None:
                raise ValueError(
                    f"Unsupported glTF animation interpolation: "
                    f"{sampler.interpolation!r}"
                )
            if channel.target.path == "weights":
                row_count = len(times) * (3 if interpolation == "cubic" else 1)
                if values.size % row_count:
                    raise ValueError("glTF morph animation output size is invalid")
                values = values.reshape((row_count, values.size // row_count))
            tracks.append(AnimationTrack(
                target, channel.target.path, times, values, interpolation,
            ))
        if tracks:
            scene.add_animation(AnimationClip(
                tracks, name=animation.name or f"animation-{animation_index}"
            ))
    if not scene.meshes:
        raise ValueError(f"No triangle meshes found in {path}")
    return scene


# Descriptive alias for discovery from ``ordinarylight.loaders``.
load_gltf = load


__all__ = ["load", "load_gltf"]
