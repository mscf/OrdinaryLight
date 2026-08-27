"""Public API for ordinarylight."""

from .reference import ReferencePathTracer
from . import animations
from .animations import (
    AnimationClip, AnimationPlayer, AnimationTrack, MorphTarget, Skin,
)
from .validation import (
    build_feature_parity_scene,
    feature_parity_camera,
    image_error_metrics,
)
from .scene import (
    Instance, Material, Mesh, MeshResource, Node,
    Scene, Texture, Texture1D,
    Transform, VertexAttributeLayout, Volume, VolumeMaterial,
    TextureTransform,
)
from . import cameras, lights
from .cameras import Camera, OrthographicCamera, PanoramicCamera, PerspectiveCamera
from .lights import DirectionalLight, EnvironmentLight, PointLight, SpotLight
from .surface import ArraySurface, RenderSurface
from . import loaders
from .loaders import load_gltf
from .pipeline import RenderPipeline, RenderStage
from .wavefront import (
    HIT_DTYPE,
    HOT_PATH_STATE_DTYPE,
    MAX_MEDIUM_STACK_DEPTH,
    MEDIUM_STACK_DTYPE,
    PATH_STATE_DTYPE,
    QUEUE_HEADER_DTYPE,
    RESOLVED_PIXEL_DTYPE,
    RAY_DTYPE,
    SECONDARY_PATH_STATE_DTYPE,
    WavefrontQueueLayout,
    create_wavefront_pipeline,
    wavefront_glsl_structs,
)
from .materials import (
    Expression,
    MATERIAL_PARAMETER_LAYOUT,
    SCATTER_ABSORB,
    SCATTER_DIFFUSE,
    SCATTER_REFLECTION,
    SCATTER_TRANSMISSION,
    MaterialContext,
    MaterialEvaluation,
    MaterialProgram,
    SurfaceResponse,
    builtin_material,
    unlit_material,
    dot,
    cosine_sample_hemisphere,
    fresnel_schlick,
    material,
    material_dispatch_glsl,
    maximum,
    mix,
    normalize,
    reflect,
    refract,
    select,
    vec2,
    vec3,
    vec4,
)
from .renderer import RenderFrame, RenderJob, Renderer, RenderStatistics
from .gpu import GpuFrame, VulkanBufferMetadata, VulkanImageMetadata
from .capabilities import RendererCapabilities
from .primitives import (
    GlyphBatch, LineBatch, PointBatch, add_glyphs, add_lines, add_points,
)
from .volume import volume_empty_space_statistics
from . import outputs
from . import backends
from .backends import (
    ProductRenderBackend,
    GpuRenderBackend,
    ReferenceBackend,
    ReferenceConfig,
    RenderBackend,
)


_LAZY_VULKAN_EXPORTS = frozenset({
    "RendererConfig",
    "VulkanDeviceInfo",
    "VulkanGlfwPresenter",
    "VulkanSurfacePresenter",
    "VulkanRayTracingBackend",
    "probe_vulkan_devices",
})


def __getattr__(name):
    """Load optional Vulkan API names only when an application requests them."""
    if name in _LAZY_VULKAN_EXPORTS:
        from .backends import vulkan

        value = getattr(vulkan, name)
        globals()[name] = value
        return value
    raise AttributeError(name)

__all__ = [
    "Material",
    "AnimationClip",
    "AnimationPlayer",
    "AnimationTrack",
    "MorphTarget",
    "Skin",
    "animations",
    "Texture",
    "Texture1D",
    "TextureTransform",
    "Volume",
    "VolumeMaterial",
    "volume_empty_space_statistics",
    "outputs",
    "backends",
    "ReferenceBackend",
    "ReferenceConfig",
    "RenderBackend",
    "ProductRenderBackend",
    "GpuRenderBackend",
    "Transform",
    "VertexAttributeLayout",
    "MATERIAL_PARAMETER_LAYOUT",
    "SCATTER_ABSORB",
    "SCATTER_DIFFUSE",
    "SCATTER_REFLECTION",
    "SCATTER_TRANSMISSION",
    "MaterialContext",
    "MaterialEvaluation",
    "MaterialProgram",
    "SurfaceResponse",
    "Expression",
    "Mesh",
    "MeshResource",
    "Instance",
    "Node",
    "GlyphBatch",
    "PointBatch",
    "LineBatch",
    "add_glyphs",
    "add_points",
    "add_lines",
    "PerspectiveCamera",
    "Camera",
    "OrthographicCamera",
    "PanoramicCamera",
    "PointLight",
    "DirectionalLight",
    "SpotLight",
    "EnvironmentLight",
    "cameras",
    "lights",
    "ArraySurface",
    "ReferencePathTracer",
    "build_feature_parity_scene",
    "feature_parity_camera",
    "image_error_metrics",
    "RenderSurface",
    "RendererConfig",
    "Renderer",
    "RenderFrame",
    "RenderJob",
    "RenderStatistics",
    "GpuFrame",
    "VulkanBufferMetadata",
    "VulkanImageMetadata",
    "RendererCapabilities",
    "RenderPipeline",
    "RenderStage",
    "RAY_DTYPE",
    "HIT_DTYPE",
    "HOT_PATH_STATE_DTYPE",
    "MEDIUM_STACK_DTYPE",
    "PATH_STATE_DTYPE",
    "SECONDARY_PATH_STATE_DTYPE",
    "QUEUE_HEADER_DTYPE",
    "RESOLVED_PIXEL_DTYPE",
    "MAX_MEDIUM_STACK_DEPTH",
    "WavefrontQueueLayout",
    "create_wavefront_pipeline",
    "wavefront_glsl_structs",
    "Scene",
    "VulkanDeviceInfo",
    "VulkanGlfwPresenter",
    "VulkanSurfacePresenter",
    "VulkanRayTracingBackend",
    "probe_vulkan_devices",
    "load_gltf",
    "loaders",
    "builtin_material",
    "unlit_material",
    "dot",
    "cosine_sample_hemisphere",
    "fresnel_schlick",
    "material",
    "material_dispatch_glsl",
    "maximum",
    "mix",
    "normalize",
    "reflect",
    "refract",
    "select",
    "vec2",
    "vec3",
    "vec4",
]
