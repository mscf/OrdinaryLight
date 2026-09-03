"""Public API for ordinarylight."""

from .renderers.reference import ReferencePathTracer
from . import animations
from .animations import (
    AnimationClip, AnimationPlayer, AnimationTrack, MorphTarget, Skin,
)
from .validation import (
    build_feature_parity_scene,
    feature_parity_camera,
    image_error_metrics,
    renderer_visual_metrics,
)
from .scene import (
    Instance, Material, Mesh, MeshResource, Node,
    Scene, Texture, Texture1D,
    Transform, VertexAttributeLayout, Volume, VolumeMaterial,
    TextureTransform,
)
from . import cameras, lights, probes
from .probes import (
    ProbeCaptureManager, capture_reflection_probe, select_reflection_probes,
)
from .cameras import (
    ArcballCameraController, Camera, OrthographicCamera, PanoramicCamera,
    PerspectiveCamera,
)
from .lights import (
    DirectionalLight, EnvironmentLight, PointLight, ReflectionProbe, SpotLight,
)
from .surface import ArraySurface, RenderSurface
from .raster import (
    RasterConfig, RasterMesh, RasterPostProcessor, RasterProgram, RasterState,
    raster_material_hook,
    RasterVertexAttribute, RasterVertexLayout, camera_matrix,
    create_raster_pipeline, rasterize_geometry_products, scene_mesh,
    triangle_mesh, CAMERA_DTYPE, DRAW_DTYPE, LIGHT_DTYPE, MATERIAL_DTYPE,
    SHADOW_DTYPE,
    RasterGpuScene, pack_raster_gpu_scene, ShadowMapRequest, plan_shadow_maps,
)
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
    MaterialLayer,
    LayeredMaterialEvaluation,
    MaterialProgram,
    SurfaceContext,
    SurfaceParameters,
    SurfaceResponse,
    builtin_material,
    unlit_material,
    dot,
    cosine_sample_hemisphere,
    fresnel_schlick,
    material,
    layered_material,
    blend_material_evaluations,
    blend_surface_parameters,
    default_material_modifier,
    material_modifier,
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
from .state import AccumulationState
from .gpu import GpuFrame, VulkanBufferMetadata, VulkanImageMetadata
from .capabilities import RendererCapabilities
from .renderers.selection import RendererSelection, select_vulkan_renderer
from .primitives import (
    GlyphBatch, LineBatch, PointBatch, add_glyphs, add_lines, add_points,
)
from .volume import volume_empty_space_statistics
from .selection import (
    PickOptions, PickResult, ViewportMapping, camera_ray, pick, pick_ray,
)
from . import effects
from . import denoising
from .denoising import (
    DenoiserFrameInfo, DenoiserSignals, PortableDenoiser,
    PortableDenoiserConfig, PortableDenoiserResult, DenoiserQualityBaseline,
    DenoiserQualityMetrics, DenoiserSequenceEvaluation,
    evaluate_denoiser_sequence, SignalValidationError,
)
from . import outputs
from . import renderers
from . import targets
from .renderers import (
    ObjectEffectRendererProtocol,
    MultiObjectEffectRendererProtocol,
    PickRendererProtocol,
    ProductRendererProtocol,
    GpuRendererProtocol,
    RendererProtocol,
    ResidentSceneRendererProtocol,
    RendererImplementation,
    RendererImplementationInfo,
)


_LAZY_VULKAN_EXPORTS = frozenset({
    "RendererConfig",
    "VulkanDeviceInfo",
    "VulkanGlfwPresenter",
    "VulkanSurfacePresenter",
    "probe_vulkan_devices",
})


def __getattr__(name):
    """Load optional Vulkan API names only when an application requests them."""
    if name in _LAZY_VULKAN_EXPORTS:
        from .targets import vulkan

        value = getattr(vulkan, name)
        globals()[name] = value
        return value
    raise AttributeError(name)

__all__ = [
    "Material",
    "MaterialLayer",
    "LayeredMaterialEvaluation",
    "RasterMesh",
    "RasterConfig",
    "raster_material_hook",
    "material_modifier",
    "RasterPostProcessor",
    "RasterProgram",
    "RasterState",
    "RasterVertexAttribute",
    "RasterVertexLayout",
    "camera_matrix",
    "create_raster_pipeline",
    "rasterize_geometry_products",
    "scene_mesh",
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
    "PickResult",
    "PickOptions",
    "ViewportMapping",
    "camera_ray",
    "pick",
    "pick_ray",
    "effects",
    "denoising",
    "DenoiserFrameInfo",
    "DenoiserSignals",
    "PortableDenoiser",
    "PortableDenoiserConfig",
    "PortableDenoiserResult",
    "DenoiserQualityBaseline",
    "DenoiserQualityMetrics",
    "DenoiserSequenceEvaluation",
    "evaluate_denoiser_sequence",
    "SignalValidationError",
    "outputs",
    "renderers",
    "targets",
    "RendererProtocol",
    "RendererImplementation",
    "RendererImplementationInfo",
    "RendererSelection",
    "select_vulkan_renderer",
    "ProductRendererProtocol",
    "GpuRendererProtocol",
    "ObjectEffectRendererProtocol",
    "MultiObjectEffectRendererProtocol",
    "PickRendererProtocol",
    "ResidentSceneRendererProtocol",
    "Transform",
    "VertexAttributeLayout",
    "MATERIAL_PARAMETER_LAYOUT",
    "SCATTER_ABSORB",
    "SCATTER_DIFFUSE",
    "SCATTER_REFLECTION",
    "SCATTER_TRANSMISSION",
    "MaterialContext",
    "MaterialEvaluation",
    "SurfaceContext",
    "SurfaceParameters",
    "blend_material_evaluations",
    "blend_surface_parameters",
    "default_material_modifier",
    "material_modifier",
    "layered_material",
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
    "ArcballCameraController",
    "Camera",
    "OrthographicCamera",
    "PanoramicCamera",
    "PointLight",
    "DirectionalLight",
    "SpotLight",
    "EnvironmentLight",
    "ReflectionProbe",
    "ProbeCaptureManager",
    "capture_reflection_probe",
    "select_reflection_probes",
    "probes",
    "cameras",
    "lights",
    "ArraySurface",
    "ReferencePathTracer",
    "build_feature_parity_scene",
    "feature_parity_camera",
    "image_error_metrics",
    "renderer_visual_metrics",
    "RenderSurface",
    "RasterMesh", "RasterProgram", "triangle_mesh",
    "CAMERA_DTYPE", "DRAW_DTYPE", "LIGHT_DTYPE", "MATERIAL_DTYPE",
    "SHADOW_DTYPE",
    "RasterGpuScene", "pack_raster_gpu_scene", "ShadowMapRequest", "plan_shadow_maps",
    "RendererConfig",
    "Renderer",
    "RenderFrame",
    "RenderJob",
    "RenderStatistics",
    "AccumulationState",
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
