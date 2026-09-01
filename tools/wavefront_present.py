"""Direct Vulkan swapchain presentation of the wavefront path."""

import csv
from dataclasses import replace
import json
import math
import os
import statistics
import time
from pathlib import Path

import numpy as np
import ordinarylight as ol
from ordinarylight.integrations.glfw_platform import load_glfw
from ordinarylight.integrations.resize import ResizeRecreationGate
from ordinarylight.validation import performance_gate_result


glfw = load_glfw()

from ordinarylight.showcases.vertex_attributes import build_vertex_attribute_showcase


def _workload_analysis(samples):
    """Return correlation and slow/fast workload comparisons."""
    usable = [sample for sample in samples if sample.get("work")]
    if len(usable) < 3:
        return []
    names = tuple(usable[0]["work"])
    gpu = np.asarray([sample["gpu"] for sample in usable], dtype=np.float64)
    ordered = sorted(usable, key=lambda sample: sample["gpu"])
    group_size = max(1, len(ordered) // 10)
    fast = ordered[:group_size]
    slow = ordered[-group_size:]
    lines = []
    for name in names:
        values = np.asarray(
            [sample["work"].get(name, 0) for sample in usable],
            dtype=np.float64,
        )
        correlation = (
            float(np.corrcoef(gpu, values)[0, 1])
            if np.ptp(gpu) > 0.0 and np.ptp(values) > 0.0 else 0.0
        )
        fast_mean = statistics.mean(sample["work"].get(name, 0) for sample in fast)
        slow_mean = statistics.mean(sample["work"].get(name, 0) for sample in slow)
        delta = (
            (slow_mean / fast_mean - 1.0) * 100.0 if fast_mean else 0.0
        )
        lines.append(
            f"{name}:corr={correlation:+.3f},slow_delta={delta:+.1f}%"
        )
    path_rays = [sample["work"].get("path_rays", 0) for sample in usable]
    total_paths = sum(path_rays)
    if total_paths:
        lines.append(
            f"aggregate_gpu_cost={sum(sample['gpu'] for sample in usable) * 1e6 / total_paths:.2f}ns/path_ray"
        )
    return lines


def _write_benchmark_csv(path, samples):
    if not samples:
        return
    work_names = sorted({
        name for sample in samples for name in sample.get("work", {})
    })
    stage_names = sorted({
        name for sample in samples for name in sample.get("stages", {})
    })
    fields = ["frame", "time", "gpu_ms", "cadence_ms", "wait_ms"]
    fields += [f"work_{name}" for name in work_names]
    fields += [f"stage_{name}_ms" for name in stage_names]
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            row = {
                "frame": sample["frame"], "time": sample["time"],
                "gpu_ms": sample["gpu"], "cadence_ms": sample["cadence"],
                "wait_ms": sample["wait"],
            }
            row.update({
                f"work_{name}": sample.get("work", {}).get(name, 0)
                for name in work_names
            })
            row.update({
                f"stage_{name}_ms": sample.get("stages", {}).get(name, 0.0)
                for name in stage_names
            })
            writer.writerow(row)


def _restir_reuse_label(timings):
    if not timings.get("wavefront_restir_di"):
        return ""
    counters = timings.get("wavefront_work_counters", {})
    accepted = counters.get("restir_history_accepted", 0)
    rejected = counters.get("restir_history_rejected", 0)
    attempted = accepted + rejected
    if not attempted:
        return ""
    return f"reuse {100.0 * accepted / attempted:.0f}% | "


def main():
    if not glfw.init():
        raise RuntimeError("GLFW initialization failed")
    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
    if os.environ.get("WAVE_RENDER_DECORATED", "1").lower() in {
        "0", "false", "no", "off"
    }:
        glfw.window_hint(glfw.DECORATED, glfw.FALSE)
    initial_width = int(os.environ.get("WAVE_RENDER_WIDTH", "1280"))
    initial_height = int(os.environ.get("WAVE_RENDER_HEIGHT", "720"))
    window = glfw.create_window(
        initial_width, initial_height, "ordinarylight wavefront", None, None
    )
    if not window:
        glfw.terminate()
        raise RuntimeError("GLFW Vulkan window creation failed")
    if os.environ.get("WAVE_RENDER_MAXIMIZE", "0").lower() not in {
        "0", "false", "no", "off"
    }:
        glfw.maximize_window(window)
        glfw.poll_events()
    scene_value = os.environ.get("WAVE_RENDER_SCENE", "attributes")
    scene_name = scene_value.lower()
    scene_is_file = Path(scene_value).is_file()
    presentation_arc_radians = None
    if scene_is_file:
        scene = ol.load_gltf(scene_value)
        bounds_min, bounds_max = scene.bounds()
        scene_center = (bounds_min + bounds_max) * 0.5
        scene_radius = max(float(np.linalg.norm(bounds_max - bounds_min)) * 0.5, 0.5)
        orbit_radius = scene_radius * 2.1
        camera_height = float(scene_center[1] + scene_radius * 0.35)
        camera_target = tuple(float(value) for value in scene_center)
        print(
            f"Loaded {len(scene.meshes)} primitives, "
            f"{sum(len(mesh.indices) for mesh in scene.meshes):,} triangles, "
            f"{len(scene.textures)} textures"
        )
    elif scene_name == "opaque":
        # Parameter-based counterpart to the diffuse regression scene.  It
        # exercises exact opaque shader specialization without custom
        # SurfaceResponse programs obscuring the eligibility contract.
        from ordinarylight.showcases.rooms import build_diffuse_room
        scene = build_diffuse_room()
        for mesh in scene.meshes:
            mesh.material = replace(mesh.material, program=None)
        orbit_radius, camera_height, camera_target = 8.5, 3.2, (0.0, 1.25, 0.0)
    elif scene_name == "attributes":
        scene = build_vertex_attribute_showcase()
        orbit_radius, camera_height, camera_target = 8.5, 3.2, (0.0, 1.25, 0.0)
    elif scene_name == "primitives":
        from ordinarylight.showcases.primitives import build_primitive_showcase
        scene = build_primitive_showcase()
        orbit_radius, camera_height, camera_target = 9.0, 3.8, (0.0, 1.4, 0.0)
        presentation_arc_radians = None
    elif scene_name == "volumes":
        from ordinarylight.showcases.volumes import build_volume_showcase
        scene = build_volume_showcase()
        orbit_radius, camera_height, camera_target = 7.5, 3.2, (0.0, 1.45, 0.0)
        presentation_arc_radians = None
    elif scene_name in {"multivolume", "multivolumes"}:
        from ordinarylight.showcases.multivolume import build_multivolume_showcase
        scene = build_multivolume_showcase()
        orbit_radius, camera_height, camera_target = 7.2, 3.0, (0.0, 1.35, 0.0)
        presentation_arc_radians = None
    elif scene_name in {"volume-scattering", "volume_scattering"}:
        from ordinarylight.showcases.volume_scattering import build_volume_scattering_showcase
        scene = build_volume_scattering_showcase()
    elif scene_name in {
        "volume-multiple-scattering", "volume_multiple_scattering",
    }:
        from ordinarylight.showcases.volume_multiple_scattering import (
            build_volume_multiple_scattering_showcase,
        )
        scene = build_volume_multiple_scattering_showcase()
        orbit_radius, camera_height, camera_target = 7.4, 3.0, (0.0, 1.35, 0.0)
        presentation_arc_radians = None
    else:
        from ordinarylight.showcases.rooms import SCENES
        if scene_name not in SCENES:
            raise ValueError(
                f"unknown WAVE_RENDER_SCENE={scene_value!r}; expected "
                f"attributes, primitives, volumes, multivolumes, "
                f"volume-scattering, volume-multiple-scattering, opaque, "
                f"a file path, or one of "
                f"{tuple(SCENES)}"
            )
        scene_spec = SCENES[scene_name]
        scene = scene_spec.build()
        # The procedural rooms are open on -Z; a negative radius keeps the
        # existing orbit equations while starting in front of their opening.
        orbit_radius = -scene_spec.orbit_radius
        camera_height = scene_spec.camera_height
        camera_target = scene_spec.target
        presentation_arc_radians = scene_spec.presentation_arc_radians
    if os.environ.get("WAVE_RENDER_FULL_ORBIT", "0").lower() not in {
        "0", "false", "no", "off"
    }:
        presentation_arc_radians = None
    benchmark_frames = int(os.environ.get("WAVE_RENDER_BENCHMARK_FRAMES", "0"))
    instancing_gate = os.environ.get(
        "WAVE_RENDER_INSTANCING_GATE", "0"
    ).lower() not in {"0", "false", "no", "off"}
    expected_instancing = scene.instancing_statistics()
    if instancing_gate:
        minimum_savings = int(os.environ.get(
            "WAVE_RENDER_INSTANCING_GATE_MIN_SHARED_BLAS_SAVINGS", "1"
        ))
        if expected_instancing["shared_blas_savings"] < minimum_savings:
            raise RuntimeError(
                "instancing gate requires at least "
                f"{minimum_savings} shared BLAS placements; scene provides "
                f"{expected_instancing['shared_blas_savings']}"
            )
        print("Instancing scene: " + " ".join(
            f"{name}={value:,}" for name, value in expected_instancing.items()
        ))
    benchmark_warmup_frames = max(0, int(os.environ.get(
        "WAVE_RENDER_BENCHMARK_WARMUP_FRAMES", "10"
    )))
    benchmark_report_interval = max(1, int(os.environ.get(
        "WAVE_RENDER_BENCHMARK_REPORT_INTERVAL", "30"
    )))
    benchmark_csv = os.environ.get("WAVE_RENDER_BENCHMARK_CSV")
    benchmark_orbits = float(os.environ.get("WAVE_RENDER_BENCHMARK_ORBITS", "1"))
    performance_gate = os.environ.get(
        "WAVE_RENDER_PERFORMANCE_GATE", "0"
    ).lower() not in {"0", "false", "no", "off"}
    gate_minimum_fps = float(os.environ.get(
        "WAVE_RENDER_PERFORMANCE_GATE_MIN_FPS", "50"
    ))
    gate_minimum_pixel_ratio = float(os.environ.get(
        "WAVE_RENDER_PERFORMANCE_GATE_MIN_PIXEL_RATIO", "0.98"
    ))
    gate_target_width = int(os.environ.get(
        "WAVE_RENDER_PERFORMANCE_GATE_TARGET_WIDTH", str(initial_width)
    ))
    gate_target_height = int(os.environ.get(
        "WAVE_RENDER_PERFORMANCE_GATE_TARGET_HEIGHT", str(initial_height)
    ))
    gate_allow_failure = os.environ.get(
        "WAVE_RENDER_PERFORMANCE_GATE_ALLOW_FAILURE", "0"
    ).lower() not in {"0", "false", "no", "off"}
    gate_override_reason = os.environ.get(
        "WAVE_RENDER_PERFORMANCE_GATE_OVERRIDE_REASON", ""
    )
    benchmark_summary = os.environ.get("WAVE_RENDER_BENCHMARK_SUMMARY")
    if performance_gate and benchmark_frames <= benchmark_warmup_frames + 1:
        raise ValueError(
            "performance gate requires at least two measured frames after "
            "benchmark warmup"
        )
    if gate_minimum_fps <= 0.0:
        raise ValueError("performance gate minimum FPS must be positive")
    if not 0.0 < gate_minimum_pixel_ratio <= 1.0:
        raise ValueError("performance gate pixel ratio must be in (0, 1]")
    if gate_allow_failure and not gate_override_reason.strip():
        raise ValueError(
            "performance gate override requires "
            "WAVE_RENDER_PERFORMANCE_GATE_OVERRIDE_REASON"
        )
    orbit_phase = float(os.environ.get("WAVE_RENDER_ORBIT_PHASE", "0"))
    dynamic_resolution = os.environ.get(
        "WAVE_RENDER_DYNAMIC_RESOLUTION", "0"
    ).lower() not in {"0", "false", "no", "off"}
    indirect_debug_view = os.environ.get(
        "WAVE_RENDER_INDIRECT_REUSE_DEBUG_VIEW", "off"
    ).lower()
    indirect_apply = os.environ.get(
        "WAVE_RENDER_INDIRECT_REUSE_APPLY", "0"
    ).lower() not in {"0", "false", "no", "off"}
    debug_candidates = indirect_debug_view != "off" or indirect_apply
    debug_acceptance = indirect_debug_view == "acceptance"
    indirect_storage = os.environ.get(
        "WAVE_RENDER_INDIRECT_REUSE_STORAGE",
        "1" if debug_candidates else "0",
    ).lower() not in {"0", "false", "no", "off"}
    indirect_candidates = os.environ.get(
        "WAVE_RENDER_INDIRECT_REUSE_CANDIDATES",
        "1" if debug_candidates else "0",
    ).lower() not in {"0", "false", "no", "off"}
    indirect_temporal = os.environ.get(
        "WAVE_RENDER_INDIRECT_REUSE_TEMPORAL",
        "1" if debug_acceptance else "0",
    ).lower() not in {"0", "false", "no", "off"}
    indirect_spatial = os.environ.get(
        "WAVE_RENDER_INDIRECT_REUSE_SPATIAL",
        "1" if debug_acceptance else "0",
    ).lower() not in {"0", "false", "no", "off"}
    config = ol.RendererConfig(
        vulkan_pipeline_cache=os.environ.get(
            "WAVE_RENDER_PIPELINE_CACHE", "1"
        ).lower() not in {"0", "false", "no", "off"},
        vulkan_pipeline_cache_path=(
            os.environ.get("WAVE_RENDER_PIPELINE_CACHE_PATH") or None
        ),
        volume_empty_space_skipping=os.environ.get(
            "WAVE_RENDER_VOLUME_EMPTY_SPACE_SKIPPING", "1"
        ).lower() not in {"0", "false", "no", "off"},
        swapchain_images=int(os.environ.get(
            "WAVE_RENDER_SWAPCHAIN_IMAGES", "0"
        )),
        present_mode=os.environ.get("WAVE_RENDER_PRESENT_MODE", "mailbox"),
        max_bounces=int(os.environ.get("WAVE_RENDER_MAX_BOUNCES", "5")),
        samples_per_pixel=int(os.environ.get("WAVE_RENDER_SAMPLES", "1")),
        area_light_samples=int(os.environ.get(
            "WAVE_RENDER_AREA_LIGHT_SAMPLES", "1"
        )),
        wavefront_secondary_area_light_samples=int(os.environ.get(
            "WAVE_RENDER_SECONDARY_AREA_LIGHT_SAMPLES", "0"
        )),
        wavefront_environment_samples=int(os.environ.get(
            "WAVE_RENDER_ENVIRONMENT_SAMPLES", "1"
        )),
        wavefront_secondary_nee_probability=float(os.environ.get(
            "WAVE_RENDER_SECONDARY_NEE_PROBABILITY", "1.0"
        )),
        wavefront_unified_secondary_nee=os.environ.get(
            "WAVE_RENDER_UNIFIED_SECONDARY_NEE", "1"
        ).lower() not in {"0", "false", "no", "off"},
        wavefront_unified_primary_restir=os.environ.get(
            "WAVE_RENDER_UNIFIED_PRIMARY_RESTIR", "0"
        ).lower() not in {"0", "false", "no", "off"},
        wavefront_stratified_primary_restir=os.environ.get(
            "WAVE_RENDER_STRATIFIED_PRIMARY_RESTIR", "0"
        ).lower() not in {"0", "false", "no", "off"},
        wavefront_restir_di=os.environ.get(
            "WAVE_RENDER_RESTIR_DI", "0"
        ).lower() not in {"0", "false", "no", "off"},
        wavefront_restir_reservoirs=int(os.environ.get(
            "WAVE_RENDER_RESTIR_RESERVOIRS", "1"
        )),
        wavefront_restir_candidates=int(os.environ.get(
            "WAVE_RENDER_RESTIR_CANDIDATES", "1"
        )),
        wavefront_restir_history_limit=int(os.environ.get(
            "WAVE_RENDER_RESTIR_HISTORY_LIMIT", "20"
        )),
        wavefront_restir_history_motion_pixels=float(os.environ.get(
            "WAVE_RENDER_RESTIR_HISTORY_MOTION_PIXELS", "16"
        )),
        wavefront_restir_spatial_reuse=os.environ.get(
            "WAVE_RENDER_RESTIR_SPATIAL", "0"
        ).lower() not in {"0", "false", "no", "off"},
        wavefront_restir_spatial_neighbors=int(os.environ.get(
            "WAVE_RENDER_RESTIR_SPATIAL_NEIGHBORS", "4"
        )),
        wavefront_restir_spatial_radius=int(os.environ.get(
            "WAVE_RENDER_RESTIR_SPATIAL_RADIUS", "4"
        )),
        wavefront_restir_pairwise_mis=os.environ.get(
            "WAVE_RENDER_RESTIR_PAIRWISE_MIS", "1"
        ).lower() not in {"0", "false", "no", "off"},
        wavefront_restir_generalized_mis=os.environ.get(
            "WAVE_RENDER_RESTIR_GENERALIZED_MIS", "0"
        ).lower() not in {"0", "false", "no", "off"},
        wavefront_restir_generalized_balance_cap=float(os.environ.get(
            "WAVE_RENDER_RESTIR_GENERALIZED_BALANCE_CAP", "2.0"
        )),
        wavefront_restir_specialization=os.environ.get(
            "WAVE_RENDER_RESTIR_SPECIALIZATION", "1"
        ).lower() not in {"0", "false", "no", "off"},
        wavefront_tile_capacity=int(os.environ.get(
            "WAVE_RENDER_TILE_CAPACITY", "131072"
        )),
        wavefront_exposure=float(os.environ.get("WAVE_RENDER_EXPOSURE", "1.0")),
        wavefront_profiling=os.environ.get(
            "WAVE_RENDER_PROFILE", "1" if benchmark_frames else "0"
        ).lower() not in {"0", "false", "no", "off"},
        wavefront_render_scale=float(os.environ.get("WAVE_RENDER_SCALE", "1.0")),
        wavefront_dynamic_resolution=dynamic_resolution,
        wavefront_dynamic_target_ms=float(os.environ.get(
            "WAVE_RENDER_DYNAMIC_TARGET_MS", "16.67"
        )),
        wavefront_dynamic_min_scale=float(os.environ.get(
            "WAVE_RENDER_DYNAMIC_MIN_SCALE", "0.5"
        )),
        wavefront_temporal_reconstruction=os.environ.get(
            "WAVE_RENDER_TEMPORAL", "1" if dynamic_resolution else "0"
        ).lower() not in {"0", "false", "no", "off"},
        wavefront_temporal_weight=float(os.environ.get(
            "WAVE_RENDER_TEMPORAL_WEIGHT", "0.85"
        )),
        wavefront_temporal_variance_confidence=os.environ.get(
            "WAVE_RENDER_TEMPORAL_VARIANCE", "0"
        ).lower() not in {"0", "false", "no", "off"},
        wavefront_temporal_variance_strength=float(os.environ.get(
            "WAVE_RENDER_TEMPORAL_VARIANCE_STRENGTH", "0.5"
        )),
        wavefront_temporal_material_confidence=os.environ.get(
            "WAVE_RENDER_TEMPORAL_MATERIAL_CONFIDENCE", "0"
        ).lower() not in {"0", "false", "no", "off"},
        wavefront_temporal_transmission_history_scale=float(os.environ.get(
            "WAVE_RENDER_TEMPORAL_TRANSMISSION_SCALE", "0.5"
        )),
        wavefront_temporal_reprojection_search=os.environ.get(
            "WAVE_RENDER_TEMPORAL_REPROJECTION_SEARCH", "0"
        ).lower() not in {"0", "false", "no", "off"},
        wavefront_temporal_outlier_confidence=os.environ.get(
            "WAVE_RENDER_TEMPORAL_OUTLIER_CONFIDENCE", "0"
        ).lower() not in {"0", "false", "no", "off"},
        wavefront_temporal_outlier_strength=float(os.environ.get(
            "WAVE_RENDER_TEMPORAL_OUTLIER_STRENGTH", "0.75"
        )),
        wavefront_temporal_motion_limit_pixels=float(os.environ.get(
            "WAVE_RENDER_TEMPORAL_MOTION_LIMIT_PIXELS", "64"
        )),
        wavefront_indirect_reuse_storage=indirect_storage,
        wavefront_indirect_reuse_candidates=indirect_candidates,
        wavefront_indirect_reuse_temporal=indirect_temporal,
        wavefront_indirect_reuse_spatial=indirect_spatial,
        wavefront_indirect_reuse_profiling=os.environ.get(
            "WAVE_RENDER_INDIRECT_REUSE_PROFILING", "0"
        ).lower() not in {"0", "false", "no", "off"},
        wavefront_indirect_reuse_apply=indirect_apply,
        wavefront_indirect_reuse_apply_strength=float(os.environ.get(
            "WAVE_RENDER_INDIRECT_REUSE_APPLY_STRENGTH", "0.35"
        )),
        wavefront_indirect_reuse_history_limit=int(os.environ.get(
            "WAVE_RENDER_INDIRECT_REUSE_HISTORY_LIMIT", "32"
        )),
        wavefront_indirect_reuse_history_motion_pixels=float(os.environ.get(
            "WAVE_RENDER_INDIRECT_REUSE_HISTORY_MOTION_PIXELS", "16"
        )),
        wavefront_indirect_reuse_debug_view=indirect_debug_view,
        wavefront_indirect_reuse_scale=float(os.environ.get(
            "WAVE_RENDER_INDIRECT_REUSE_SCALE", "0.5"
        )),
        wavefront_indirect_reuse_budget_mib=float(os.environ.get(
            "WAVE_RENDER_INDIRECT_REUSE_BUDGET_MIB", "128.0"
        )),
        wavefront_diffuse_filter=os.environ.get(
            "WAVE_RENDER_DIFFUSE_FILTER", "0"
        ).lower() not in {"0", "false", "no", "off"},
        wavefront_diffuse_filter_strength=float(os.environ.get(
            "WAVE_RENDER_DIFFUSE_FILTER_STRENGTH", "0.35"
        )),
        wavefront_russian_roulette=os.environ.get(
            "WAVE_RENDER_RUSSIAN_ROULETTE", "0"
        ).lower() not in {"0", "false", "no", "off"},
        wavefront_russian_roulette_start=int(os.environ.get(
            "WAVE_RENDER_RUSSIAN_ROULETTE_START", "3"
        )),
        wavefront_russian_roulette_min_survival=float(os.environ.get(
            "WAVE_RENDER_RUSSIAN_ROULETTE_MIN_SURVIVAL", "0.1"
        )),
        wavefront_fused_secondary=os.environ.get(
            "WAVE_RENDER_FUSED_SECONDARY", "1"
        ).lower() not in {"0", "false", "no", "off"},
        wavefront_subgroup_enqueue=os.environ.get(
            "WAVE_RENDER_SUBGROUP_ENQUEUE", "1"
        ).lower() not in {"0", "false", "no", "off"},
        direct_swapchain_storage=os.environ.get(
            "WAVE_RENDER_DIRECT_SWAPCHAIN", "1"
        ).lower() not in {"0", "false", "no", "off"},
        wavefront_execution_strategy=os.environ.get(
            "WAVE_RENDER_EXECUTION_STRATEGY", "auto"
        ).lower(),
        wavefront_auto_megakernel_transmission_fraction=float(os.environ.get(
            "WAVE_RENDER_AUTO_MEGAKERNEL_TRANSMISSION_FRACTION", "0.25"
        )),
        wavefront_auto_megakernel_triangle_threshold=int(os.environ.get(
            "WAVE_RENDER_AUTO_MEGAKERNEL_TRIANGLE_THRESHOLD", "16384"
        )),
        wavefront_hybrid_inline_bounces=int(os.environ.get(
            "WAVE_RENDER_HYBRID_INLINE_BOUNCES", "3"
        )),
        wavefront_device_local_textures=os.environ.get(
            "WAVE_RENDER_DEVICE_LOCAL_TEXTURES", "1"
        ).lower() not in {"0", "false", "no", "off"},
        wavefront_native_textures=os.environ.get(
            "WAVE_RENDER_NATIVE_TEXTURES", "0"
        ).lower() not in {"0", "false", "no", "off"},
        wavefront_material_bucketing=os.environ.get(
            "WAVE_RENDER_MATERIAL_BUCKETING", "0"
        ).lower() not in {"0", "false", "no", "off"},
        wavefront_material_bucketing_start_bounce=int(os.environ.get(
            "WAVE_RENDER_MATERIAL_BUCKETING_START_BOUNCE", "2"
        )),
        wavefront_persistent_coarse_tiles=os.environ.get(
            "WAVE_RENDER_PERSISTENT_COARSE_TILES", "0"
        ).lower() not in {"0", "false", "no", "off"},
        wavefront_persistent_continuations=os.environ.get(
            "WAVE_RENDER_PERSISTENT_CONTINUATIONS", "0"
        ).lower() not in {"0", "false", "no", "off"},
        wavefront_scene_specialization=os.environ.get(
            "WAVE_RENDER_SCENE_SPECIALIZATION", "1"
        ).lower() not in {"0", "false", "no", "off"},
        wavefront_ordinaryshade_shade=os.environ.get(
            "WAVE_RENDER_ORDINARYSHADE_SHADE", "1"
        ).lower() not in {"0", "false", "no", "off"},
        wavefront_megakernel_single_warp=os.environ.get(
            "WAVE_RENDER_MEGAKERNEL_SINGLE_WARP", "0"
        ).lower() not in {"0", "false", "no", "off"},
        wavefront_megakernel_group_swizzle=int(os.environ.get(
            "WAVE_RENDER_MEGAKERNEL_GROUP_SWIZZLE", "0"
        )),
        wavefront_ser=os.environ.get(
            "WAVE_RENDER_SER", "0"
        ).lower() not in {"0", "false", "no", "off"},
        wavefront_ser_reorder=os.environ.get(
            "WAVE_RENDER_SER_REORDER", "1"
        ).lower() not in {"0", "false", "no", "off"},
        wavefront_untextured_specialization=os.environ.get(
            "WAVE_RENDER_UNTEXTURED_SPECIALIZATION", "0"
        ).lower() not in {"0", "false", "no", "off"},
    )
    started = time.perf_counter()
    rendered_frames = 0
    benchmark_samples = []
    restir_key_down = False
    gate_exit_code = 0
    resize_gate = ResizeRecreationGate(
        max(
            0.0,
            float(os.environ.get("WAVE_RENDER_RESIZE_SETTLE_MS", "150")) / 1000.0,
        )
    )
    try:
        with ol.VulkanGlfwPresenter(window, config=config) as presenter:
            print(f"Vulkan device: {presenter.device_name}")
            while not glfw.window_should_close(window):
                glfw.poll_events()
                if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
                    glfw.set_window_should_close(window, True)
                    continue
                # Benchmarks must be reproducible even if their window happens
                # to receive keyboard focus while the user is typing.
                restir_pressed = benchmark_frames == 0 and (
                    glfw.get_key(window, glfw.KEY_R) == glfw.PRESS
                )
                if restir_pressed and not restir_key_down:
                    if config.wavefront_restir_di:
                        enabled = presenter.toggle_wavefront_restir()
                        print(
                            "Direct lighting: "
                            + ("temporal ReSTIR" if enabled else "conventional")
                        )
                    else:
                        print(
                            "ReSTIR A/B unavailable: launch with "
                            "WAVE_RENDER_RESTIR_DI=1 to allocate reservoirs"
                        )
                restir_key_down = restir_pressed
                width, height = glfw.get_framebuffer_size(window)
                if width < 1 or height < 1:
                    glfw.wait_events_timeout(0.05)
                    continue
                current_extent = (width, height)
                if not resize_gate.should_render(
                    current_extent, time.perf_counter(),
                    resources_allocated=presenter.swapchain_image_count > 0,
                ):
                    # Native Wayland can emit alternating intermediate extents
                    # while maximizing. Avoid repeatedly allocating complete
                    # HDR/history image sets until the configure sequence has
                    # settled on its final framebuffer size.
                    glfw.wait_events_timeout(min(resize_gate.settle_seconds, 0.01))
                    continue
                if benchmark_frames > benchmark_warmup_frames:
                    measured_index = max(
                        rendered_frames - benchmark_warmup_frames, 0)
                    measured_count = max(
                        benchmark_frames - benchmark_warmup_frames - 1, 1)
                    orbit_progress = orbit_phase + (
                        2.0 * math.pi * benchmark_orbits
                        * measured_index / measured_count
                    )
                else:
                    orbit_progress = (
                        orbit_phase + (time.perf_counter() - started) * 0.22
                    )
                angle = (
                    presentation_arc_radians * math.sin(orbit_progress)
                    if presentation_arc_radians is not None
                    else orbit_progress
                )
                camera = ol.PerspectiveCamera(
                    position=(
                        float(scene_center[0]) + orbit_radius * math.sin(angle)
                        if scene_is_file else orbit_radius * math.sin(angle),
                        camera_height,
                        float(scene_center[2]) + orbit_radius * math.cos(angle)
                        if scene_is_file else orbit_radius * math.cos(angle),
                    ),
                    target=camera_target,
                )
                presenter.present_wavefront(scene, camera, width, height)
                timings = presenter.last_timings
                if instancing_gate and rendered_frames == 0:
                    actual = {
                        "instance_count": int(timings.get("instance_count", -1)),
                        "geometry_count": int(timings.get("blas_count", -1)),
                        "shared_blas_savings": int(
                            timings.get("shared_blas_savings", -1)
                        ),
                    }
                    expected = {
                        name: expected_instancing[name] for name in actual
                    }
                    if actual != expected:
                        raise RuntimeError(
                            "Vulkan instancing gate failed: "
                            f"expected {expected}, got {actual}"
                        )
                    print(
                        "PASS: Vulkan shared-BLAS instancing gate "
                        f"({actual['instance_count']} instances, "
                        f"{actual['geometry_count']} BLAS)"
                    )
                rendered_frames += 1
                if (
                    rendered_frames == 1
                    and os.environ.get(
                        "WAVE_RENDER_MAXIMIZE_AFTER_FRAME", "0"
                    ).lower() not in {"0", "false", "no", "off"}
                ):
                    glfw.maximize_window(window)
                    glfw.poll_events()
                if (benchmark_frames
                        and rendered_frames > benchmark_warmup_frames):
                    benchmark_samples.append({
                        "frame": rendered_frames,
                        "time": time.perf_counter(),
                        "gpu": timings.get("gpu_frame_ms", 0.0),
                        "cadence": timings.get("wavefront_cadence_ms", 0.0),
                        "wait": timings.get("fence_wait_ms", 0.0),
                        "wavefront_temporal_motion_pixels": timings.get(
                            "wavefront_temporal_motion_pixels", 0.0
                        ),
                        "wavefront_restir_effective_history_limit": timings.get(
                            "wavefront_restir_effective_history_limit"
                        ),
                        "wavefront_indirect_reuse_effective_history_limit": (
                            timings.get(
                                "wavefront_indirect_reuse_effective_history_limit"
                            )
                        ),
                        "work": dict(timings.get("wavefront_work_counters", {})),
                        "indirect_reuse": dict(timings.get(
                            "wavefront_indirect_reuse_counters", {}
                        )),
                        "indirect_reuse_metrics": dict(timings.get(
                            "wavefront_indirect_reuse_metrics", {}
                        )),
                        "stages": dict(timings.get("wavefront_stage_ms", {})),
                        "extent": tuple(timings.get(
                            "wavefront_render_extent", (width, height)
                        )),
                    })
                report_benchmark = benchmark_frames and (
                    rendered_frames <= 3
                    or rendered_frames % benchmark_report_interval == 0
                    or rendered_frames == benchmark_frames
                )
                report_indirect = (
                    config.wavefront_indirect_reuse_profiling
                    and (
                        rendered_frames <= 3
                        or rendered_frames % benchmark_report_interval == 0
                    )
                )
                if report_benchmark or report_indirect:
                    print(
                        f"frame={rendered_frames} total="
                        f"{timings.get('wavefront_frame_ms', 0.0):.2f}ms "
                        f"gpu={timings.get('gpu_frame_ms', 0.0):.2f}ms "
                        f"wait={timings.get('fence_wait_ms', 0.0):.2f}ms "
                        f"cadence={timings.get('wavefront_cadence_ms', 0.0):.2f}ms "
                        f"fps={timings.get('wavefront_fps', 0.0):.1f} "
                        f"direct={'yes' if timings.get('direct_swapchain_storage') else 'no'} "
                        f"strategy={timings.get('wavefront_execution_strategy', 'wavefront')} "
                        f"cache={'hit' if timings.get('wavefront_command_cache_hit') else 'build'} "
                        f"output={width}x{height} "
                        f"internal={timings.get('wavefront_render_extent', (width, height))[0]}x"
                        f"{timings.get('wavefront_render_extent', (width, height))[1]} "
                        f"reservoir={timings.get('wavefront_restir_reservoir_bytes', 0) / 1048576:.1f}MiB "
                        f"gbuffer={timings.get('wavefront_gbuffer_bytes', 0) / 1048576:.1f}MiB "
                        f"media={timings.get('wavefront_medium_stack_bytes', 0) / 1048576:.1f}MiB "
                        f"swapchain_images={presenter.swapchain_image_count}"
                    )
                    stages = timings.get("wavefront_stage_ms", {})
                    if stages:
                        print("  " + " ".join(
                            f"{name}={duration:.2f}ms"
                            for name, duration in stages.items()
                        ))
                    counters = timings.get("wavefront_work_counters", {})
                    if counters:
                        print(
                            "  work["
                            f"{timings.get('wavefront_work_counter_scope', 'primary_only')}"
                            "] " + " ".join(
                            f"{name}={value:,}"
                            for name, value in counters.items()
                            )
                        )
                    indirect_counters = timings.get(
                        "wavefront_indirect_reuse_counters", {}
                    )
                    if indirect_counters:
                        print("  indirect " + " ".join(
                            f"{name}={value:,}"
                            for name, value in indirect_counters.items()
                        ))
                        print("  indirect_metrics " + " ".join(
                            f"{name}={value:.3f}"
                            for name, value in timings.get(
                                "wavefront_indirect_reuse_metrics", {}
                            ).items()
                        ))
                    print(
                        "  cpu " + " ".join(
                            f"{name}={timings.get(f'wavefront_{name}_ms', 0.0):.2f}ms"
                            for name in (
                                "scene", "swapchain", "acquire", "record",
                                "submit", "present"
                            )
                        )
                    )
                if benchmark_frames and rendered_frames >= benchmark_frames:
                    if benchmark_samples:
                        gpu = sorted(sample["gpu"] for sample in benchmark_samples)
                        cadence = sorted(
                            sample["cadence"] for sample in benchmark_samples
                            if sample["cadence"] > 0.0
                        )
                        wait = sorted(sample["wait"] for sample in benchmark_samples)
                        elapsed = (benchmark_samples[-1]["time"]
                                   - benchmark_samples[0]["time"])
                        throughput_fps = (
                            (len(benchmark_samples) - 1) / elapsed
                            if elapsed > 0.0 else 0.0
                        )
                        percentile_index = min(
                            len(gpu) - 1, int(0.9 * len(gpu))
                        )
                        print(
                            "SUMMARY "
                            f"frames={len(benchmark_samples)} "
                            f"throughput={throughput_fps:.2f}fps "
                            f"gpu_median={statistics.median(gpu):.2f}ms "
                            f"gpu_p90={gpu[percentile_index]:.2f}ms "
                            f"cadence_median={statistics.median(cadence):.2f}ms "
                            f"wait_median={statistics.median(wait):.2f}ms"
                        )
                        gate_result = performance_gate_result(
                            throughput_fps, gate_minimum_fps,
                            benchmark_samples[-1]["extent"],
                            (gate_target_width, gate_target_height),
                            gate_minimum_pixel_ratio,
                            allow_failure=gate_allow_failure,
                            override_reason=gate_override_reason,
                        )
                        if benchmark_summary:
                            Path(benchmark_summary).write_text(json.dumps({
                                "frames": len(benchmark_samples),
                                "warmup_frames": benchmark_warmup_frames,
                                "gpu_median_ms": statistics.median(gpu),
                                "gpu_p90_ms": gpu[percentile_index],
                                "cadence_median_ms": statistics.median(cadence),
                                "wait_median_ms": statistics.median(wait),
                                "temporal_motion_pixels": benchmark_samples[-1].get(
                                    "wavefront_temporal_motion_pixels", 0.0
                                ),
                                "restir_effective_history_limit": (
                                    benchmark_samples[-1].get(
                                        "wavefront_restir_effective_history_limit"
                                    )
                                ),
                                "indirect_effective_history_limit": (
                                    benchmark_samples[-1].get(
                                        "wavefront_indirect_reuse_effective_history_limit"
                                    )
                                ),
                                "indirect_reuse": benchmark_samples[-1].get(
                                    "indirect_reuse", {}
                                ),
                                "indirect_reuse_metrics": benchmark_samples[-1].get(
                                    "indirect_reuse_metrics", {}
                                ),
                                "gate": gate_result,
                            }, indent=2) + "\n")
                            print(
                                "Wrote benchmark summary: "
                                f"{Path(benchmark_summary).resolve()}"
                            )
                        if performance_gate:
                            print(
                                f"{gate_result['status'].upper()}: "
                                "4K performance gate"
                            )
                            for failure in gate_result["failures"]:
                                print(f"  {failure}")
                            if gate_result["status"] == "override":
                                print(
                                    "  explicit pass: "
                                    f"{gate_result['override_reason']}"
                                )
                            elif gate_result["status"] == "fail":
                                gate_exit_code = 1
                        for line in _workload_analysis(benchmark_samples):
                            print(f"WORKLOAD {line}")
                        if benchmark_csv:
                            _write_benchmark_csv(benchmark_csv, benchmark_samples)
                            print(f"Wrote benchmark trace: {Path(benchmark_csv).resolve()}")
                    glfw.set_window_should_close(window, True)
                glfw.set_window_title(
                    window,
                    f"wavefront {width}x{height} | "
                    f"internal {timings.get('wavefront_render_extent', (width, height))[0]}x"
                    f"{timings.get('wavefront_render_extent', (width, height))[1]} | "
                    f"{timings.get('wavefront_render_scale', 1.0):.2f}x"
                    f"{' DRS' if timings.get('wavefront_dynamic_resolution') else ''} | "
                    f"{'ReSTIR-history | ' if timings.get('wavefront_restir_di') else 'conventional | '}"
                    f"{'spatial | ' if timings.get('wavefront_restir_spatial_reuse') else ''}"
                    f"{_restir_reuse_label(timings)}"
                    f"{'RH ' + str(timings.get('wavefront_restir_effective_history_limit')) + ' | ' if timings.get('wavefront_restir_di') else ''}"
                    f"{'IH ' + str(timings.get('wavefront_indirect_reuse_effective_history_limit')) + ' | ' if config.wavefront_indirect_reuse_apply else ''}"
                    f"{config.samples_per_pixel} spp | "
                    f"{timings.get('wavefront_fps', 0.0):.1f} FPS | "
                    f"GPU {timings.get('gpu_frame_ms', 0.0):.1f} ms | "
                    f"wait {timings.get('fence_wait_ms', 0.0):.1f} ms | "
                    f"{timings.get('present_mode', presenter.present_mode)} | "
                    f"{'direct' if timings.get('direct_swapchain_storage') else 'blit'} | "
                    f"{timings.get('wavefront_execution_strategy', 'wavefront')} | "
                    f"{'bucketed | ' if timings.get('wavefront_material_bucketing') else ''}"
                    f"{timings.get('wavefront_texture_backend', 'packed')} textures | "
                    f"{timings.get('wavefront_tiles', 0)} tiles",
                )
    finally:
        glfw.destroy_window(window)
        glfw.terminate()
    if gate_exit_code:
        raise SystemExit(gate_exit_code)


if __name__ == "__main__":
    main()
