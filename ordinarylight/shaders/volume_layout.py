"""Mechanical resource ABI and variant guards. Algorithms are typed helpers."""
LAYOUT = r'''// OrdinaryLight volume transport ABI v1; bindings controlled by WAVE_VOLUME_* macros.
#include "native_intersection.glsl"
#ifndef WAVE_VOLUME_HEADER_BINDING
#define WAVE_VOLUME_HEADER_BINDING 25
#define WAVE_VOLUME_SCALAR_BINDING 26
#define WAVE_VOLUME_TRANSFER_BINDING 27
#define WAVE_VOLUME_TRIANGLE_BINDING 28
#define WAVE_VOLUME_SAMPLER_BINDING 29
#endif

#ifndef WAVE_OVERLAPPING_VOLUMES
#define WAVE_OVERLAPPING_VOLUMES 0
#endif
#ifndef WAVE_VOLUME_SCATTERING
#define WAVE_VOLUME_SCATTERING 0
#endif
#ifndef WAVE_VOLUME_MULTIPLE_SCATTERING
#define WAVE_VOLUME_MULTIPLE_SCATTERING 0
#endif
#ifndef WAVE_VOLUME_EMPTY_SPACE_SKIPPING
#define WAVE_VOLUME_EMPTY_SPACE_SKIPPING 0
#endif

struct VolumeHeader {
    mat4 world_to_local;
    uvec4 dimensions_offset;
    vec4 value_parameters;
    vec4 render_parameters;
    vec4 scattering_parameters;
    vec4 phase_parameters;
    vec4 multiple_scattering_parameters;
    uvec4 acceleration_parameters;
    uvec4 clip_parameters;
    vec4 clip_planes[8];
};

layout(set = 0, binding = WAVE_VOLUME_HEADER_BINDING, std430)
readonly buffer VolumeHeaderBuffer { VolumeHeader volume_headers[]; };
layout(set = 0, binding = WAVE_VOLUME_SCALAR_BINDING, std430)
readonly buffer VolumeScalarBuffer { float volume_scalars[]; };
layout(set = 0, binding = WAVE_VOLUME_TRANSFER_BINDING, std430)
readonly buffer VolumeTransferBuffer { vec4 volume_transfer[]; };
layout(set = 0, binding = WAVE_VOLUME_TRIANGLE_BINDING, std430)
readonly buffer TriangleVolumeBuffer { uint triangle_volumes[]; };
layout(set = 0, binding = WAVE_VOLUME_SAMPLER_BINDING)
uniform sampler3D volume_textures[16];

@isVolumePrimitive@

@volumeInterval@

@volumeSliceDistances@

@sampleVolumeTexture@

@volumeScalar@

@volumeTransferSample@

#if WAVE_VOLUME_EMPTY_SPACE_SKIPPING
@volumeBrickIndexFromVoxel@

@volumeBrickOccupiedAtVoxel@

@volumeVoxelRay@

@volumeBrickExitDistanceAtVoxel@

@volumeBrickOccupied@

@volumeBrickExitDistance@
#endif

#if WAVE_VOLUME_SCATTERING
vec3 environmentColor(vec3 direction);

@volumePhase@

@volumeOpaqueVisibility@

@approximateVolumeLightTransmittance@

@volumePointScattering@
#endif

@integrateVolumeUntil@

@integrateVolume@

#if WAVE_OVERLAPPING_VOLUMES
@integrateOverlappingVolumesBeforeSurface@
#endif

@integrateVolumesBeforeSurface@

@volumeShadowTransmittance@
'''
