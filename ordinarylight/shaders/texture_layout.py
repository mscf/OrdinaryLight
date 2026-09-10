"""Mechanical resource ABI and variant guards. Algorithms are typed helpers."""
LAYOUT = r'''#include "wavefront_srgb.glsl"

#define ordinarylight_output_queue_count output_queue.count
#define ordinarylight_output_queue output_queue
#define ordinarylight_vertices vertices
#define ordinarylight_attributes attributes
#define ordinarylight_materials materials
#define ordinarylight_medium_stacks stacks
#define ordinarylight_paths paths
#define ordinarylight_secondary_paths secondary_paths
#include "ordinaryshade_primary.glsl"
#undef ordinarylight_secondary_paths
#undef ordinarylight_paths
#undef ordinarylight_medium_stacks
#undef ordinarylight_materials
#undef ordinarylight_attributes
#undef ordinarylight_vertices
#undef ordinarylight_output_queue
#undef ordinarylight_output_queue_count

@wrapTextureCoordinate@

@wrapTextureIndex@

@decodeTextureTexel@

@fetchTextureTexel@

@textureMipOffset@

@sampleTextureLevel@

@sampleSceneTexture@

@sampleMaterialTexture@

@triangleUvDensity@

@materialHasTextures@

@applyMaterialTextures@

@applyNormalTexture@

@textureBindingUsesUv1@

@materialUsesUv1@

@triangleTangent@
'''
