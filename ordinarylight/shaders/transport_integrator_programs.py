"""Typed transport helpers. Generated artifacts must be rebuilt from this source."""
import ordinaryshade as osh
from .transport_programs import MaterialData, MaterialEvaluation, OrdinaryLightSurfaceSample, TransportMaterialRecord, OrdinaryLightHit, OrdinaryLightDielectricEvent, OrdinaryLightEmitterSample

@osh.structure
class SampleAccumulation:
    radiance: osh.vec4
    counts: osh.uvec4
    events: osh.uvec4

@osh.function
def main() -> osh.void:
    i = gl_GlobalInvocationID.x
    if osh.specialization('defined(OL_GPU_REDUCTION)'):
        if gpu_control.z != osh.u32(0) or i >= osh.minimum(pc.count, gpu_control.x):
            return
    elif i >= pc.count:
        return
    input_sample = transport_samples[i]
    owner = input_sample.identity.x
    result = SampleAccumulation(osh.vec4(0), osh.uvec4(0), osh.uvec4(0))
    invalid = (((((osh.any_value(osh.is_nan(input_sample.position)) or osh.any_value(osh.is_inf(input_sample.position))) or osh.any_value(osh.is_nan(input_sample.incoming))) or osh.any_value(osh.is_inf(input_sample.incoming))) or osh.absolute(osh.dot(input_sample.incoming.xyz, input_sample.incoming.xyz) - 1.0) > 0.001) or input_sample.identity.w > osh.u32(1)) or input_sample.media.w >= pc.initial_stack_count
    if input_sample.identity.w == osh.u32(1):
        gn = input_sample.geometric_normal.xyz
        sn = input_sample.shading_normal.xyz
        invalid = (((((((((((invalid or osh.is_nan(input_sample.geometric_normal.w)) or osh.is_inf(input_sample.geometric_normal.w)) or osh.is_nan(input_sample.shading_normal.w)) or osh.is_inf(input_sample.shading_normal.w)) or osh.any_value(osh.is_nan(gn))) or osh.any_value(osh.is_inf(gn))) or osh.any_value(osh.is_nan(sn))) or osh.any_value(osh.is_inf(sn))) or osh.absolute(osh.dot(gn, gn) - 1.0) > 0.001) or osh.absolute(osh.dot(sn, sn) - 1.0) > 0.001) or osh.dot(gn, sn) <= 0.0) or input_sample.identity.z >= OL_MATERIAL_COUNT
        boundary_id = input_sample.media.z
        boundary_index = osh.u32(4294967295)
        if boundary_id != osh.u32(4294967295):
            j = osh.u32(0)
            while j < OL_BOUNDARY_COUNT:
                if medium_boundaries[j].z == boundary_id:
                    boundary_index = j
                    break
                j = j + 1
            if boundary_index == osh.u32(4294967295):
                invalid = True
        input_sample.media.z = boundary_index
        if not invalid:
            dielectric = transport_materials[input_sample.identity.z].albedo_kind.w == 1.0
            if dielectric != (boundary_index != osh.u32(4294967295)):
                invalid = True
    if invalid:
        result.counts = osh.uvec4(pc.samples_per_element, 0, 32, 0)
        accumulated[i] = result
        return
    sample_index = osh.u32(0)
    while sample_index < pc.samples_per_element:
        rng = secondaryNeeHash(owner ^ secondaryNeeHash(pc.seed) ^ secondaryNeeHash(pc.sample_offset + sample_index) ^ secondaryNeeHash(input_sample.identity.y) ^ secondaryNeeHash(i))
        medium_stack = osh.local_array(osh.u32, 8)
        boundary_stack = osh.local_array(osh.u32, 8)
        stack_offset = input_sample.media.w * osh.u32(8)
        depth = initial_stack[stack_offset].z
        j = osh.u32(0)
        while j < depth:
            medium_stack[j] = initial_stack[stack_offset + j].x
            boundary_stack[j] = initial_stack[stack_offset + j].y
            j = j + 1
        origin = input_sample.position.xyz
        direction = input_sample.incoming.xyz
        throughput = osh.vec3(1)
        radiance = osh.vec3(0)
        status = osh.u32(0)
        truncated = False
        previous_pdf = 0.0
        previous_delta = True
        previous_position = origin
        events = osh.uvec4(0)
        bounce = osh.u32(0)
        while bounce <= pc.max_bounces:
            hit = OrdinaryLightHit(osh.vec4(0), osh.vec4(0), osh.vec4(0), osh.uvec4(0), osh.uvec4(0))
            if bounce == osh.u32(0) and input_sample.identity.w & osh.u32(1) != osh.u32(0):
                hit.position_distance = osh.vec4(origin, 0)
                hit.geometric_normal = input_sample.geometric_normal
                hit.shading_normal = input_sample.shading_normal
                hit.identity = osh.uvec4(3, 0, owner, input_sample.identity.z)
                hit.boundary = osh.uvec4(input_sample.media.z, 0, 0, 0)
                if hit.boundary.x != osh.u32(4294967295):
                    hit.boundary.yz = medium_boundaries[hit.boundary.x].xy
            else:
                hit = ordinarylightIntersect(origin, direction, 0.0, pc.max_distance, pc.tolerance, pc.max_steps)
            if hit.boundary.w != osh.u32(0):
                status = status | hit.boundary.w
                break
            if hit.identity.x == osh.u32(0):
                if depth != osh.u32(1):
                    status = status | osh.u32(8)
                else:
                    radiance = radiance + throughput * pc.environment.rgb * (powerHeuristic(previous_pdf, 1.0 / (4.0 * OL_PI)) if pc.environment_nee & osh.u32(1) != osh.u32(0) and (not previous_delta) else 1.0)
                break
            throughput = throughput * ordinarylightBeer(optical_media[medium_stack[depth - osh.u32(1)]].rgb, hit.position_distance.w)
            material = transport_materials[hit.identity.w]
            evaluated = ordinarylightEvaluateMaterial(hit.identity.w, hit, direction, osh.f32(bounce), optical_media[medium_stack[depth - osh.u32(1)]].a, optical_media[medium_stack[depth - osh.u32(2)]].a if depth > osh.u32(1) else 1.0, osh.vec2(randomFloat(rng), randomFloat(rng)))
            if not ordinarylightValidTransportMaterial(evaluated, material, hit):
                status = status | osh.u32(16)
                break
            evaluated.metallic = osh.clamp(evaluated.metallic, 0.0, 1.0)
            evaluated.roughness = osh.clamp(evaluated.roughness, 0.0, 1.0)
            evaluated.base_color = osh.clamp(evaluated.base_color, osh.vec3(0), osh.vec3(1))
            material.albedo_kind.rgb = osh.clamp(evaluated.base_color, osh.vec3(0), osh.vec3(1))
            material.emission.rgb = osh.maximum(evaluated.emission, osh.vec3(0))
            if osh.dot(direction, hit.geometric_normal.xyz) < 0.0 or material.emission.a > 0.5:
                emission_weight = 1.0
                if pc.environment_nee & osh.u32(2) != osh.u32(0) and (not previous_delta):
                    emission_weight = ordinarylightEmissiveMisWeight(previous_pdf, ordinarylightSurfaceLightPdf(previous_position, hit, status))
                radiance = radiance + throughput * material.emission.rgb * emission_weight
            dielectric = material.albedo_kind.w == 1.0
            if material.albedo_kind.w == 3.0:
                break
            if material.albedo_kind.w == 0.0 and osh.maximum(material.albedo_kind.r, osh.maximum(material.albedo_kind.g, material.albedo_kind.b)) == 0.0:
                break
            if bounce == pc.max_bounces:
                truncated = True
                break
            geometric = hit.geometric_normal.xyz
            entering = osh.dot(direction, geometric) < 0.0
            scattering_normal = hit.shading_normal.xyz if entering else -hit.shading_normal.xyz
            if dielectric and pc.environment_nee & osh.u32(4) == osh.u32(0):
                scattering_normal = geometric if entering else -geometric
            if dielectric and osh.dot(scattering_normal, -direction) <= 0.0:
                break
            eta_i = optical_media[medium_stack[depth - osh.u32(1)]].a
            eta_t = eta_i
            target_medium = medium_stack[depth - osh.u32(1)]
            if dielectric and hit.boundary.x != osh.u32(4294967295):
                target_medium = medium_boundaries[hit.boundary.x].y if entering else medium_boundaries[hit.boundary.x].x
                eta_t = optical_media[target_medium].a
            light_index = osh.u32(0)
            while light_index < OL_ANALYTIC_LIGHT_COUNT:
                p = analytic_lights[light_index * osh.u32(4)]
                d = analytic_lights[light_index * osh.u32(4) + osh.u32(1)]
                color = analytic_lights[light_index * osh.u32(4) + osh.u32(2)]
                cone = analytic_lights[light_index * osh.u32(4) + osh.u32(3)]
                directional = p.w == 1.0
                offset = p.xyz - hit.position_distance.xyz
                distance = pc.max_distance if directional else osh.length(offset)
                if distance <= 2.0 * pc.ray_epsilon or ((not directional and d.w > 0.0) and distance > d.w):
                    light_index = light_index + 1
                    continue
                outgoing = -d.xyz if directional else offset / distance
                attenuation = 1.0 if directional else 1.0 / (distance * distance)
                if p.w == 2.0:
                    attenuation = attenuation * (osh.f32(osh.dot(d.xyz, -outgoing) >= cone.x) if cone.x == cone.y else osh.smoothstep(cone.y, cone.x, osh.dot(d.xyz, -outgoing)))
                if osh.dot(outgoing, scattering_normal) * osh.dot(outgoing, geometric if entering else -geometric) <= 0.0:
                    light_index = light_index + 1
                    continue
                bsdf = olBsdf(evaluated, osh.u32(material.albedo_kind.w), scattering_normal, -direction, outgoing, eta_i, eta_t)
                if osh.maximum(bsdf.r, osh.maximum(bsdf.g, bsdf.b)) <= 0.0:
                    light_index = light_index + 1
                    continue
                shadow_medium = medium_stack[depth - osh.u32(1)] if osh.dot(outgoing, scattering_normal) > 0.0 else target_medium
                if directional and shadow_medium != osh.u32(0):
                    light_index = light_index + 1
                    continue
                blocker = ordinarylightIntersect(hit.position_distance.xyz + outgoing * pc.ray_epsilon, outgoing, 0.0, distance - 2.0 * pc.ray_epsilon, pc.tolerance, pc.max_steps)
                status = status | blocker.boundary.w
                if blocker.identity.x == osh.u32(0) and blocker.boundary.w == osh.u32(0):
                    radiance = radiance + throughput * bsdf.rgb * osh.absolute(osh.dot(scattering_normal, outgoing)) * color.rgb * color.a * attenuation * ordinarylightBeer(optical_media[shadow_medium].rgb, 0.0 if directional else distance)
                light_index = light_index + 1
            if pc.environment_nee & osh.u32(1) != osh.u32(0) and osh.maximum(pc.environment.r, osh.maximum(pc.environment.g, pc.environment.b)) > 0.0:
                z = 1.0 - 2.0 * randomFloat(rng)
                phi = 2.0 * OL_PI * randomFloat(rng)
                radius = osh.sqrt(osh.maximum(0.0, 1.0 - z * z))
                outgoing = osh.vec3(radius * osh.cosine(phi), radius * osh.sine(phi), z)
                bsdf = olBsdf(evaluated, osh.u32(material.albedo_kind.w), scattering_normal, -direction, outgoing, eta_i, eta_t)
                shadow_medium = medium_stack[depth - osh.u32(1)] if osh.dot(outgoing, scattering_normal) > 0.0 else target_medium
                if (bsdf.a > 0.0 and shadow_medium == osh.u32(0)) and osh.dot(outgoing, scattering_normal) * osh.dot(outgoing, geometric if entering else -geometric) > 0.0:
                    blocker = ordinarylightIntersect(hit.position_distance.xyz + outgoing * pc.ray_epsilon, outgoing, 0.0, pc.max_distance, pc.tolerance, pc.max_steps)
                    status = status | blocker.boundary.w
                    if blocker.identity.x == osh.u32(0) and blocker.boundary.w == osh.u32(0):
                        radiance = radiance + throughput * bsdf.rgb * osh.absolute(osh.dot(scattering_normal, outgoing)) * pc.environment.rgb * (4.0 * OL_PI) * powerHeuristic(1.0 / (4.0 * OL_PI), bsdf.a)
            if pc.environment_nee & osh.u32(2) != osh.u32(0):
                light_sample = OrdinaryLightEmitterSample(osh.vec3(0), osh.vec3(0), osh.f32(0), osh.u32(0), osh.u32(0))
                sampled = ordinarylightSampleSurface(osh.vec4(randomFloat(rng), randomFloat(rng), randomFloat(rng), randomFloat(rng)), light_sample)
                if sampled > osh.u32(1):
                    status = status | osh.u32(1)
                    break
                if sampled == osh.u32(1):
                    offset = light_sample.position - hit.position_distance.xyz
                    distance = osh.length(offset)
                    if distance > 2.0 * pc.ray_epsilon and distance < pc.max_distance:
                        outgoing = offset / distance
                        cosine = osh.absolute(osh.dot(light_sample.normal, -outgoing))
                        if cosine > 0.0 and osh.dot(outgoing, scattering_normal) * osh.dot(outgoing, geometric if entering else -geometric) > 0.0:
                            bsdf = olBsdf(evaluated, osh.u32(material.albedo_kind.w), scattering_normal, -direction, outgoing, eta_i, eta_t)
                            if bsdf.a > 0.0 and osh.maximum(bsdf.r, osh.maximum(bsdf.g, bsdf.b)) > 0.0:
                                emitter = ordinarylightIntersect(hit.position_distance.xyz + outgoing * pc.ray_epsilon, outgoing, 0.0, distance + pc.ray_epsilon, pc.tolerance, pc.max_steps)
                                status = status | emitter.boundary.w
                                if ((emitter.boundary.w == osh.u32(0) and emitter.identity.x == light_sample.kind) and emitter.identity.y == light_sample.primitive) and osh.length(emitter.position_distance.xyz - light_sample.position) <= 4.0 * pc.ray_epsilon:
                                    if osh.dot(emitter.geometric_normal.xyz, light_sample.normal) < 0.999:
                                        status = status | osh.u32(1)
                                        break
                                    light_pdf = light_sample.area_pdf * distance * distance / (osh.f32(OL_SURFACE_SLOT_COUNT) * cosine)
                                    if not ordinarylightValidAreaPdf(light_pdf) or light_pdf == 0.0:
                                        status = status | osh.u32(1)
                                        break
                                    shadow_medium = medium_stack[depth - osh.u32(1)] if osh.dot(outgoing, scattering_normal) > 0.0 else target_medium
                                    emission = ordinarylightEvaluateMaterial(emitter.identity.w, emitter, outgoing, osh.f32(bounce + osh.u32(1)), optical_media[shadow_medium].a, eta_i, osh.vec2(randomFloat(rng), randomFloat(rng)))
                                    if not ordinarylightValidTransportMaterial(emission, transport_materials[emitter.identity.w], emitter):
                                        status = status | osh.u32(16)
                                        break
                                    if osh.dot(outgoing, emitter.geometric_normal.xyz) < 0.0 or transport_materials[emitter.identity.w].emission.a > 0.5:
                                        radiance = radiance + throughput * bsdf.rgb * osh.absolute(osh.dot(scattering_normal, outgoing)) * osh.maximum(emission.emission, osh.vec3(0)) * ordinarylightBeer(optical_media[shadow_medium].rgb, distance) * ordinarylightEmissiveMisWeight(light_pdf, bsdf.a) / light_pdf
            previous_delta = evaluated.roughness == 0.0 or eta_i == eta_t if dielectric else False
            if dielectric:
                if hit.boundary.x == osh.u32(4294967295):
                    status = status | osh.u32(2)
                    break
                boundary = medium_boundaries[hit.boundary.x]
                target = boundary.y if entering else boundary.x
                if entering:
                    if medium_stack[depth - osh.u32(1)] != boundary.x:
                        status = status | osh.u32(2)
                        break
                    j = osh.u32(1)
                    while j < depth:
                        if boundary_stack[j] == hit.boundary.x:
                            status = status | osh.u32(2)
                        j = j + 1
                    if status != osh.u32(0):
                        break
                else:
                    if depth < osh.u32(2):
                        status = status | osh.u32(2)
                        break
                    if (boundary_stack[depth - osh.u32(1)] != hit.boundary.x or medium_stack[depth - osh.u32(1)] != boundary.y) or medium_stack[depth - osh.u32(2)] != boundary.x:
                        status = status | osh.u32(2)
                        break
                weight = osh.vec3(0)
                pdf = osh.f32(0.0)
                reflected = osh.boolean(False)
                tir = osh.boolean(False)
                direction = olSampleBsdf(evaluated, osh.u32(1), scattering_normal, -direction, optical_media[medium_stack[depth - osh.u32(1)]].a, optical_media[target].a, osh.vec3(randomFloat(rng), randomFloat(rng), osh.minimum(randomFloat(rng), 0.99999994)), weight, pdf, reflected, tir)
                if pdf == 0.0:
                    break
                geometric_side = osh.dot(direction, geometric if entering else -geometric)
                if geometric_side == 0.0 or (geometric_side > 0.0) != reflected:
                    break
                throughput = throughput * weight
                previous_pdf = pdf
                if reflected:
                    events.y = events.y + 1
                    if tir:
                        events.w = events.w + 1
                else:
                    events.z = events.z + 1
                    if entering:
                        if depth == osh.u32(8):
                            status = status | osh.u32(4)
                            break
                        medium_stack[depth] = target
                        boundary_stack[depth] = hit.boundary.x
                        depth = depth + 1
                    else:
                        depth = depth - 1
            else:
                normal = hit.shading_normal.xyz if entering else -hit.shading_normal.xyz
                weight = osh.vec3(0)
                pdf = osh.f32(0.0)
                reflected = osh.boolean(False)
                tir = osh.boolean(False)
                mirror = osh.reflect(direction, normal)
                direction = olSampleBsdf(evaluated, osh.u32(material.albedo_kind.w), normal, -direction, 1.0, 1.0, osh.vec3(randomFloat(rng), randomFloat(rng), randomFloat(rng)), weight, pdf, reflected, tir)
                if pdf == 0.0 or osh.dot(direction, geometric if entering else -geometric) <= 0.0:
                    break
                throughput = throughput * weight
                previous_pdf = pdf
                previous_delta = (material.albedo_kind.w == 2.0 and evaluated.roughness == 0.0) and osh.dot(direction, mirror) > 0.999999
                events.x = events.x + 1
            throughput = throughput * ordinarylightBeer(optical_media[medium_stack[depth - osh.u32(1)]].rgb, pc.ray_epsilon)
            previous_position = hit.position_distance.xyz
            origin = hit.position_distance.xyz + direction * pc.ray_epsilon
            if dielectric and evaluated.roughness > 0.0:
                origin = hit.position_distance.xyz + geometric * (pc.ray_epsilon if osh.dot(direction, geometric) > 0.0 else -pc.ray_epsilon)
            if ((osh.any_value(osh.is_nan(throughput)) or osh.any_value(osh.is_inf(throughput))) or osh.any_value(osh.is_nan(radiance))) or osh.any_value(osh.is_inf(radiance)):
                status = status | osh.u32(16)
                break
            if osh.maximum(throughput.r, osh.maximum(throughput.g, throughput.b)) == 0.0:
                break
            bounce = bounce + 1
        if ((osh.any_value(osh.is_nan(throughput)) or osh.any_value(osh.is_inf(throughput))) or osh.any_value(osh.is_nan(radiance))) or osh.any_value(osh.is_inf(radiance)):
            status = status | osh.u32(16)
        result.counts.x = result.counts.x + 1
        result.counts.z = result.counts.z | status
        if status == osh.u32(0):
            result.radiance.rgb = result.radiance.rgb + radiance
            result.counts.y = result.counts.y + 1
        if truncated:
            result.counts.w = result.counts.w + 1
        result.events = result.events + events
        sample_index = sample_index + 1
    accumulated[i] = result

@osh.structure
class PipelineConstants:
    count: osh.u32
    samples_per_element: osh.u32
    max_bounces: osh.u32
    sample_offset: osh.u32
    seed: osh.u32
    initial_depth: osh.u32
    tolerance: osh.f32
    ray_epsilon: osh.f32
    max_steps: osh.u32
    max_distance: osh.f32
    initial_stack_count: osh.u32
    environment_nee: osh.u32
    environment: osh.vec4
