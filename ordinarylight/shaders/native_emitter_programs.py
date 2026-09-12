"""Typed application-owned area-emitter distribution for native camera GI."""
import ordinaryshade as osh
from .native_intersection_programs import NativeIntersection


@osh.structure
class NativeEmitterSample:
    position: osh.vec3
    normal: osh.vec3
    emission: osh.vec3
    area_pdf: osh.f32
    two_sided: osh.boolean


@osh.external
def nativeEmitterCount() -> osh.u32:
    pass


@osh.external
def nativeSelectEmitter(selector: osh.f32) -> osh.u32:
    pass


@osh.external
def nativeEvaluateEmitter(emitter: osh.u32, coordinates: osh.vec2) -> NativeEmitterSample:
    pass


@osh.external
def nativeEmitterPdf(hit: NativeIntersection) -> osh.f32:
    pass


@osh.function
def nativeAreaLightCount(fallback: osh.u32) -> osh.u32:
    if osh.specialization('WAVE_NATIVE_EMITTERS'):
        return osh.minimum(nativeEmitterCount(), osh.u32(33554430))
    return fallback


@osh.function
def nativeEmitterValid(sample: NativeEmitterSample) -> osh.boolean:
    return (sample.area_pdf > 0.0 and not osh.is_nan(sample.area_pdf)
            and not osh.is_inf(sample.area_pdf)
            and not osh.any_value(osh.is_nan(sample.position))
            and not osh.any_value(osh.is_inf(sample.position))
            and not osh.any_value(osh.is_nan(sample.normal))
            and not osh.any_value(osh.is_inf(sample.normal))
            and osh.absolute(osh.dot(sample.normal, sample.normal) - 1.0) < 0.001
            and not osh.any_value(osh.is_nan(sample.emission))
            and not osh.any_value(osh.is_inf(sample.emission))
            and osh.all_value(sample.emission >= osh.vec3(0.0)))
