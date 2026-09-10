"""Typed interfaces to scene-specific dispatch modules."""
import ordinaryshade as osh
from .transport_programs import OrdinaryLightCustomHit, OrdinaryLightHit, MaterialEvaluation


@osh.external
def ordinarylightCustomIntersect(program: osh.u32, origin: osh.vec3, direction: osh.vec3,
                                 t_min: osh.f32, t_max: osh.f32, parameters: osh.vec4,
                                 tolerance: osh.f32, max_steps: osh.u32,
                                 result: osh.inout(OrdinaryLightCustomHit)) -> osh.u32:
    pass


@osh.external
def ordinarylightEvaluateMaterial(index: osh.u32, hit: OrdinaryLightHit, direction: osh.vec3,
                                  bounce: osh.f32, current_ior: osh.f32, exterior_ior: osh.f32,
                                  randoms: osh.vec2) -> MaterialEvaluation:
    pass


@osh.external
def boundaryIndex(identity: osh.u32) -> osh.u32:
    pass


@osh.external
def dielectricMaterial(index: osh.u32) -> osh.boolean:
    pass
