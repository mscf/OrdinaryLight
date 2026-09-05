import numpy as np
import pytest

from ordinarylight.geometry import (
    SdfSphere,
    UniformTransform,
    FieldKind,
    FieldComposition,
    intersect_field,
    intersect_triangle,
)
from ordinarylight.transport import (
    OpticalMedium,
    MediumBoundary,
    MediumStack,
    dielectric_event,
    ray_samples,
    surface_samples,
)


def test_signed_distance_gradient_bounds_and_inside_traversal():
    field = SdfSphere((1, 2, 3), 2)
    np.testing.assert_allclose(
        field.evaluate([[1, 2, 3], [1, 2, 5], [1, 2, 6]]), [-2, 0, 1]
    )
    np.testing.assert_allclose(field.gradient([1, 4, 3]), [0, 1, 0])
    np.testing.assert_allclose(field.bounds, [[-1, 0, 1], [3, 4, 5]])
    distance, normal = field.intersect([1, 2, 3], [0, 0, 1])
    assert distance == pytest.approx(2)
    np.testing.assert_allclose(normal, [0, 0, 1])
    assert field.intersect([4, 2, 8], [0, 0, -1]) is None
    # Near the previous boundary but moving deeper: do not hit the origin again.
    distance, _ = SdfSphere().intersect([0, 0, 1 - 1e-6], [0, 0, -1])
    assert distance == pytest.approx(2 - 1e-6, abs=1e-5)


def test_uniform_transform_preserves_distance_and_rejects_nonuniform_scale():
    matrix = np.array([[0, -2, 0, 4], [2, 0, 0, 5], [0, 0, 2, 6], [0, 0, 0, 1]], float)
    field = SdfSphere((1, 0, 0), 0.5).transformed(UniformTransform(matrix))
    np.testing.assert_allclose(field.center, [4, 7, 6])
    assert field.radius == 1
    assert field.evaluate([4, 7, 8]) == 1
    with pytest.raises(ValueError, match="uniform"):
        UniformTransform(np.diag([1, 2, 1, 1]))


def test_conservative_composition_and_scalar_fields_have_distinct_contracts():
    union = FieldComposition(SdfSphere((-0.5, 0, 0)), SdfSphere((0.5, 0, 0)))
    assert union.kind is FieldKind.CONSERVATIVE_DISTANCE
    distance, _ = intersect_field(union, [-3, 0, 0], [1, 0, 0])
    assert distance == pytest.approx(1.5)
    empty = FieldComposition(
        SdfSphere((-3, 0, 0)), SdfSphere((3, 0, 0)), "intersection"
    )
    assert intersect_field(empty, [-6, 0, 0], [1, 0, 0]) is None

    class Scalar:
        kind = FieldKind.SCALAR
        bounds = ((-1, -1, -1), (1, 1, 1))

        def evaluate(self, p):
            return np.sum(np.asarray(p) ** 2, axis=-1) - 1

        def gradient(self, p):
            return 2 * np.asarray(p)

    with pytest.raises(ValueError, match="conservative distance bound"):
        intersect_field(Scalar(), [0, 0, 3], [0, 0, -1])
    with pytest.raises(RuntimeError, match="step budget"):
        SdfSphere().intersect([0.99, 0, 3], [0, 0, -1], max_steps=1)


def test_triangle_geometric_normal_is_oriented_independently_of_ray_side():
    vertices = [[-1, -1, 0], [1, -1, 0], [0, 1, 0]]
    front = intersect_triangle([0, 0, 2], [0, 0, -1], vertices, boundary=31)
    back = intersect_triangle([0, 0, -2], [0, 0, 1], vertices, boundary=31)
    assert front.distance == back.distance == 2
    np.testing.assert_array_equal(front.geometric_normal, back.geometric_normal)
    assert front.boundary == 31


def test_fresnel_snell_eta_squared_and_total_internal_reflection():
    reflected = dielectric_event([0, 0, -1], [0, 0, 1], 1, 1.5, 0.01)
    transmitted = dielectric_event([0.5, 0, -np.sqrt(0.75)], [0, 0, 1], 1, 1.5, 0.8)
    assert reflected.fresnel == pytest.approx(0.04)
    assert reflected.reflected and reflected.throughput == 1
    np.testing.assert_allclose(reflected.direction, [0, 0, 1])
    assert transmitted.direction[0] == pytest.approx(1 / 3)
    assert transmitted.throughput == pytest.approx(4 / 9)
    tir = dielectric_event([np.sqrt(0.75), 0, -0.5], [0, 0, 1], 1.5, 1, 0.99)
    assert tir.total_internal_reflection and tir.fresnel == 1 and tir.throughput == 1
    same = dielectric_event([0, 0, -1], [0, 0, 1], 1, 1, 0)
    assert not same.reflected and same.fresnel == 0


def test_nested_medium_stack_rejects_overlaps_and_preserves_reflection_state():
    stack = MediumStack(capacity=3)
    outer = MediumBoundary(10, 0, 1)
    inner = MediumBoundary(20, 1, 2)
    stack.transmit(outer, True)
    assert stack.target(inner, True) == 2 and stack.media == [0, 1]
    stack.transmit(inner, True)
    with pytest.raises(ValueError, match="non-nested"):
        stack.transmit(outer, False)
    with pytest.raises(ValueError, match="capacity"):
        stack.transmit(MediumBoundary(30, 2, 3), True)
    stack.transmit(inner, False)
    stack.transmit(outer, False)
    assert stack.media == [0]
    with pytest.raises(ValueError, match="overlapping"):
        stack.transmit(inner, True)
    np.testing.assert_allclose(
        OpticalMedium(1.5, (0.2, 0.4, 0.6)).transmittance(2),
        np.exp(-np.array([0.4, 0.8, 1.2])),
    )


def test_sample_identity_and_normal_validation():
    with pytest.raises(ValueError, match="unique"):
        ray_samples([[0, 0, 0], [1, 0, 0]], [[0, 0, 1], [0, 0, 1]], identities=[7, 7])
    samples = surface_samples(
        [[0, 0, 0]],
        [[0, 0, 1]],
        materials=2,
        identities=[7],
        shading_normals=[[0.6, 0, 0.8]],
        boundaries=[10],
    )
    assert samples["identity"][0].tolist() == [7, 0, 2, 1]
    assert samples["media"][0, 2] == 10
    np.testing.assert_allclose(samples["geometric_normal"][0, :3], [0, 0, 1])
