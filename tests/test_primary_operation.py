from types import SimpleNamespace as NS

import pytest
import vulkan as vk
from ordinarylight.pipeline.vulkan import VulkanResource
from ordinarylight.runtime.primary import primary_operation
from ordinarylight.wavefront.primary_bindings import primary_bindings


def kernel():
    runtime = object()
    owner = NS(runtime=runtime, require_open=lambda: None)
    bindings, arrays = {}, {}
    for spec in primary_bindings():
        if spec.count > 1:
            arrays[spec.binding] = [
                (
                    VulkanResource(owner, "image", 100),
                    VulkanResource(owner, "sampler", 101),
                )
            ] * spec.count
        else:
            bindings[spec.binding] = VulkanResource(
                owner, spec.kind, spec.binding, 256 if spec.kind == "buffer" else 0
            )
    return NS(
        runtime=runtime,
        require_open=lambda: None,
        bindings=bindings,
        image_arrays={},
        sampled_image_arrays=arrays,
        sampled_image_layouts={},
    )


def test_primary_dispatch_and_aliased_resource_access(monkeypatch):
    k = kernel()
    # A previous-camera alias must retain both resource ranges in the barrier.
    k.bindings[18] = k.bindings[7].byte_range(64, 64)
    # Current/previous reservoirs can share a placeholder; retain read/write access.
    k.bindings[17] = k.bindings[16]
    calls = []
    k.bind = lambda cmd, data: calls.append((cmd, data))
    monkeypatch.setattr(vk, "vkCmdDispatch", lambda *args: calls.append(args))
    operation = primary_operation(k, bytes(176), workgroups=(3, 2, 1))
    stage = operation.passes[0]
    assert (
        stage.workgroups is None
    )  # Recorder dispatches; scheduler must not repeat it.
    assert len([u for u in stage.uses if u.resource.handle == 100]) == 1
    use = next(u for u in stage.uses if u.resource == k.bindings[16])
    assert use.access == vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT
    assert next(u for u in stage.uses if u.resource.handle == 7).resource.size == 256
    stage.record("command")
    assert calls == [("command", bytes(176)), ("command", 3, 2, 1)]


@pytest.mark.parametrize(
    "change", ["missing", "array", "runtime", "kind", "constants", "groups"]
)
def test_primary_rejects_invalid_contract(change):
    k = kernel()
    constants, groups = bytes(176), (1, 1, 1)
    if change == "missing":
        del k.bindings[23]
    elif change == "array":
        k.sampled_image_arrays[29] = []
    elif change == "runtime":
        k.runtime = object()
    elif change == "kind":
        k.bindings[0] = k.bindings[1]
    elif change == "constants":
        constants = bytes(175)
    else:
        groups = (0, 1, 1)
    with pytest.raises(ValueError):
        primary_operation(k, constants, workgroups=groups)
