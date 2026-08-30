from ordinarylight.integrations.qt_vulkan import QtVulkanSurface


def test_surface_can_be_recreated_between_renderer_owners():
    provider = QtVulkanSurface.__new__(QtVulkanSurface)
    provider._closed = False
    provider.instance = "instance"
    provider.surface = "old-surface"
    destroyed = []
    provider._destroy_surface = (
        lambda instance, surface, allocator:
        destroyed.append((instance, surface, allocator))
    )
    provider._new_surface = lambda: "new-surface"

    assert provider.recreate_surface() == "new-surface"
    assert provider.surface == "new-surface"
    assert destroyed == [("instance", "old-surface", None)]


def test_closed_surface_cannot_be_recreated():
    provider = QtVulkanSurface.__new__(QtVulkanSurface)
    provider._closed = True

    try:
        provider.recreate_surface()
    except RuntimeError as error:
        assert "closed" in str(error)
    else:
        raise AssertionError("closed surface recreation should fail")
