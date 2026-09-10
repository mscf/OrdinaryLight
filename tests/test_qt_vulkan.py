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


def test_renderer_handoff_recreates_instance_after_releasing_old_surface(monkeypatch):
    from types import SimpleNamespace
    provider = QtVulkanSurface.__new__(QtVulkanSurface)
    provider._closed = False
    window = object()
    provider.window = window
    provider.instance, provider.surface = 'old-instance', 'old-surface'
    events = []
    provider._destroy_surface = lambda *args: events.append(('surface', *args))
    provider._vk = SimpleNamespace(
        vkDestroyInstance=lambda *args: events.append(('instance', *args)),
    )

    def initialize(self, same_window):
        assert self._closed
        assert self.instance is None and self.surface is None
        assert same_window is window
        events.append(('initialize', same_window))
        self.instance, self.surface = 'new-instance', 'new-surface'
        self._closed = False

    monkeypatch.setattr(QtVulkanSurface, '__init__', initialize)
    assert provider.recreate_instance() == 'new-surface'
    assert provider.instance == 'new-instance'
    assert provider.window is window
    assert events == [
        ('surface', 'old-instance', 'old-surface', None),
        ('instance', 'old-instance', None), ('initialize', window),
    ]


def test_closed_instance_cannot_be_recreated():
    import pytest
    provider = QtVulkanSurface.__new__(QtVulkanSurface)
    provider._closed = True
    with pytest.raises(RuntimeError, match='closed'):
        provider.recreate_instance()
