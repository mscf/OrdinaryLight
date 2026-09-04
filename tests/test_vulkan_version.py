from ordinarylight.targets._vulkan_version import (
    VULKAN_API_VERSION, vulkan_api_version,
)


def test_vulkan_api_baseline_is_1_2():
    class Binding:
        @staticmethod
        def VK_MAKE_VERSION(major, minor, patch):
            return major, minor, patch

    assert VULKAN_API_VERSION == (1, 2, 0)
    assert vulkan_api_version(Binding) == (1, 2, 0)
