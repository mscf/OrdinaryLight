"""Qt-owned native windows exposed as Vulkan presentation surfaces."""

from __future__ import annotations

from ..targets._vulkan_version import vulkan_api_version


class QtVulkanSurface:
    """Own a Vulkan instance/surface pair for a Qt ``QWindow``.

    PySide currently exposes the XCB application connection but not Wayland's
    per-window ``wl_surface``.  Consequently this provider requires Qt's
    ``xcb`` platform (native X11 or XWayland) while keeping the window itself
    fully Qt-owned.
    """

    def __init__(self, window):
        try:
            import vulkan as vk
            from PySide6 import QtGui
        except ImportError as error:
            raise RuntimeError(
                "Qt Vulkan presentation requires ordinarylight[qt,vulkan]"
            ) from error
        app = QtGui.QGuiApplication.instance()
        if app is None:
            raise RuntimeError("create QGuiApplication before QtVulkanSurface")
        if app.platformName() != "xcb":
            raise RuntimeError(
                "PySide does not expose a Wayland QWindow wl_surface; launch "
                "with QT_QPA_PLATFORM=xcb to use Qt Vulkan presentation"
            )
        window.setSurfaceType(QtGui.QWindow.SurfaceType.VulkanSurface)
        window.create()
        native = app.nativeInterface()
        connection = native.connection()
        if not connection:
            raise RuntimeError("Qt did not provide its XCB connection")
        app_info = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName="Ordinary Light Qt",
            applicationVersion=1,
            pEngineName="Ordinary Light",
            engineVersion=1,
            apiVersion=vulkan_api_version(vk),
        )
        extensions = ["VK_KHR_surface", "VK_KHR_xcb_surface"]
        self.instance = vk.vkCreateInstance(vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=app_info,
            enabledExtensionCount=len(extensions),
            ppEnabledExtensionNames=extensions,
        ), None)
        self._vk = vk
        self._destroy_surface = vk.vkGetInstanceProcAddr(
            self.instance, "vkDestroySurfaceKHR",
        )
        self._create_surface = vk.vkGetInstanceProcAddr(
            self.instance, "vkCreateXcbSurfaceKHR",
        )
        self._connection = vk.ffi.cast(
            "xcb_connection_t *", int(connection),
        )
        self.window = window
        self.surface = self._new_surface()
        self._closed = False

    def _new_surface(self):
        return self._create_surface(
            self.instance, self._vk.VkXcbSurfaceCreateInfoKHR(
                sType=self._vk.VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR,
                connection=self._connection,
                window=int(self.window.winId()),
            ), None,
        )

    def recreate_surface(self):
        """Replace the native surface between independent renderer owners.

        A renderer must be closed before this method is called.  The Qt
        ``QWindow`` and Vulkan instance stay alive, while a fresh surface
        prevents sequential logical devices from inheriting presentation
        state associated with the previous swapchain.
        """
        if self._closed:
            raise RuntimeError("cannot recreate a closed Qt Vulkan surface")
        if self.surface is not None:
            self._destroy_surface(self.instance, self.surface, None)
        self.surface = self._new_surface()
        return self.surface

    def close(self):
        """Destroy the surface after its renderer has been closed."""
        if self._closed:
            return
        self._closed = True
        self._destroy_surface(self.instance, self.surface, None)
        self._vk.vkDestroyInstance(self.instance, None)
        self.surface = None
        self.instance = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


__all__ = ["QtVulkanSurface"]
