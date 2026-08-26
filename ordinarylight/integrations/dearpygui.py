"""Dear PyGui render surface integration."""

import itertools

import numpy as np

from ..surface import RenderSurface


_surface_ids = itertools.count()


class DearPyGuiSurface(RenderSurface):
    """A ordinarylight surface displayed by a Dear PyGui image widget.

    A Dear PyGui context must exist before constructing this object.
    """

    def __init__(self, width, height, parent, *, label=None):
        super().__init__(width, height)
        try:
            import dearpygui.dearpygui as dpg
        except ImportError as error:
            raise RuntimeError(
                "Install the GUI integration with: pip install -e '.[gui]'"
            ) from error

        self._dpg = dpg
        self._parent = parent
        self._label = label
        surface_id = next(_surface_ids)
        self.texture_tag = f"ordinarylight_texture_{surface_id}"
        self.registry_tag = f"ordinarylight_texture_registry_{surface_id}"
        self.image_tag = f"ordinarylight_image_{surface_id}"

        self._create_items()

    def _create_items(self):
        dpg = self._dpg
        with dpg.texture_registry(show=False, tag=self.registry_tag):
            dpg.add_raw_texture(
                self.width,
                self.height,
                np.zeros(self.width * self.height * 4, dtype=np.float32),
                format=dpg.mvFormat_Float_rgba,
                tag=self.texture_tag,
            )
        dpg.add_image(
            self.texture_tag,
            parent=self._parent,
            width=self.width,
            height=self.height,
            tag=self.image_tag,
            label=self._label,
        )

    def resize(self, width, height):
        """Recreate the texture at a new render resolution."""
        width = max(1, int(width))
        height = max(1, int(height))
        if (width, height) == (self.width, self.height):
            return False
        self.destroy()
        self.width = width
        self.height = height
        self._create_items()
        return True

    def present(self, rgba):
        pixels = self.validate(rgba)
        # Dear PyGui raw RGBA textures use normalized float components.
        texture = np.ascontiguousarray(pixels, dtype=np.float32).ravel() / 255.0
        self._dpg.set_value(self.texture_tag, texture)
        return pixels

    def fit(self, available_width, available_height, preserve_aspect=True):
        """Resize the image widget; the underlying render resolution stays fixed."""
        available_width = max(1, int(available_width))
        available_height = max(1, int(available_height))
        if preserve_aspect:
            scale = min(available_width / self.width, available_height / self.height)
            display_width = max(1, int(self.width * scale))
            display_height = max(1, int(self.height * scale))
        else:
            display_width = available_width
            display_height = available_height
        self._dpg.configure_item(
            self.image_tag,
            width=display_width,
            height=display_height,
        )

    def destroy(self):
        """Remove the image and texture registry if they still exist."""
        if self._dpg.does_item_exist(self.image_tag):
            self._dpg.delete_item(self.image_tag)
        if self._dpg.does_item_exist(self.registry_tag):
            self._dpg.delete_item(self.registry_tag)
