"""Native frame bindings for independently reusable ReLAX graph components."""

import vulkan as vk

from ...pipeline.graph import VulkanGraph
from ...pipeline.vulkan import VulkanResource
from ...runtime.relax import VulkanRelaxSpatial
from ...runtime.relax_temporal import VulkanRelaxHistory, VulkanRelaxTemporal


class _Binding:
    """Identity-hashable borrowed allocation view, owned by this adapter's lease."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


class _QueueCompletion:
    """External submission lease; the presenter owns command buffers and fences."""

    def __init__(self, owner):
        self.owner = owner
        self.runtime = owner.runtime

    def wait(self):
        if not self.owner.idle:
            vk.vkQueueWaitIdle(self.owner.core.queue)
        return self


class NativeDenoiserGraph:
    def __init__(self, core):
        self.core, self.runtime = core, core.runtime
        self.executor = core.wavefront_executor
        self.closed = False
        self.idle = False
        self.histories, self.temporal, self.spatial = [], [], []
        self.recordings = [None, None]
        self.key = (
            self.executor,
            core.config.denoiser_iterations,
            core.config.denoiser_color_weight,
        )
        with self.runtime.lock:
            before = set(self.runtime._consumers)
            try:
                self._initialize()
            except Exception:
                self.close()
                raise
            self.consumers = set(self.runtime._consumers) - before

    def require_open(self):
        self.runtime.require_open()
        if self.closed:
            raise RuntimeError("Native denoiser bindings have been retired")

    def _image(self, slot, name, format):
        frame = self.core.window_frames[slot]
        width, height = frame["wavefront_allocation_extent"]
        # These views borrow the core's allocation lease. No allocation or copy.
        return _Binding(
            runtime=self.runtime,
            image=frame[name + "_image"],
            view=frame[name + "_view"],
            width=width,
            height=height,
            format=format,
            layout=vk.VK_IMAGE_LAYOUT_GENERAL,
            usage=vk.VK_IMAGE_USAGE_STORAGE_BIT,
            require_open=self.require_open,
        )

    def _initialize(self):
        rgba, scalar, uint = (
            vk.VK_FORMAT_R16G16B16A16_SFLOAT,
            vk.VK_FORMAT_R32_SFLOAT,
            vk.VK_FORMAT_R32_UINT,
        )
        self.views = []
        for slot in range(2):
            names = {
                "diffuse": rgba,
                "specular": rgba,
                "normal_roughness": rgba,
                "view_z": scalar,
                "motion": rgba,
                "identity": uint,
                "temporal_diffuse": rgba,
                "temporal_specular": rgba,
                "diffuse_history": scalar,
                "specular_history": scalar,
                "atrous_diffuse": rgba,
                "atrous_specular": rgba,
            }
            views = {
                name: self._image(slot, "wavefront_relax_" + name, fmt)
                for name, fmt in names.items()
            }
            views["material"] = self._image(slot, "wavefront_material", uint)
            views["hdr"] = self._image(slot, "wavefront_hdr", rgba)
            self.views.append(views)
            self.histories.append(
                VulkanRelaxHistory(
                    self.runtime,
                    normal_roughness=views["normal_roughness"],
                    view_z=views["view_z"],
                    material=views["material"],
                    identity=views["identity"],
                    images=tuple(
                        views[name]
                        for name in (
                            "temporal_diffuse",
                            "temporal_specular",
                            "diffuse_history",
                            "specular_history",
                        )
                    ),
                )
            )
        for slot, views in enumerate(self.views):
            buffer = self.executor.relax_temporal_constant_buffers[slot]
            policy = _Binding(
                runtime=self.runtime,
                buffer=buffer.buffer,
                byte_size=32,
                usage=vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                require_open=self.require_open,
            )
            self.temporal.append(
                VulkanRelaxTemporal(
                    self.runtime,
                    diffuse=views["diffuse"],
                    specular=views["specular"],
                    motion=views["motion"],
                    previous=self.histories[1 - slot],
                    output=self.histories[slot],
                    policy_buffer=policy,
                )
            )
            self.spatial.append(
                VulkanRelaxSpatial(
                    self.runtime,
                    diffuse=views["temporal_diffuse"],
                    specular=views["temporal_specular"],
                    normal_roughness=views["normal_roughness"],
                    view_z=views["view_z"],
                    material=views["material"],
                    output=views["hdr"],
                    iterations=self.core.config.denoiser_iterations,
                    color_weight=self.core.config.denoiser_color_weight,
                    scratch=tuple(
                        views[name]
                        for name in (
                            "atrous_diffuse",
                            "diffuse",
                            "atrous_specular",
                            "specular",
                        )
                    ),
                )
            )

    def record(self, command, slot, extent):
        with self.runtime.lock:
            self.require_open()
            graph = VulkanGraph().add(
                "temporal",
                self.temporal[slot].operation(extent=extent),
                reads=tuple(
                    VulkanResource.image(self.views[slot][name]).version(0)
                    for name in ("diffuse", "specular")
                ),
            )
            # Spatial ping-pong reuses raw signal storage; explicitly order the
            # temporal reader before the spatial writer of that same allocation.
            graph.add(
                "spatial",
                self.spatial[slot].operation(extent=extent),
                after=("temporal",),
            )
            recording = graph.compile().prepare_recording(self.runtime)
            if recording.wait_semaphores or recording.signal_semaphores:
                raise RuntimeError(
                    "Denoiser graph unexpectedly requires external semaphores"
                )
            recording.record(command)
            self.recordings[slot] = recording

    def submitted(self, slot):
        recording = self.recordings[slot]
        if recording is not None:
            recording.submitted(_QueueCompletion(self))

    def close(self):
        if self.closed:
            return
        with self.runtime.lock:
            try:
                vk.vkDeviceWaitIdle(self.runtime.device)
            except vk.VkErrorDeviceLost:
                pass
            self.idle = True
            for stage in reversed(self.spatial):
                stage.close()
            for stage in reversed(self.temporal):
                stage.close()
            for history in reversed(self.histories):
                history.close()
            self.recordings = [None, None]
            self.closed = True
