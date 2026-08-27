"""Print Vulkan renderer capability information."""

from .backends.vulkan import probe_vulkan_devices


def main():
    devices = probe_vulkan_devices()
    if not devices:
        print("No Vulkan devices found")
        return
    for device in devices:
        version = ".".join(map(str, device.api_version))
        print(f"{device.name} (Vulkan {version})")
        print(f"  Hardware adapter: {'yes' if device.is_hardware_adapter else 'no'}")
        print(f"  Ray-query extensions: {'yes' if device.supports_ray_query else 'no'}")
        print(f"  Hardware RT candidate: {'yes' if device.supports_hardware_ray_tracing else 'no'}")
        if device.missing_ray_tracing_extensions:
            print(f"  Missing: {', '.join(sorted(device.missing_ray_tracing_extensions))}")


if __name__ == "__main__":
    main()
