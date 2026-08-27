"""Define a backend-neutral material program in Python."""

import ordinarylight as ol


@ol.material
def tinted_mirror(ctx):
    return ol.SurfaceResponse(
        emission=ctx.emission,
        weight=ctx.base_color * ol.vec3(0.8, 0.95, 1.0),
        next_direction=ol.reflect(ctx.direction, ctx.normal),
        event=ol.SCATTER_REFLECTION,
        pdf=1.0,
    )


def main():
    config = ol.backends.vulkan.RendererConfig(
        material_program=tinted_mirror,
        max_bounces=6,
    )
    print(config.material_program.name)


if __name__ == "__main__":
    main()
