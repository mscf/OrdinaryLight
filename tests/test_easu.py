from types import SimpleNamespace
import ordinaryshade as osh
from ordinarylight.shaders.easu import EASU_HELPERS, easu_resolve
from ordinarylight.integrations.raster_workbench import _gi_config


@osh.compute(workgroup_size=(1, 1, 1))
def easu_probe(
    values: osh.storage_buffer(osh.vec4, binding=0, access="read"),
    output: osh.storage_buffer(osh.vec4, binding=1),
):
    output[0] = osh.vec4(
        easu_resolve(
            values[0].xy,
            values[1].rgb,
            values[2].rgb,
            values[3].rgb,
            values[4].rgb,
            values[5].rgb,
            values[6].rgb,
            values[7].rgb,
            values[8].rgb,
            values[9].rgb,
            values[10].rgb,
            values[11].rgb,
            values[12].rgb,
        ),
        1.0,
    )


def test_easu_kernel_compiles_for_both_shader_languages():
    for target in ("glsl", "wgsl"):
        shader = osh.compile(easu_probe, helpers=EASU_HELPERS, target=target)
        assert shader.source
        assert "#include" not in shader.source
        assert "FsrEasuF" not in shader.source


def test_easu_port_is_selectable_without_changing_defaults():
    scene = SimpleNamespace(id="glass-detail-camera", renderer={})
    assert _gi_config(scene).wavefront_upscale_filter == "bilinear"
    assert (
        _gi_config(scene, upscale_filter="fsr1-shade").wavefront_upscale_filter
        == "fsr1-shade"
    )
