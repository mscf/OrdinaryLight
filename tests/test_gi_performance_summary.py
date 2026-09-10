from ordinarylight.integrations.raster_workbench import _gi_performance_text


def test_gpu_stages_group_bounces_without_adding_host_waits():
    text = _gi_performance_text({
        'wavefront_render_extent': (3840, 2160), 'gpu_frame_ms': 20,
        'wavefront_stage_ms': {'shade.0': 3, 'shade.1': 4, 'primary': 5},
        'fence_wait_ms': 18, 'wavefront_scene_ms': 2,
    })
    assert '3840 × 2160 · GPU    20.00 ms' in text
    assert f"{'shade':<16} {7:8.2f} ms" in text
    assert text.index('shade') < text.index('primary')
    assert 'wait    18.00' in text


def test_missing_gpu_samples_are_safe_during_startup():
    assert 'GPU     0.00 ms' in _gi_performance_text({})


def test_scaled_output_dimensions_are_explicit():
    text = _gi_performance_text({
        'wavefront_render_extent': (1920, 1080),
        'wavefront_output_extent': (3840, 2160),
    })
    assert '1920 × 1080 → 3840 × 2160' in text
