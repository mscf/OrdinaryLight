"""Exercise the Qt window's retirement methods with a deliberately slow renderer."""

import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, get_ident
from types import SimpleNamespace

from ordinarylight.runtime.lifecycle import timed_call


def window_methods():
    # Load only lifecycle methods; no Qt application or GPU is needed to exercise
    # the exact methods used by the locally defined DirectWindow class.
    source = (
        Path(__file__).parents[1] / "ordinarylight/integrations/raster_workbench.py"
    )
    tree = ast.parse(source.read_text())
    window = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "DirectWindow"
    )
    methods = [
        n
        for n in window.body
        if isinstance(n, ast.FunctionDef)
        and n.name
        in {"restart", "_retire_renderer", "_finish_pending_restart", "closeEvent"}
    ]
    cls = ast.ClassDef(
        name="Lifecycle", bases=[], keywords=[], body=methods, decorator_list=[]
    )
    namespace = {"timed_call": timed_call,
                 "GI_TARGETS": frozenset(("wavefront-gi", "wavefront-gi-fast"))}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
            str(source),
            "exec",
        ),
        namespace,
    )
    return namespace["Lifecycle"]


def test_restart_and_close_do_not_wait_for_renderer_teardown():
    entered, release = Event(), Event()
    threads = []

    def close():
        threads.append(get_ident())
        entered.set()
        assert release.wait(5)

    window = window_methods()()
    window.renderer = SimpleNamespace(close=close)
    window.future = window.renderer_start_future = window.renderer_update_future = None
    window.renderer_close_future = None
    window.restart_pending = window.close_pending = False
    window.pending_renderer_updates = []
    window._extension_call = lambda *args: None
    window.status = SimpleNamespace(setText=lambda text: None)
    hidden = []
    window.hide = lambda: hidden.append("window")
    window.fps_overlay = SimpleNamespace(hide=lambda: hidden.append("overlay"))
    with ThreadPoolExecutor(max_workers=1) as executor:
        window.executor = executor
        try:
            window.restart()
            assert entered.wait(1)
            assert window.renderer is None
            assert window.restart_pending
            assert not window.renderer_close_future.done()
            assert len(threads) == 1 and threads[0] != get_ident()
            future = window.renderer_close_future
            # Repeated requests wait for the same retirement; no surface reuse.
            window.restart()
            window._finish_pending_restart()
            assert window.renderer_close_future is future
            ignored = []
            window.closeEvent(SimpleNamespace(ignore=lambda: ignored.append(True)))
            assert ignored == [True] and window.close_pending
            assert hidden == ["window", "overlay"]
            assert not future.done()
        finally:
            release.set()
            if window.renderer_close_future is not None:
                window.renderer_close_future.result(timeout=2)


def test_repeated_switch_requests_coalesce_while_work_is_pending():
    """Scene/backend selections cannot retire a renderer during queued GPU work."""
    from concurrent.futures import Future

    window = window_methods()()
    window.renderer = SimpleNamespace(close=lambda: None)
    window.renderer_close_future = None
    window.restart_pending = window.close_pending = False
    window.pending_renderer_updates = []
    cancellations = []
    window._extension_call = lambda *args: cancellations.append(args)
    window.status = SimpleNamespace(setText=lambda text: None)
    original = window.renderer
    for pending in ("future", "renderer_start_future", "renderer_update_future"):
        window.future = window.renderer_start_future = window.renderer_update_future = (
            None
        )
        work = Future()
        setattr(window, pending, work)
        for _ in range(20):
            window.restart()
            window._finish_pending_restart()
            assert window.renderer is original
            assert window.renderer_close_future is None
            assert window.restart_pending
        work.set_result(None)
        # tick owns harvesting completed futures; completion alone must not
        # allow restart to race that bookkeeping.
        window._finish_pending_restart()
        assert window.renderer is original
    assert len(cancellations) == 60


def test_restart_resets_instance_before_constructing_another_vulkan_owner():
    for previous in ("vulkan-raster", "wavefront-gi-fast", "webgpu-raster"):
        window = window_methods()()
        window.future = window.renderer_start_future = None
        window.renderer_update_future = window.renderer_close_future = None
        window.renderer = None
        window.renderer_target = previous
        window.target = SimpleNamespace(currentData=lambda: "wavefront-gi-fast",
                                        currentText=lambda: "GI")
        window.denoiser = SimpleNamespace(isChecked=lambda: False)
        window.status = SimpleNamespace(setText=lambda text: None)
        events = []
        window._close_renderer = lambda: events.append("drain") or []
        def recreate():
            events.append("fresh-instance")
            # Stop before starting a real Qt/GPU renderer in this lifecycle test.
            raise RuntimeError("test boundary")
        window.surface = SimpleNamespace(recreate_instance=recreate)
        window.restart()
        assert events == ["drain", "fresh-instance"]
