"""Prepare a scientific package without a native device and serve a browser host."""

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=Path(__file__).with_name("dmc.json")
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--example", choices=("dmc", "hh"), default="dmc")
    args = parser.parse_args()
    from ordinaryscience.rt_density import load_rt_program
    from ordinaryscience.vector_volume import (
        VectorHistogramPreparation,
        reprepare_volume,
    )
    import ordinarylight.portable

    if args.example == "hh":
        from ordinaryscience.hh_neuron import prepare_hh_volume

        package = prepare_hh_volume().export()
        scenario = dict(
            output="voltage",
            live_changes={"parameters": {"g_k": 42.0}},
            structural_changes={
                "parameters": {"dt_ms": 0.005},
                "preparation": {"bins": 44},
            },
            expected_samples=4000,
            expected_bins=44,
            float_atol=0.02,
            float_rtol=0.0,
        )
    else:
        package = VectorHistogramPreparation(
            load_rt_program(args.model),
            x="drift.v",
            y="response.boundary",
            x_range=(0.1, 2),
            y_range=(0.1, 1.2),
            resolution=4,
            bins=16,
            value_range=(0.0, 1.5),
            units="seconds",
            values=dict(trial_count=16, dt=0.01, simulation_time=1.5, seed=7),
        ).export()
        scenario = dict(
            output="rts",
            live_changes={"parameters": {"seed": 9}},
            structural_changes={
                "parameters": {"trial_count": 24},
                "preparation": {"bins": 20},
            },
            expected_samples=24,
            expected_bins=20,
            float_atol=1e-5,
            float_rtol=1e-5,
        )
    temp = tempfile.TemporaryDirectory(prefix="ordinarylight-portable-")
    root = args.output or Path(temp.name)
    package.write(root / "initial")
    (root / "conformance.json").write_text(json.dumps(scenario))
    web = Path(ordinarylight.portable.__file__).parent / "web"
    shutil.copy(web / "runtime.js", root / "runtime.js")
    shutil.copy(web / "index.html", root / "index.html")
    print(root, flush=True)
    if args.prepare_only:
        if not args.output:
            parser.error("--prepare-only requires --output")
        return
    packages = {package.manifest["id"]: package}

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(root), **kw)

        def do_POST(self):
            try:
                if self.path != "/prepare":
                    raise ValueError("Unknown endpoint")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1024**2:
                    raise ValueError("Invalid request size")
                request = json.loads(self.rfile.read(length))
                snapshot = request["snapshot"]
                current = packages[snapshot["package_id"]]
                replacement = reprepare_volume(current, snapshot, request["changes"])
                identity = replacement.manifest["id"]
                replacement.write(root / "packages" / identity)
                packages[identity] = replacement
                body = dict(url=f"/packages/{identity}/manifest.json")
                code = 200
            except Exception as error:
                body = dict(error=str(error))
                code = 400
            encoded = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    print(f"http://127.0.0.1:{args.port}", flush=True)
    try:
        ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        temp.cleanup()


if __name__ == "__main__":
    main()
