"""Run browser/native conformance with an installed scientific Python stack.

Requires Chrome, Node >=20, and ordinarylight[webgpu] plus the pinned upstream
packages. Owns a temporary browser profile and server; preserves diagnostic logs.
"""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen


def wait_for(process, ready, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Process exited with {process.returncode}; see logs")
        try:
            result = ready()
            if result:
                return result
        except (OSError, ValueError):
            pass
        time.sleep(0.2)
    raise TimeoutError("Startup timed out; see logs")


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chrome", default="google-chrome")
    parser.add_argument("--node", default="node")
    parser.add_argument("--headless", action="store_true", help="Also run hardware Chrome without a window")
    args = parser.parse_args()
    for executable in (args.chrome, args.node):
        if not shutil.which(executable):
            parser.error(f"Executable not found: {executable}")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    example = Path(__file__).resolve().parent
    # Copy the host so imports cannot resolve against a repository source root.
    for name in ("serve.py", "dmc.json", "verify_results.py"):
        shutil.copy2(example / name, root / name)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}/"
    with ExitStack() as stack:
        profile = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="portable-chrome-")))

        def launch(command, log):
            stream = stack.enter_context((root / log).open("w"))
            process = subprocess.Popen(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
            stack.callback(stop, process)
            return process

        server = launch([sys.executable, "-I", str(root / "serve.py"), "--port", str(port),
                         "--output", str(root)], "server.log")

        def server_ready():
            with urlopen(url + "initial/manifest.json", timeout=1) as response:
                return response.status == 200

        wait_for(server, server_ready)
        flags = ["--ozone-platform=x11", "--use-angle=vulkan", "--enable-features=Vulkan,VulkanFromANGLE"]
        if args.headless:
            flags.append("--headless=new")
        browser = launch([args.chrome, f"--user-data-dir={profile}", "--remote-debugging-port=0",
                          "--no-first-run", "--no-default-browser-check",
                          *flags, "about:blank"], "chrome.log")
        cdp_port = wait_for(browser, lambda: (profile / "DevToolsActivePort").read_text().splitlines()[0])
        commands = [
            [args.node, "--experimental-websocket", str(example / "browser_probe.mjs"),
             f"http://127.0.0.1:{cdp_port}", url, str(root / "browser.json")],
            [sys.executable, "-I", str(root / "verify_results.py"), str(root)],
        ]
        for command, log in zip(commands, ("probe.log", "native.log")):
            with (root / log).open("w") as stream:
                subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT,
                               check=True, timeout=180)
        report = json.loads((root / "verification.json").read_text())
        report["headless"] = args.headless
        report["chrome_version"] = subprocess.check_output([args.chrome, "--version"], text=True).strip()
        (root / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Browser/native conformance passed: {root / 'verification.json'}")


if __name__ == "__main__":
    main()
