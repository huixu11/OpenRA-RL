"""Docker orchestration for the OpenRA-RL game server."""

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from openra_env.cli.console import error, info, step, success, warn

IMAGE_REPO = "ghcr.io/yxc20089/openra-rl"
IMAGE = f"{IMAGE_REPO}:latest"
CONTAINER_NAME = "openra-rl-server"
REPLAY_CONTAINER = "openra-rl-replay"
REPLAY_DIR_IN_CONTAINER = "/root/.config/openra/Replays/ra"
LOCAL_REPLAY_DIR = Path.home() / ".openra-rl" / "replays"
MANIFEST_PATH = LOCAL_REPLAY_DIR / "manifest.json"


def _run(args: list[str], capture: bool = True, **kwargs) -> subprocess.CompletedProcess:
    """Run a subprocess command, capturing output by default."""
    return subprocess.run(
        args,
        capture_output=capture,
        text=True,
        **kwargs,
    )


def check_docker() -> bool:
    """Verify docker CLI is available and daemon is running."""
    if not shutil.which("docker"):
        error("Docker not found. Install it from https://docs.docker.com/get-docker/")
        return False
    result = _run(["docker", "info"])
    if result.returncode != 0:
        error("Docker daemon is not running. Start Docker Desktop and try again.")
        return False
    return True


def _image_tag(version: Optional[str] = None) -> str:
    """Return the full image tag for a given version (default: latest)."""
    tag = version or "latest"
    return f"{IMAGE_REPO}:{tag}"


def pull_image(version: Optional[str] = None, quiet: bool = False) -> bool:
    """Pull the game server image from GHCR."""
    image = _image_tag(version)
    if not quiet:
        step(f"Pulling game server image ({image})...")
    result = subprocess.run(
        ["docker", "pull", image],
        stdout=sys.stdout if not quiet else subprocess.DEVNULL,
        stderr=sys.stderr if not quiet else subprocess.DEVNULL,
    )
    if result.returncode != 0:
        error(f"Failed to pull {image}")
        return False
    if not quiet:
        success("Image pulled successfully.")
    return True


def image_exists(version: Optional[str] = None) -> bool:
    """Check if the game server image is available locally."""
    image = _image_tag(version)
    result = _run(["docker", "images", "-q", image])
    return bool(result.stdout.strip())


def list_local_versions() -> list[str]:
    """List all locally available openra-rl image versions (tags), newest first."""
    result = _run([
        "docker", "images", IMAGE_REPO,
        "--format", "{{.Tag}}",
    ])
    if result.returncode != 0:
        return []
    tags = [t.strip() for t in result.stdout.splitlines() if t.strip()]
    # Put "latest" first, then sort the rest in reverse
    versions = sorted([t for t in tags if t != "latest"], reverse=True)
    if "latest" in tags:
        versions.insert(0, "latest")
    return versions


def get_running_image_tag() -> Optional[str]:
    """Get the image tag of the currently running game server container."""
    if not is_running():
        return None
    result = _run([
        "docker", "inspect", CONTAINER_NAME,
        "--format", "{{.Config.Image}}",
    ])
    if result.returncode != 0:
        return None
    image = result.stdout.strip()
    # Extract tag from "ghcr.io/yxc20089/openra-rl:0.2.1"
    if ":" in image:
        return image.split(":")[-1]
    return "latest"


# ── Replay manifest ──────────────────────────────────────────────────


def _load_manifest() -> dict:
    """Load the replay manifest (replay filename → image tag)."""
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_manifest(manifest: dict) -> None:
    """Save the replay manifest."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")


def get_replay_image_tag(replay_filename: str) -> Optional[str]:
    """Look up which image tag was used to record a replay."""
    manifest = _load_manifest()
    return manifest.get(replay_filename)


def _record_replays_in_manifest(filenames: list[str], image_tag: str) -> None:
    """Record which image tag was used for newly copied replays."""
    if not filenames:
        return
    manifest = _load_manifest()
    for f in filenames:
        manifest[f] = image_tag
    _save_manifest(manifest)


def is_running() -> bool:
    """Check if the game server container is running."""
    result = _run([
        "docker", "ps", "--filter", f"name={CONTAINER_NAME}",
        "--format", "{{.Names}}"
    ])
    return CONTAINER_NAME in result.stdout


def start_server(
    port: int = 8000,
    difficulty: str = "normal",
    detach: bool = True,
    version: Optional[str] = None,
) -> bool:
    """Start the game server container."""
    if is_running():
        info(f"Server already running on port {port}.")
        return True

    image = _image_tag(version)

    # Ensure image exists
    if not image_exists(version):
        if not pull_image(version):
            return False

    step(f"Starting game server on port {port} ({image})...")
    cmd = [
        "docker", "run", "--rm",
        "-d" if detach else "",
        "-p", f"{port}:8000",
        "--name", CONTAINER_NAME,
        "-e", f"BOT_TYPE={difficulty}",
        image,
    ]
    # Remove empty strings from cmd
    cmd = [c for c in cmd if c]

    result = _run(cmd)
    if result.returncode != 0:
        error(f"Failed to start server: {result.stderr.strip()}")
        return False
    return True


def stop_server() -> bool:
    """Stop and remove the game server container."""
    if not is_running():
        info("Server is not running.")
        return True
    step("Stopping game server...")
    result = _run(["docker", "stop", CONTAINER_NAME])
    if result.returncode != 0:
        error(f"Failed to stop server: {result.stderr.strip()}")
        return False
    success("Server stopped.")
    return True


def wait_for_health(port: int = 8000, timeout: int = 120) -> bool:
    """Poll the health endpoint until the server is ready."""
    import urllib.request
    import urllib.error

    url = f"http://localhost:{port}/health"
    step(f"Waiting for server to be ready (timeout {timeout}s)...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.urlopen(url, timeout=3)
            if req.status == 200:
                success("Server is ready!")
                return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2)
    error(f"Server did not become healthy within {timeout}s.")
    return False


def get_logs(follow: bool = False) -> None:
    """Print container logs."""
    if not is_running():
        # Try to get logs from stopped container too
        pass
    cmd = ["docker", "logs"]
    if follow:
        cmd.append("-f")
    cmd.append(CONTAINER_NAME)
    subprocess.run(cmd)


def server_status() -> Optional[dict]:
    """Get server container status info."""
    if not is_running():
        return None
    result = _run([
        "docker", "ps", "--filter", f"name={CONTAINER_NAME}",
        "--format", "{{.Status}}\t{{.Ports}}"
    ])
    if result.stdout.strip():
        parts = result.stdout.strip().split("\t")
        return {
            "status": parts[0] if parts else "unknown",
            "ports": parts[1] if len(parts) > 1 else "",
        }
    return None


# ── Replay viewer ────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReplayViewerSettings:
    """Tunable replay viewer settings for quality/speed tradeoffs."""
    width: int
    height: int
    xvfb_depth: int
    ui_scale: float
    max_fps: int
    vnc_quality: int
    vnc_compression: int
    viewport_distance: str
    mute: bool
    render_mode: str


def _parse_int_setting(name: str, value: str, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer, got: {value!r}") from None
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}, got: {parsed}")
    return parsed


def _parse_float_setting(name: str, value: str, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got: {value!r}") from None
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}, got: {parsed}")
    return parsed


def _parse_bool_setting(name: str, value: str) -> bool:
    value_norm = str(value).strip().lower()
    if value_norm in ("1", "true", "yes", "on"):
        return True
    if value_norm in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be true/false, got: {value!r}")


def _parse_resolution(value: str) -> tuple[int, int]:
    raw = value.strip().lower().replace(" ", "")
    if "x" in raw:
        left, right = raw.split("x", 1)
    elif "," in raw:
        left, right = raw.split(",", 1)
    else:
        raise ValueError(f"resolution must be WxH (e.g. 960x540), got: {value!r}")
    width = _parse_int_setting("resolution width", left, 320, 7680)
    height = _parse_int_setting("resolution height", right, 240, 4320)
    return width, height


def _normalize_viewport_distance(value: str) -> str:
    mapping = {
        "close": "Close",
        "medium": "Medium",
        "far": "Far",
    }
    key = value.strip().lower()
    if key not in mapping:
        raise ValueError(f"viewport distance must be one of close/medium/far, got: {value!r}")
    return mapping[key]


def _normalize_render_mode(value: str) -> str:
    mode = value.strip().lower()
    if mode not in ("auto", "gpu", "cpu"):
        raise ValueError(f"render mode must be one of auto/gpu/cpu, got: {value!r}")
    return mode


def load_replay_viewer_settings(
    resolution: Optional[str] = None,
    max_fps: Optional[int] = None,
    ui_scale: Optional[float] = None,
    vnc_quality: Optional[int] = None,
    vnc_compression: Optional[int] = None,
    render_mode: Optional[str] = None,
) -> ReplayViewerSettings:
    """Load replay viewer settings from CLI overrides and environment variables."""
    env = os.environ

    resolution_value = resolution or env.get("OPENRA_RL_REPLAY_RESOLUTION", "960x540")
    width, height = _parse_resolution(resolution_value)

    fps_value = _parse_int_setting(
        "fps",
        str(max_fps) if max_fps is not None else env.get("OPENRA_RL_REPLAY_MAX_FPS", "2"),
        1,
        120,
    )
    ui_scale_value = _parse_float_setting(
        "ui-scale",
        str(ui_scale) if ui_scale is not None else env.get("OPENRA_RL_REPLAY_UI_SCALE", "0.75"),
        0.5,
        3.0,
    )
    vnc_quality_value = _parse_int_setting(
        "vnc-quality",
        str(vnc_quality) if vnc_quality is not None else env.get("OPENRA_RL_REPLAY_VNC_QUALITY", "8"),
        0,
        9,
    )
    vnc_compression_value = _parse_int_setting(
        "vnc-compression",
        str(vnc_compression) if vnc_compression is not None else env.get("OPENRA_RL_REPLAY_VNC_COMPRESSION", "4"),
        0,
        9,
    )
    render_mode_value = _normalize_render_mode(
        render_mode if render_mode is not None else env.get("OPENRA_RL_REPLAY_RENDER", "auto")
    )
    viewport_distance_value = _normalize_viewport_distance(
        env.get("OPENRA_RL_REPLAY_VIEWPORT_DISTANCE", "close")
    )
    xvfb_depth_value = _parse_int_setting(
        "OPENRA_RL_REPLAY_XVFB_DEPTH",
        env.get("OPENRA_RL_REPLAY_XVFB_DEPTH", "24"),
        16,
        32,
    )
    mute_value = _parse_bool_setting(
        "OPENRA_RL_REPLAY_MUTE",
        env.get("OPENRA_RL_REPLAY_MUTE", "true"),
    )

    return ReplayViewerSettings(
        width=width,
        height=height,
        xvfb_depth=xvfb_depth_value,
        ui_scale=ui_scale_value,
        max_fps=fps_value,
        vnc_quality=vnc_quality_value,
        vnc_compression=vnc_compression_value,
        viewport_distance=viewport_distance_value,
        mute=mute_value,
        render_mode=render_mode_value,
    )


def list_replays() -> list[str]:
    """List .orarep files inside the game server container."""
    if not is_running():
        return []
    result = _run([
        "docker", "exec", CONTAINER_NAME,
        "find", REPLAY_DIR_IN_CONTAINER, "-name", "*.orarep", "-type", "f",
    ])
    if result.returncode != 0:
        return []
    files = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    files.sort()
    return files


def get_latest_replay() -> Optional[str]:
    """Return the path of the newest replay inside the game server container."""
    replays = list_replays()
    return replays[-1] if replays else None


def copy_replays() -> list[str]:
    """Copy all replays from the game server container to ~/.openra-rl/replays/.

    Returns list of newly copied filenames.
    Also records the image tag in the manifest so replay watch uses the right version.
    """
    if not is_running():
        error("Game server is not running — cannot copy replays.")
        return []

    LOCAL_REPLAY_DIR.mkdir(parents=True, exist_ok=True)

    # Get list of replays in container
    replays = list_replays()
    if not replays:
        return []

    # Get existing local files to detect new ones
    existing = {f.name for f in LOCAL_REPLAY_DIR.iterdir() if f.suffix == ".orarep"}

    # Copy each replay individually (docker cp doesn't glob well)
    for replay_path in replays:
        filename = os.path.basename(replay_path)
        result = _run([
            "docker", "cp",
            f"{CONTAINER_NAME}:{replay_path}",
            str(LOCAL_REPLAY_DIR / filename),
        ])
        if result.returncode != 0:
            error(f"Failed to copy {filename}: {result.stderr.strip()}")

    # Determine which files are new
    after = {f.name for f in LOCAL_REPLAY_DIR.iterdir() if f.suffix == ".orarep"}
    new_files = sorted(after - existing)

    # Record the image version that produced these replays
    if new_files:
        tag = get_running_image_tag() or "latest"
        _record_replays_in_manifest(new_files, tag)

    return new_files


def is_replay_viewer_running() -> bool:
    """Check if the replay viewer container is running."""
    result = _run([
        "docker", "ps", "--filter", f"name={REPLAY_CONTAINER}",
        "--format", "{{.Names}}"
    ])
    return REPLAY_CONTAINER in result.stdout


def replay_viewer_exists() -> bool:
    """Check if the replay viewer container exists (running or exited)."""
    result = _run([
        "docker", "ps", "-a", "--filter", f"name={REPLAY_CONTAINER}",
        "--format", "{{.Names}}"
    ])
    return REPLAY_CONTAINER in result.stdout


def get_replay_viewer_logs(tail: int = 200) -> str:
    """Return replay viewer logs, or an empty string if unavailable."""
    if not replay_viewer_exists():
        return ""
    result = _run(["docker", "logs", "--tail", str(tail), REPLAY_CONTAINER])
    if result.returncode != 0:
        return result.stderr.strip() or result.stdout.strip()
    return result.stdout.strip()


def _is_missing_entrypoint_error(stderr: str, entrypoint: str) -> bool:
    """Return True if docker run failed because the entrypoint path does not exist."""
    s = stderr.lower()
    ep = entrypoint.lower()
    if ep not in s:
        return False
    missing_markers = (
        "no such file or directory",
        "executable file not found",
        "stat",
    )
    return any(marker in s for marker in missing_markers)


def _is_name_conflict_error(stderr: str, container_name: str) -> bool:
    """Return True if docker run failed because the container name is already in use."""
    s = stderr.lower()
    return (
        "conflict" in s
        and "already in use" in s
        and container_name.lower() in s
    )


def _render_variants(mode: str) -> list[tuple[str, list[str]]]:
    """Return docker run argument variants for replay rendering backend selection."""
    mode = _normalize_render_mode(mode)
    cpu = ("cpu", ["-e", "LIBGL_ALWAYS_SOFTWARE=1"])
    gpu_variants = [
        ("gpu-nvidia", ["--gpus", "all", "-e", "LIBGL_ALWAYS_SOFTWARE=0"]),
        ("gpu-dxg", ["--device", "/dev/dxg:/dev/dxg", "-e", "LIBGL_ALWAYS_SOFTWARE=0"]),
        ("gpu-dri", ["--device", "/dev/dri:/dev/dri", "-e", "LIBGL_ALWAYS_SOFTWARE=0"]),
    ]
    if mode == "cpu":
        return [cpu]
    if mode == "gpu":
        return gpu_variants
    return [*gpu_variants, cpu]


def _run_replay_container(
    pre_image_cmd: list[str],
    image_and_args: list[str],
    render_mode: str,
) -> tuple[subprocess.CompletedProcess, str]:
    """Run replay container using GPU variants first, then CPU fallback."""
    last_result: Optional[subprocess.CompletedProcess] = None
    last_mode = "unknown"
    for mode_label, render_args in _render_variants(render_mode):
        cmd = [*pre_image_cmd, *render_args, *image_and_args]
        # Retry each render variant once after cleaning up stale-name conflicts.
        for attempt in range(2):
            result = _run(cmd)
            last_result = result
            last_mode = mode_label
            if result.returncode == 0:
                return result, mode_label

            stderr = result.stderr.strip()
            # Failed starts can still leave an exited container with reserved name.
            _run(["docker", "rm", "-f", REPLAY_CONTAINER])

            if _is_name_conflict_error(stderr, REPLAY_CONTAINER) and attempt == 0:
                continue

            break
    # Should never be None because at least one variant always exists.
    assert last_result is not None
    return last_result, last_mode


def _common_replay_env_args(settings: ReplayViewerSettings) -> list[str]:
    """Common docker env args for replay viewer startup."""
    mute_str = "true" if settings.mute else "false"
    return [
        "-e", "SDL_AUDIODRIVER=dummy",
        "-e", "OPENRA_DISPLAY_SCALE=1",
        "-e", "vblank_mode=0",
        "-e", f"OPENRA_RL_REPLAY_RESOLUTION={settings.width}x{settings.height}",
        "-e", f"OPENRA_RL_REPLAY_XVFB_DEPTH={settings.xvfb_depth}",
        "-e", f"OPENRA_RL_REPLAY_UI_SCALE={settings.ui_scale}",
        "-e", f"OPENRA_RL_REPLAY_MAX_FPS={settings.max_fps}",
        "-e", f"OPENRA_RL_REPLAY_VIEWPORT_DISTANCE={settings.viewport_distance}",
        "-e", f"OPENRA_RL_REPLAY_MUTE={mute_str}",
    ]


def _start_replay_viewer_inline(
    image: str,
    container_replay_path: str,
    port: int,
    local_file: Optional[str],
    settings: ReplayViewerSettings,
) -> tuple[subprocess.CompletedProcess, str]:
    """Start replay viewer with an inline script (independent from image entrypoint layout)."""
    ui_scale_str = f"{settings.ui_scale:.3f}".rstrip("0").rstrip(".")
    mute_str = "True" if settings.mute else "False"
    inline_script = f"""
set -e
REPLAY_FILE="$1"
if [ -z "$REPLAY_FILE" ]; then
    echo "Usage: replay viewer <replay_file_path>"
    exit 1
fi
if [ ! -f "$REPLAY_FILE" ]; then
    echo "ERROR: Replay file not found: $REPLAY_FILE"
    exit 1
fi

REPLAY_DIR="/root/.config/openra/Replays/ra/{{DEV_VERSION}}"
mkdir -p "$REPLAY_DIR"
REPLAY_BASENAME=$(basename "$REPLAY_FILE")
cp "$REPLAY_FILE" "$REPLAY_DIR/$REPLAY_BASENAME"
REPLAY_PATH="$REPLAY_DIR/$REPLAY_BASENAME"

Xvfb :99 -screen 0 {settings.width}x{settings.height}x{settings.xvfb_depth} -ac +extension GLX +render -noreset &
XVFB_PID=$!
sleep 2
if ! kill -0 "$XVFB_PID" 2>/dev/null; then
    echo "ERROR: Xvfb failed to start"
    exit 1
fi
export DISPLAY=:99

x11vnc -display :99 -forever -nopw -shared -rfbport 5900 -noxdamage -wait 50 -defer 50 -quiet &
VNC_PID=$!
sleep 1

NOVNC_WEB=/usr/share/novnc
if [ ! -d "$NOVNC_WEB" ]; then
    NOVNC_WEB=/usr/local/share/novnc
fi
websockify --web "$NOVNC_WEB" 6080 localhost:5900 &
NOVNC_PID=$!

if [ -f /opt/openra/bin/OpenRA.dll ]; then
    OPENRA_DLL=/opt/openra/bin/OpenRA.dll
    OPENRA_DIR=/opt/openra
elif [ -f /openra/bin/OpenRA.dll ]; then
    OPENRA_DLL=/openra/bin/OpenRA.dll
    OPENRA_DIR=/openra
elif [ -f /app/OpenRA.dll ]; then
    OPENRA_DLL=/app/OpenRA.dll
    OPENRA_DIR=/app
else
    echo "ERROR: OpenRA.dll not found in known locations."
    exit 1
fi

cleanup() {{
    kill "$NOVNC_PID" 2>/dev/null || true
    kill "$VNC_PID" 2>/dev/null || true
    kill "$XVFB_PID" 2>/dev/null || true
    wait 2>/dev/null || true
}}
trap cleanup TERM INT

exec dotnet "$OPENRA_DLL" \
    Engine.EngineDir="$OPENRA_DIR" \
    Game.Mod=ra \
    Game.Platform=Default \
    Graphics.Mode=Windowed \
    Graphics.WindowedSize={settings.width},{settings.height} \
    Graphics.UIScale={ui_scale_str} \
    Graphics.VSync=False \
    Graphics.CapFramerate=True \
    Graphics.MaxFramerate={settings.max_fps} \
    Graphics.DisableGLDebugMessageCallback=True \
    Graphics.ViewportDistance={settings.viewport_distance} \
    Sound.Mute={mute_str} \
    "Launch.Replay=$REPLAY_PATH"
""".strip()

    pre_image_cmd = [
        "docker", "run", "-d",
        "-p", f"{port}:6080",
        "--name", REPLAY_CONTAINER,
        "--entrypoint", "/bin/sh",
    ]
    pre_image_cmd.extend(_common_replay_env_args(settings))

    if local_file:
        pre_image_cmd.extend(["-v", f"{local_file}:{container_replay_path}:ro"])
    elif is_running():
        pre_image_cmd.extend(["--volumes-from", CONTAINER_NAME])

    image_and_args = [image, "-c", inline_script, "openra-replay-inline", container_replay_path]
    return _run_replay_container(pre_image_cmd, image_and_args, settings.render_mode)


def start_replay_viewer(
    replay_path: str,
    port: int = 6080,
    version: Optional[str] = None,
    settings: Optional[ReplayViewerSettings] = None,
    _refreshed_latest: bool = False,
) -> bool:
    """Start the replay viewer container.

    Args:
        replay_path: Path to .orarep file (container path or local path).
        port: noVNC port to expose (default 6080).
        version: Docker image version to use (default: auto-detect from manifest).
        settings: Replay viewer tuning knobs (resolution/fps/vnc/render mode).
    """
    if settings is None:
        settings = load_replay_viewer_settings()

    if is_replay_viewer_running():
        error("Replay viewer is already running. Stop it first with: openra-rl replay stop")
        return False
    if replay_viewer_exists():
        step("Removing stale replay viewer container...")
        remove = _run(["docker", "rm", "-f", REPLAY_CONTAINER])
        if remove.returncode != 0:
            error(f"Failed to remove stale replay viewer container: {remove.stderr.strip()}")
            return False

    # Auto-detect version from manifest if not specified
    if version is None:
        filename = os.path.basename(replay_path)
        version = get_replay_image_tag(filename)
        if version:
            info(f"Using image version '{version}' (from manifest)")

    image = _image_tag(version)

    if not image_exists(version):
        step(f"Image {image} not found locally, pulling...")
        if not pull_image(version):
            return False

    # Determine if this is a local file or a container path.
    # If the file exists on the host, mount it into the container.
    # Otherwise, assume it's a path inside the game server container.
    local_file = None
    container_replay_path = replay_path
    local_path = Path(replay_path).resolve()

    if local_path.exists():
        local_file = str(local_path)
        container_replay_path = f"/tmp/replay/{local_path.name}"
    elif replay_path.startswith("/") and is_running():
        # Prefer copying replay to host and bind-mounting it.
        # `--volumes-from` only shares named volumes, not container writable layers.
        filename = os.path.basename(replay_path)
        LOCAL_REPLAY_DIR.mkdir(parents=True, exist_ok=True)
        local_copy = LOCAL_REPLAY_DIR / filename
        copy_result = _run([
            "docker", "cp",
            f"{CONTAINER_NAME}:{replay_path}",
            str(local_copy),
        ])
        if copy_result.returncode == 0:
            local_file = str(local_copy.resolve())
            container_replay_path = f"/tmp/replay/{filename}"
            info(f"Copied replay from server container: {filename}")
        else:
            warn(
                "Could not copy replay from server container; "
                "trying container path fallback."
            )
    elif not replay_path.startswith("/"):
        error(f"Replay file not found: {local_path}")
        return False

    step(f"Starting replay viewer on port {port} ({image})...")

    last_stderr = ""

    # Prefer the inline launcher for consistent behavior across image versions.
    inline_result, inline_mode = _start_replay_viewer_inline(
        image=image,
        container_replay_path=container_replay_path,
        port=port,
        local_file=local_file,
        settings=settings,
    )
    if inline_result.returncode == 0:
        info("Using inline replay viewer launcher.")
        if inline_mode.startswith("gpu"):
            info(f"Rendering mode: {inline_mode} (hardware acceleration)")
        else:
            warn("Rendering mode: cpu (software fallback)")
        success("Replay viewer started.")
        return True

    stderr = inline_result.stderr.strip()
    if _is_name_conflict_error(stderr, REPLAY_CONTAINER):
        step("Replay viewer container name is in use; removing stale container and retrying...")
        remove = _run(["docker", "rm", "-f", REPLAY_CONTAINER])
        if remove.returncode != 0:
            error(f"Failed to remove conflicting replay viewer container: {remove.stderr.strip()}")
            return False
        inline_result, inline_mode = _start_replay_viewer_inline(
            image=image,
            container_replay_path=container_replay_path,
            port=port,
            local_file=local_file,
            settings=settings,
        )
        if inline_result.returncode == 0:
            info("Using inline replay viewer launcher.")
            if inline_mode.startswith("gpu"):
                info(f"Rendering mode: {inline_mode} (hardware acceleration)")
            else:
                warn("Rendering mode: cpu (software fallback)")
            success("Replay viewer started.")
            return True
        stderr = inline_result.stderr.strip()

    last_stderr = stderr
    _run(["docker", "rm", "-f", REPLAY_CONTAINER])
    warn("Inline replay launcher failed; trying image entrypoints...")

    # Support multiple image layouts across released tags.
    entrypoint_candidates = [
        "/replay-viewer.sh",
        "/app/replay-viewer.sh",
        "/app/docker/replay-viewer.sh",
        "/docker/replay-viewer.sh",
    ]

    for idx, entrypoint in enumerate(entrypoint_candidates):
        pre_image_cmd = [
            "docker", "run", "-d",
            "-p", f"{port}:6080",
            "--name", REPLAY_CONTAINER,
            "--entrypoint", entrypoint,
        ]
        pre_image_cmd.extend(_common_replay_env_args(settings))

        if local_file:
            # Mount the local replay file
            pre_image_cmd.extend(["-v", f"{local_file}:{container_replay_path}:ro"])
        elif is_running():
            # Share replay volume from game server container
            pre_image_cmd.extend(["--volumes-from", CONTAINER_NAME])

        image_and_args = [image, container_replay_path]

        result, render_mode = _run_replay_container(
            pre_image_cmd,
            image_and_args,
            settings.render_mode,
        )
        if result.returncode == 0:
            if idx > 0:
                info(f"Using replay viewer entrypoint: {entrypoint}")
            if render_mode.startswith("gpu"):
                info(f"Rendering mode: {render_mode} (hardware acceleration)")
            else:
                warn("Rendering mode: cpu (software fallback)")
            success("Replay viewer started.")
            return True

        stderr = result.stderr.strip()
        if _is_name_conflict_error(stderr, REPLAY_CONTAINER):
            step("Replay viewer container name is in use; removing stale container and retrying...")
            remove = _run(["docker", "rm", "-f", REPLAY_CONTAINER])
            if remove.returncode != 0:
                error(f"Failed to remove conflicting replay viewer container: {remove.stderr.strip()}")
                return False
            result, render_mode = _run_replay_container(
                pre_image_cmd,
                image_and_args,
                settings.render_mode,
            )
            if result.returncode == 0:
                if idx > 0:
                    info(f"Using replay viewer entrypoint: {entrypoint}")
                if render_mode.startswith("gpu"):
                    info(f"Rendering mode: {render_mode} (hardware acceleration)")
                else:
                    warn("Rendering mode: cpu (software fallback)")
                success("Replay viewer started.")
                return True
            stderr = result.stderr.strip()

        last_stderr = stderr
        # A failed `docker run` may leave an exited container with the same name.
        _run(["docker", "rm", "-f", REPLAY_CONTAINER])
        if _is_missing_entrypoint_error(stderr, entrypoint):
            continue

        error(f"Failed to start replay viewer: {stderr}")
        return False

    warn(
        "Replay viewer entrypoint not found in image after inline fallback; tried: "
        + ", ".join(entrypoint_candidates)
    )
    if not _refreshed_latest and (version in (None, "latest")):
        warn("Replay viewer failed on local 'latest' image. Pulling fresh image and retrying once...")
        if pull_image("latest"):
            return start_replay_viewer(
                replay_path=replay_path,
                port=port,
                version="latest",
                settings=settings,
                _refreshed_latest=True,
            )
    if last_stderr:
        error(f"Failed to start replay viewer: {last_stderr}")
    else:
        error("Failed to start replay viewer.")
    return False


def stop_replay_viewer() -> bool:
    """Stop the replay viewer container."""
    if not replay_viewer_exists():
        info("Replay viewer is not running.")
        return True
    step("Stopping replay viewer...")
    result = _run(["docker", "rm", "-f", REPLAY_CONTAINER])
    if result.returncode != 0:
        error(f"Failed to stop replay viewer: {result.stderr.strip()}")
        return False
    success("Replay viewer stopped.")
    return True
