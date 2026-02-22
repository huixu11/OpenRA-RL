#!/bin/bash
set -e

REPLAY_FILE="$1"
if [ -z "$REPLAY_FILE" ]; then
    echo "Usage: /replay-viewer.sh <replay_file_path>"
    exit 1
fi

if [ ! -f "$REPLAY_FILE" ]; then
    echo "ERROR: Replay file not found: $REPLAY_FILE"
    exit 1
fi

# Tunable replay viewer settings (can be overridden via environment variables)
REPLAY_RESOLUTION="${OPENRA_RL_REPLAY_RESOLUTION:-960x540}"
if [[ "$REPLAY_RESOLUTION" == *x* ]]; then
    REPLAY_WIDTH="${REPLAY_RESOLUTION%x*}"
    REPLAY_HEIGHT="${REPLAY_RESOLUTION#*x}"
elif [[ "$REPLAY_RESOLUTION" == *,* ]]; then
    REPLAY_WIDTH="${REPLAY_RESOLUTION%,*}"
    REPLAY_HEIGHT="${REPLAY_RESOLUTION#*,}"
else
    REPLAY_WIDTH="960"
    REPLAY_HEIGHT="540"
fi
REPLAY_XVFB_DEPTH="${OPENRA_RL_REPLAY_XVFB_DEPTH:-24}"
REPLAY_UI_SCALE="${OPENRA_RL_REPLAY_UI_SCALE:-0.75}"
REPLAY_MAX_FPS="${OPENRA_RL_REPLAY_MAX_FPS:-2}"
REPLAY_VIEWPORT_DISTANCE="${OPENRA_RL_REPLAY_VIEWPORT_DISTANCE:-Close}"
REPLAY_MUTE="${OPENRA_RL_REPLAY_MUTE:-true}"
if [[ "${REPLAY_MUTE,,}" == "0" || "${REPLAY_MUTE,,}" == "false" || "${REPLAY_MUTE,,}" == "no" || "${REPLAY_MUTE,,}" == "off" ]]; then
    REPLAY_MUTE_BOOL="False"
else
    REPLAY_MUTE_BOOL="True"
fi

# Copy replay to the expected directory structure so OpenRA can read metadata
REPLAY_DIR="/root/.config/openra/Replays/ra/{DEV_VERSION}"
mkdir -p "$REPLAY_DIR"
REPLAY_BASENAME=$(basename "$REPLAY_FILE")
cp "$REPLAY_FILE" "$REPLAY_DIR/$REPLAY_BASENAME"
REPLAY_PATH="$REPLAY_DIR/$REPLAY_BASENAME"
echo "Replay copied to: $REPLAY_PATH"

# Start Xvfb (virtual framebuffer)
echo "Starting Xvfb on display :99..."
Xvfb :99 -screen 0 ${REPLAY_WIDTH}x${REPLAY_HEIGHT}x${REPLAY_XVFB_DEPTH} -ac +extension GLX +render -noreset &
XVFB_PID=$!
sleep 2
if ! kill -0 $XVFB_PID 2>/dev/null; then
    echo "ERROR: Xvfb failed to start"
    exit 1
fi
export DISPLAY=:99

# Start x11vnc
echo "Starting VNC server on port 5900..."
x11vnc -display :99 -forever -nopw -shared -rfbport 5900 -noxdamage -wait 50 -defer 50 -quiet &
VNC_PID=$!
sleep 1

# Start noVNC (websockify proxy)
echo "Starting noVNC on port 6080..."
websockify --web /usr/share/novnc 6080 localhost:5900 &
NOVNC_PID=$!
sleep 1

echo ""
echo "=== Replay viewer ready ==="
echo "Open in browser: http://localhost:6080/vnc.html"
echo "Press Ctrl+C to stop"
echo ""

# Clean shutdown on signals
cleanup() {
    echo "Shutting down replay viewer..."
    kill $NOVNC_PID 2>/dev/null || true
    kill $VNC_PID 2>/dev/null || true
    kill $XVFB_PID 2>/dev/null || true
    wait 2>/dev/null || true
    exit 0
}
trap cleanup SIGTERM SIGINT

# Launch OpenRA with low-overhead rendering settings for VNC replay viewing.
# This prioritizes replay simulation speed over visual quality.
exec dotnet /opt/openra/bin/OpenRA.dll \
    Engine.EngineDir=/opt/openra \
    Game.Mod=ra \
    Game.Platform=Default \
    Graphics.Mode=Windowed \
    Graphics.WindowedSize=${REPLAY_WIDTH},${REPLAY_HEIGHT} \
    Graphics.UIScale=${REPLAY_UI_SCALE} \
    Graphics.VSync=False \
    Graphics.CapFramerate=True \
    Graphics.MaxFramerate=${REPLAY_MAX_FPS} \
    Graphics.DisableGLDebugMessageCallback=True \
    Graphics.ViewportDistance=${REPLAY_VIEWPORT_DISTANCE} \
    Sound.Mute=${REPLAY_MUTE_BOOL} \
    "Launch.Replay=$REPLAY_PATH"
