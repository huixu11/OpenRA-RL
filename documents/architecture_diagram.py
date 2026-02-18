"""
OpenRA-RL Architecture Diagram
From the LLM Agent's perspective, showing MCP servers, tools, and game engine internals.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Style constants ──────────────────────────────────────────────
C = {
    "bg": "#0d1117",
    "box_border": "#30363d",
    "text": "#e6edf3",
    "text_dim": "#8b949e",
    "text_sub": "#b0b8c4",
    "agent": "#bc8cff",
    "mcp_client": "#58a6ff",
    "mcp_server": "#2b5b84",
    "env": "#2b5b84",
    "grpc_client": "#E84D31",
    "process": "#58a6ff",
    "proto": "#F5A623",
    "grpc_port": "#E84D31",
    "csharp": "#178600",
    "csharp_dark": "#2d6a2d",
    "world": "#2d4a2d",
    "channels": "#d29922",
    "docker": "#2496ED",
    "reward": "#3fb950",
    "models": "#bc8cff",
    "arrow_obs": "#58a6ff",
    "arrow_action": "#f0883e",
    "arrow_ctrl": "#8b949e",
    "arrow_mcp": "#bc8cff",
    "game_loop": "#00d4aa",
    "layer_agent": "#1f1a30",
    "layer_mcp": "#1a2235",
    "layer_backend": "#1a2235",
    "layer_bridge": "#2a1a1a",
    "layer_csharp": "#1a2a1a",
}

W, H = 24, 34
fig, ax = plt.subplots(1, 1, figsize=(W, H), facecolor=C["bg"])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect("equal")
ax.axis("off")
fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01)


def box(x, y, w, h, label, color, sub=None, fs=11, bold=True, tc=None, alpha=0.85, cr=0.3):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={cr}",
                        facecolor=color, edgecolor="white", linewidth=1.2, alpha=alpha, zorder=3)
    ax.add_patch(p)
    t = tc or C["text"]
    wt = "bold" if bold else "normal"
    if sub:
        ax.text(x + w/2, y + h/2 + 0.18, label, ha="center", va="center",
                fontsize=fs, fontweight=wt, color=t, zorder=4)
        sc = "#1a1a1a" if tc == "#1a1a1a" else C["text_sub"]
        ax.text(x + w/2, y + h/2 - 0.22, sub, ha="center", va="center",
                fontsize=max(fs - 3, 7), color=sc, zorder=4, style="italic")
    else:
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=fs, fontweight=wt, color=t, zorder=4)


def layer(x, y, w, h, color, label, lx=None, ly=None):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.4",
                        facecolor=color, edgecolor=C["box_border"], linewidth=1.5, alpha=0.5, zorder=1)
    ax.add_patch(p)
    ax.text(lx or x + 0.5, ly or y + h - 0.3, label, ha="left", va="top",
            fontsize=12, fontweight="bold", color=C["text_dim"], zorder=2, fontstyle="italic")


def arrow(x1, y1, x2, y2, color, label=None, lw=2.0, lo=(0, 0), cs="arc3,rad=0", fs=9, style="-|>"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, color=color, lw=lw,
                         connectionstyle=cs, zorder=5, mutation_scale=18)
    ax.add_patch(a)
    if label:
        mx, my = (x1+x2)/2 + lo[0], (y1+y2)/2 + lo[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=fs, color=color,
                zorder=6, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", facecolor=C["bg"], edgecolor=color,
                          alpha=0.9, linewidth=0.8))


def tool_list_box(x, y_top, w, title, items, title_color, bg_color):
    """Compact tool box with tight, centered 2-column layout. y_top = top edge. Returns (y_bottom, y_top)."""
    cols = 2
    n = len(items)
    rows = (n + cols - 1) // cols
    line_h = 0.36
    pad_top = 0.55
    pad_bot = 0.25
    h = pad_top + rows * line_h + pad_bot
    y = y_top - h
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.2",
                        facecolor=bg_color, edgecolor=title_color, linewidth=1.5, alpha=0.6, zorder=2)
    ax.add_patch(p)
    ax.text(x + w/2, y_top - 0.28, title, ha="center", va="center",
            fontsize=11, fontweight="bold", color=title_color, zorder=4)
    # Measure actual column widths and center them in the box
    char_w = 0.125
    col0_items = items[:rows]
    col1_items = items[rows:]
    max_col0 = max(len(s) for s in col0_items) * char_w if col0_items else 0
    max_col1 = max(len(s) for s in col1_items) * char_w if col1_items else 0
    gap = 0.5   # gap between columns
    total_content_w = max_col0 + gap + max_col1
    left_pad = (w - total_content_w) / 2
    first_item_y = y_top - pad_top
    for i, item in enumerate(items):
        col = i // rows
        row = i % rows
        if col == 0:
            ix = x + left_pad
        else:
            ix = x + left_pad + max_col0 + gap
        iy = first_item_y - row * line_h
        ax.text(ix, iy, item, ha="left", va="center",
                fontsize=11, color=C["text_sub"], zorder=4, fontfamily="monospace")
    return y, y_top


# ══════════════════════════════════════════════════════════════════
#  TITLE
# ══════════════════════════════════════════════════════════════════

ax.text(W/2, H - 0.6, "OpenRA-RL System Architecture", ha="center", va="center",
        fontsize=26, fontweight="bold", color=C["text"], zorder=10)
ax.text(W/2, H - 1.15, "From the LLM Agent's Perspective", ha="center", va="center",
        fontsize=13, color=C["text_dim"], zorder=10)

# ══════════════════════════════════════════════════════════════════
#  LAYER BACKGROUNDS
# ══════════════════════════════════════════════════════════════════

# Agent layer (OUTSIDE Docker)
layer(0.5, 30.0, W - 1, 2.5, C["layer_agent"], "LLM AGENT")

# Docker container outline (covers layers 2-5, fully wrapping MCP Server layer)
docker_top = 28.0   # just above MCP layer top (19.5 + 8.3 = 27.8)
docker_box = FancyBboxPatch((0.3, 0.3), W - 0.6, docker_top - 0.3,
                             boxstyle="round,pad=0,rounding_size=0.5",
                             facecolor="none", edgecolor=C["docker"], linewidth=2.5,
                             linestyle=(0, (8, 4)), alpha=0.7, zorder=0)
ax.add_patch(docker_box)
ax.text(W - 1.0, docker_top - 0.3, "Docker Container", ha="right", va="top",
        fontsize=12, fontweight="bold", color=C["docker"], zorder=2,
        bbox=dict(boxstyle="round,pad=0.2", facecolor=C["bg"], edgecolor=C["docker"],
                  alpha=0.9, linewidth=1.0))

# Inner layers (inside Docker)
layer(0.6, 19.5, W - 1.2, 8.3, C["layer_mcp"], "MCP SERVER (FastMCP + OpenEnv)")
layer(0.6, 12.0, W - 1.2, 7.0, C["layer_backend"], "PYTHON BACKEND")
layer(0.6, 8.5, W - 1.2, 3.0, C["layer_bridge"], "gRPC + PROTOBUF BRIDGE")
layer(0.6, 0.6, W - 1.2, 7.5, C["layer_csharp"], "C# GAME ENGINE (OpenRA / .NET 8)")

# ══════════════════════════════════════════════════════════════════
#  AGENT LAYER  (y: 28.0 – 30.5)
# ══════════════════════════════════════════════════════════════════

box(3.0, 30.7, 6.5, 1.3, "LLM Agent", C["agent"],
    sub="examples/llm_agent.py (41 MCP tools)", fs=14)

box(11.5, 30.7, 5.5, 1.3, "LLM Model", "#6e40aa",
    sub="Claude / GPT via OpenRouter", fs=14)

arrow(9.5, 31.35, 11.5, 31.35, C["text_dim"], label="prompt + tools", lw=1.8, lo=(0, 0.35), fs=9)

# ══════════════════════════════════════════════════════════════════
#  MCP SERVER LAYER  (y: 19.5 – 27.3)
# ══════════════════════════════════════════════════════════════════

box(3.0, 26.0, 6.5, 1.3, "MCP Client", C["mcp_client"],
    sub="WebSocket ws://localhost:8000/ws", fs=12)

box(13.5, 26.0, 7.0, 1.3, "OpenEnv Server", C["mcp_server"],
    sub="app.py (FastMCP + Uvicorn)", fs=12)

# Agent -> MCP Client (straight down from agent center)
arrow(6.25, 30.7, 6.25, 27.3, C["arrow_mcp"], label="JSON-RPC", lw=2.0, lo=(-1.1, 0), fs=9)

# MCP Client -> OpenEnv Server
arrow(9.5, 26.65, 13.5, 26.65, C["mcp_client"], label="tools/call", lw=2.0, lo=(0, 0.32), fs=9)

# ── Tool boxes ───────────────────────────────────────────────────

obs_tools = ["get_game_state", "get_economy", "get_units", "get_buildings",
             "get_enemies", "get_production", "get_map_info", "get_terrain_at"]

act_tools = ["move_units", "attack_move", "attack_target", "stop_units",
             "build_unit", "build_structure", "build_and_place", "place_building",
             "deploy_unit", "sell_building", "repair_building", "set_rally_point",
             "guard_target", "set_stance", "harvest", "cancel_production",
             "power_down", "set_primary", "surrender", "advance"]

kc_tools = ["lookup_unit", "lookup_building", "lookup_tech_tree", "lookup_faction",
            "batch", "plan", "assign_group", "add_to_group",
            "get_groups", "command_group", "get_valid_placements", "get_replay_path"]

TOOLS_TOP = 24.2
obs_bot, _ = tool_list_box(1.0, TOOLS_TOP, 6.5, "Observation Tools (8)", obs_tools, "#58a6ff", "#0d1f3a")
act_bot, _ = tool_list_box(8.0, TOOLS_TOP, 7.5, "Action Tools (20)", act_tools, "#f0883e", "#2a1a0d")
kc_bot, _  = tool_list_box(16.0, TOOLS_TOP, 7.0, "Knowledge & Composite (13)", kc_tools, "#3fb950", "#0d2a14")

# OpenEnv Server -> tool boxes (spread across bottom of server, not from single point)
srv_by = 26.0   # bottom y of server
arrow(14.5, srv_by, 4.25, TOOLS_TOP, "#58a6ff", lw=1.3, cs="arc3,rad=0.15")
arrow(17.0, srv_by, 11.75, TOOLS_TOP, "#f0883e", lw=1.3, cs="arc3,rad=0.05")
arrow(19.5, srv_by, 19.5, TOOLS_TOP, "#3fb950", lw=1.3)

# ══════════════════════════════════════════════════════════════════
#  PYTHON BACKEND LAYER  (y: 12.0 – 19.0)
# ══════════════════════════════════════════════════════════════════

# Top row
box(4.0, 16.5, 7.0, 1.5, "OpenRAEnvironment", C["env"],
    sub="MCPEnvironment impl", fs=13)
box(14.0, 16.5, 5.0, 1.5, "Game Data", C["channels"],
    sub="game_data.py (static RA stats)", fs=10, alpha=0.7)

# Bottom row
box(1.5, 13.0, 5.5, 1.5, "BridgeClient", C["grpc_client"],
    sub="bridge_client.py (async gRPC)", fs=12)
box(8.5, 13.0, 5.5, 1.5, "ProcessManager", C["process"],
    sub="openra_process.py", fs=12)
box(15.5, 13.0, 3.5, 1.5, "Models", C["models"],
    sub="models.py (Pydantic)", fs=11)
box(20.0, 13.0, 3.0, 1.5, "Reward", C["reward"],
    sub="reward.py", fs=11)

# Tools -> OpenRAEnvironment (obs & action arrive at top-center of env)
env_tx = 7.5    # top center x of OpenRAEnvironment
env_ty = 18.0   # top y
arrow(4.25, obs_bot, env_tx, env_ty, "#58a6ff", lw=1.5, cs="arc3,rad=0.05")
arrow(11.75, act_bot, env_tx, env_ty, "#f0883e", lw=1.5, cs="arc3,rad=-0.05")

# Knowledge tools -> Game Data (separate target)
arrow(19.5, kc_bot, 16.5, 18.0, "#3fb950", label="static data", lw=1.2, lo=(0.8, 0.3), fs=8,
      cs="arc3,rad=-0.1")

# OpenRAEnvironment -> backend row (all from bottom-center of env)
env_bx = 7.5    # bottom center x
env_by = 16.5   # bottom y
arrow(env_bx, env_by, 4.25, 14.5, C["arrow_obs"], lw=1.8, cs="arc3,rad=0.08")
arrow(env_bx, env_by, 11.25, 14.5, C["arrow_ctrl"], label="launch/kill",
      lw=1.5, lo=(0.8, 0.3), fs=8)
arrow(env_bx, env_by, 17.25, 14.5, C["text_dim"], lw=1.0, cs="arc3,rad=-0.05")
arrow(env_bx, env_by, 21.5, 14.5, C["text_dim"], lw=1.0, cs="arc3,rad=-0.08")

# ══════════════════════════════════════════════════════════════════
#  gRPC + PROTOBUF BRIDGE  (y: 8.5 – 11.5)
# ══════════════════════════════════════════════════════════════════

box(1.5, 9.2, 10.0, 1.5, "rl_bridge.proto", C["proto"],
    sub="GameSession (bidir stream) + GetState (unary)", fs=12, tc="#1a1a1a")

box(13.0, 9.2, 6.0, 1.5, "Port 9999 (gRPC)", C["grpc_port"],
    sub="Kestrel Server in C# Engine", fs=11)

# BridgeClient -> Bridge (from bottom-center of BridgeClient, fan to proto targets)
bc_bx = 4.25   # bottom center x of BridgeClient
bc_by = 13.0   # bottom y
arrow(bc_bx, bc_by, 3.5, 10.7, C["arrow_obs"], label="Observations",
      lw=2.5, lo=(-1.5, 0), fs=10, cs="arc3,rad=0.05")
arrow(bc_bx, bc_by, 7.5, 10.7, C["arrow_action"], label="Actions",
      lw=2.5, lo=(1.5, 0), fs=10, cs="arc3,rad=-0.05")

# ProcessManager -> Game (down to Port 9999 area)
arrow(11.25, 13.0, 14.0, 10.7, C["arrow_ctrl"], label="dotnet OpenRA.dll",
      lw=1.5, lo=(0.5, 0.5), fs=8)

# ══════════════════════════════════════════════════════════════════
#  C# GAME ENGINE LAYER  (y: 0.6 – 8.1)
# ══════════════════════════════════════════════════════════════════

# Top row (y=5.5, smaller boxes with gaps)
box(1.5, 5.5, 4.5, 1.3, "ExternalBotBridge", C["csharp"],
    sub="IBot + ITick + Kestrel", fs=11)

box(7.0, 5.5, 4.5, 1.3, "RLBridgeService", C["csharp"],
    sub="gRPC (decoupled loops)", fs=11)

box(12.5, 5.5, 4.5, 1.3, "ActionHandler", C["csharp"],
    sub="Proto -> OpenRA Orders", fs=11)

box(18.0, 5.5, 4.0, 1.3, "Spatial Map", C["csharp_dark"],
    sub="9ch H x W x 9", fs=10)

# Bottom row (y=2.5, smaller boxes with gaps)
box(1.5, 2.5, 4.0, 1.3, "Channels", C["channels"],
    sub="DropOldest (obs=1)", fs=10, alpha=0.8)

box(6.5, 2.5, 5.0, 1.3, "ObservationSerializer", C["csharp_dark"],
    sub="World state -> protobuf", fs=10)

box(12.5, 2.5, 4.5, 1.3, "Game World", C["world"],
    sub="Actors, Map, Rules, Fog", fs=10)

box(18.0, 2.5, 4.0, 1.3, "Game Loop", C["csharp"],
    sub="~25 ticks/sec", fs=10)

# ── C# internal arrows ──────────────────────────────────────────

# Bridge -> ExternalBotBridge (both from same origin point on proto box bottom)
proto_bx = 4.5   # single origin x (center-left of proto box)
proto_by = 9.2   # bottom y of proto box
arrow(proto_bx, proto_by, 2.5, 6.8, C["arrow_action"], lw=2.0, cs="arc3,rad=0.08")
arrow(proto_bx, proto_by, 5.0, 6.8, C["arrow_obs"], lw=2.0, cs="arc3,rad=-0.08")

# ExternalBotBridge -> RLBridgeService
arrow(6.0, 6.15, 7.0, 6.15, C["reward"], lw=1.5,
      label="Task.WhenAny", lo=(0, 0.32), fs=8)

# RLBridgeService -> ActionHandler
arrow(11.5, 6.15, 12.5, 6.15, C["arrow_action"], lw=1.5)

# ExternalBotBridge -> Channels (straight down)
arrow(3.75, 5.5, 3.5, 3.8, C["channels"], lw=1.5)

# Channels -> ObservationSerializer
arrow(5.5, 3.15, 6.5, 3.15, C["arrow_obs"], lw=1.5)

# Game World -> ObservationSerializer
arrow(12.5, 3.15, 11.5, 3.15, C["arrow_obs"], lw=1.5)

# ActionHandler -> Game World (straight down)
arrow(14.75, 5.5, 14.75, 3.8, C["arrow_action"], lw=1.5)

# Game World -> Spatial Map (diagonal up)
arrow(17.0, 3.4, 18.0, 5.5, C["arrow_obs"], lw=1.2, cs="arc3,rad=-0.2")

# Game Loop -> Game World (left arrow)
arrow(18.0, 3.15, 17.0, 3.15, C["game_loop"], lw=1.5)

# Game Loop ticks ExternalBotBridge (route around bottom, connect to box left edge)
arrow(20.0, 2.5, 20.0, 1.5, C["game_loop"], lw=1.5)    # down from game loop
ax.annotate("", xy=(1.0, 1.5), xytext=(20.0, 1.5),
            arrowprops=dict(arrowstyle="-", color=C["game_loop"], lw=1.5), zorder=5)
ax.annotate("", xy=(1.0, 6.15), xytext=(1.0, 1.5),
            arrowprops=dict(arrowstyle="-", color=C["game_loop"], lw=1.5), zorder=5)
arrow(1.0, 6.15, 1.5, 6.15, C["game_loop"], lw=1.5)    # horizontal to box left edge
ax.text(10.5, 1.2, "ITick (~25/sec)", ha="center", va="center", fontsize=10,
        color=C["game_loop"], zorder=6, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.15", facecolor=C["bg"], edgecolor=C["game_loop"],
                  alpha=0.9, linewidth=1.0))

# ══════════════════════════════════════════════════════════════════
#  LEGEND (top-right)
# ══════════════════════════════════════════════════════════════════

lx, ly = 18.5, 32.2
legend_h = 2.2
legend_bg = FancyBboxPatch((lx - 0.3, ly - legend_h), 5.5, legend_h + 0.3,
    boxstyle="round,pad=0,rounding_size=0.2",
    facecolor=C["bg"], edgecolor=C["box_border"], linewidth=1.2, alpha=1.0, zorder=9)
ax.add_patch(legend_bg)
ax.text(lx, ly, "Data Flow:", fontsize=10, fontweight="bold", color=C["text_dim"], zorder=10)

for i, (color, label) in enumerate([
    (C["arrow_mcp"], "MCP (agent <-> server)"),
    (C["arrow_obs"], "Observations (game -> agent)"),
    (C["arrow_action"], "Actions (agent -> game)"),
    (C["arrow_ctrl"], "Process / Control"),
    (C["game_loop"], "Game Loop (ITick)"),
]):
    yy = ly - 0.38 * (i + 1)
    ax.plot([lx, lx + 0.7], [yy, yy], color=color, lw=2.5, zorder=10)
    ax.text(lx + 0.95, yy, label, fontsize=9, color=C["text_dim"], va="center", zorder=10)

# ── Save ─────────────────────────────────────────────────────────
out = "/Users/berta/Projects/OpenRA-RL/documents/architecture_diagram.png"
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=C["bg"], edgecolor="none")
plt.close()
print(f"Architecture diagram saved to: {out}")
