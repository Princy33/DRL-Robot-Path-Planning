import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CSV_DIR = os.path.join(RESULTS_DIR, "csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
BACKEND_DIR = os.path.join(BASE_DIR, "backend")

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from environment import RobotPathPlanningEnv

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DRL Lab | Robot Path Planning",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# PREMIUM UI / CSS
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
    --purple: #8b5cf6;
    --violet: #a78bfa;
    --pink: #ec4899;
    --teal: #2dd4bf;
    --gold: #f5c16c;
    --cream: #fff8f0;
    --ink: #151326;
}

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 7% 7%, rgba(139,92,246,.27), transparent 24%),
        radial-gradient(circle at 94% 4%, rgba(45,212,191,.18), transparent 22%),
        radial-gradient(circle at 90% 70%, rgba(236,72,153,.18), transparent 27%),
        radial-gradient(circle at 5% 90%, rgba(245,193,108,.12), transparent 23%),
        linear-gradient(135deg, #080a18 0%, #10142a 38%, #1b1630 68%, #081b24 100%);
    color: #f7f5ff;
    min-height: 100vh;
}

.block-container {
    max-width: 1480px;
    padding-top: 2.2rem;
    padding-bottom: 4rem;
}

h1 {
    color: #ffffff !important;
    font-weight: 900 !important;
    letter-spacing: -1.8px;
}

h2 {
    color: #faf7ff !important;
    font-weight: 850 !important;
}

h3 {
    color: #eee9ff !important;
    font-weight: 800 !important;
}

p, .stMarkdown, .stCaption {
    color: #c8c3d8;
}

hr {
    border-color: rgba(255,255,255,.10) !important;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background:
        radial-gradient(circle at 15% 8%, rgba(139,92,246,.24), transparent 28%),
        radial-gradient(circle at 90% 88%, rgba(45,212,191,.14), transparent 25%),
        linear-gradient(180deg, #080a17 0%, #11142b 48%, #091a22 100%);
    border-right: 1px solid rgba(255,255,255,.11);
    box-shadow: 14px 0 45px rgba(0,0,0,.34);
}

.drl-logo-box {
    text-align: center;
    padding: 21px 12px 18px;
    border-radius: 25px;
    background:
        radial-gradient(circle at 50% 0%, rgba(236,72,153,.22), transparent 45%),
        linear-gradient(145deg, rgba(139,92,246,.22), rgba(45,212,191,.09));
    border: 1px solid rgba(255,255,255,.13);
    box-shadow: 0 20px 45px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.09);
}

.drl-logo-icon {
    width: 70px;
    height: 70px;
    margin: 0 auto 11px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 37px;
    border-radius: 22px;
    background: linear-gradient(135deg, #8b5cf6, #ec4899, #14b8a6);
    box-shadow: 0 0 30px rgba(168,85,247,.42), 0 12px 30px rgba(0,0,0,.32);
}

.drl-logo-title {
    font-size: 25px;
    font-weight: 900;
    letter-spacing: 2px;
    color: white;
}

.drl-logo-subtitle {
    margin-top: 5px;
    font-size: 9px;
    font-weight: 800;
    letter-spacing: 1.7px;
    color: #c9c2d9;
}

.decor {
    text-align: center;
    color: #d9b5ff;
    font-size: 20px;
    letter-spacing: 8px;
    margin: 9px 0 20px;
    text-shadow: 0 0 18px rgba(216,167,255,.45);
}

.nav-title {
    font-size: 10px;
    font-weight: 900;
    letter-spacing: 1.8px;
    color: #9e98ae;
    margin: 15px 3px 8px;
}

/* Selected / hover navigation */
div[role="radiogroup"] {
    gap: 7px !important;
}

div[role="radiogroup"] label {
    width: 100% !important;
    padding: 11px 12px !important;
    border-radius: 14px !important;
    border: 1px solid transparent !important;
    background: transparent !important;
    color: #bdb7cc !important;
    transition: all .23s ease !important;
}

div[role="radiogroup"] label:hover {
    background: linear-gradient(90deg, rgba(139,92,246,.17), rgba(236,72,153,.10)) !important;
    border-color: rgba(168,85,247,.23) !important;
    color: white !important;
    transform: translateX(4px) !important;
}

div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(100deg, rgba(124,58,237,.44), rgba(192,38,211,.27), rgba(20,184,166,.14)) !important;
    border-color: rgba(216,180,254,.40) !important;
    color: white !important;
    box-shadow: 0 9px 27px rgba(124,58,237,.28), inset 0 1px 0 rgba(255,255,255,.11) !important;
    transform: translateX(5px) !important;
}

div[role="radiogroup"] input {
    accent-color: #c084fc;
}

.sidebar-card {
    margin-top: 16px;
    padding: 15px;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(255,255,255,.075), rgba(255,255,255,.025));
    border: 1px solid rgba(255,255,255,.10);
    box-shadow: inset 0 1px 0 rgba(255,255,255,.06), 0 12px 30px rgba(0,0,0,.20);
}

.sidebar-card-title {
    color: white;
    font-size: 12px;
    font-weight: 800;
}

.sidebar-card-text {
    color: #aaa5b8;
    font-size: 10px;
    line-height: 1.7;
    margin-top: 5px;
}

.model-pill {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 7px;
    padding: 8px 10px;
    border-radius: 10px;
    background: rgba(20,184,166,.08);
    border: 1px solid rgba(45,212,191,.14);
    color: #b9f7ec;
    font-size: 10px;
    font-weight: 800;
}

.model-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #34d399;
    box-shadow: 0 0 10px rgba(52,211,153,.85);
}

/* HERO */
.hero {
    position: relative;
    overflow: hidden;
    padding: 34px;
    border-radius: 30px;
    background:
        radial-gradient(circle at 90% 15%, rgba(45,212,191,.19), transparent 24%),
        radial-gradient(circle at 15% 0%, rgba(139,92,246,.26), transparent 30%),
        linear-gradient(135deg, rgba(255,255,255,.10), rgba(255,255,255,.035));
    border: 1px solid rgba(255,255,255,.13);
    box-shadow: 0 28px 65px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.09);
}

.hero:after {
    content: "✦  ❀  ✦  ❀  ✦";
    position: absolute;
    right: 28px;
    bottom: 18px;
    color: rgba(255,255,255,.20);
    font-size: 18px;
    letter-spacing: 7px;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    padding: 8px 15px;
    border-radius: 50px;
    background: linear-gradient(90deg, rgba(139,92,246,.25), rgba(236,72,153,.16), rgba(45,212,191,.13));
    border: 1px solid rgba(255,255,255,.14);
    color: #eee6ff;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.2px;
}

.hero h1 {
    margin: 18px 0 7px !important;
    font-size: 43px !important;
}

.hero-sub {
    max-width: 850px;
    color: #c9c4d6;
    font-size: 15px;
    line-height: 1.75;
}

.section-label {
    margin-top: 26px;
    margin-bottom: 10px;
    color: #d8d0e7;
    font-size: 11px;
    font-weight: 900;
    letter-spacing: 2px;
}

.glass-card {
    padding: 22px;
    border-radius: 23px;
    background: linear-gradient(135deg, rgba(255,255,255,.085), rgba(255,255,255,.028));
    border: 1px solid rgba(255,255,255,.11);
    box-shadow: 0 18px 42px rgba(0,0,0,.20), inset 0 1px 0 rgba(255,255,255,.06);
}

.info-card {
    min-height: 180px;
    padding: 22px;
    border-radius: 22px;
    background: linear-gradient(145deg, rgba(139,92,246,.14), rgba(236,72,153,.08), rgba(45,212,191,.06));
    border: 1px solid rgba(255,255,255,.10);
    box-shadow: 0 16px 35px rgba(0,0,0,.18);
}

.info-icon {
    font-size: 28px;
}

.info-title {
    color: #fff;
    font-size: 17px;
    font-weight: 850;
    margin: 7px 0;
}

.info-text {
    color: #bdb7cc;
    font-size: 12px;
    line-height: 1.7;
}

/* METRICS */
div[data-testid="stMetric"] {
    min-height: 120px;
    padding: 19px;
    border-radius: 21px;
    background: linear-gradient(135deg, rgba(255,255,255,.095), rgba(255,255,255,.03));
    border: 1px solid rgba(255,255,255,.12);
    box-shadow: 0 17px 42px rgba(0,0,0,.23), inset 0 1px 0 rgba(255,255,255,.06);
    backdrop-filter: blur(14px);
}

div[data-testid="stMetricLabel"] {
    color: #aaa4b9 !important;
    font-weight: 700;
}

div[data-testid="stMetricValue"] {
    color: white !important;
    font-size: 29px !important;
    font-weight: 900 !important;
}

/* BUTTONS */
.stButton > button {
    min-height: 43px;
    border: 1px solid rgba(255,255,255,.13);
    border-radius: 13px;
    background: linear-gradient(135deg, #7c3aed, #c026d3);
    color: white;
    font-weight: 800;
    box-shadow: 0 8px 25px rgba(124,58,237,.28);
    transition: all .22s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 13px 35px rgba(192,38,211,.38);
    border-color: rgba(255,255,255,.25);
}

.stButton > button:disabled {
    opacity: .42;
}

/* ALERTS */
div[data-testid="stAlert"] {
    border-radius: 17px !important;
    border: 1px solid rgba(255,255,255,.10) !important;
    background: rgba(255,255,255,.055) !important;
    backdrop-filter: blur(12px);
}

/* SELECTBOX */
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,.055);
    border: 1px solid rgba(255,255,255,.12);
    border-radius: 12px;
}

/* IMAGES */
img {
    border-radius: 19px;
    box-shadow: 0 20px 50px rgba(0,0,0,.28);
}

/* TABLE */
div[data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,.10);
}

/* SIMULATOR */
.sim-title {
    text-align: center;
    color: white;
    font-size: 25px;
    font-weight: 900;
    margin-bottom: 3px;
}

.sim-subtitle {
    text-align: center;
    color: #aaa5b9;
    font-size: 12px;
    margin-bottom: 18px;
}

.status-success {
    padding: 13px 17px;
    border-radius: 15px;
    background: rgba(52,211,153,.12);
    border: 1px solid rgba(52,211,153,.25);
    color: #a7f3d0;
    font-weight: 800;
}

.status-collision {
    padding: 13px 17px;
    border-radius: 15px;
    background: rgba(251,113,133,.11);
    border: 1px solid rgba(251,113,133,.23);
    color: #fecdd3;
    font-weight: 800;
}

.status-ready {
    padding: 13px 17px;
    border-radius: 15px;
    background: rgba(167,139,250,.11);
    border: 1px solid rgba(167,139,250,.22);
    color: #ddd6fe;
    font-weight: 800;
}

.footer {
    margin-top: 55px;
    padding: 27px;
    text-align: center;
    border-top: 1px solid rgba(255,255,255,.08);
    color: #777186;
    font-size: 11px;
    line-height: 1.8;
}

::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: #080a18; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(#7c3aed, #c026d3);
    border-radius: 10px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# HELPERS
# ============================================================

def find_file(filename):
    """Search recursively inside results so folder differences do not break the app."""
    if not os.path.exists(RESULTS_DIR):
        return None

    for root, _, files in os.walk(RESULTS_DIR):
        for file in files:
            if file.lower() == filename.lower():
                return os.path.join(root, file)

    return None


def load_csv(filename):
    path = find_file(filename)
    if path is None:
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.error(f"Unable to read {filename}: {exc}")
        return None


def show_graph(filename, title=None, height=None):
    path = find_file(filename)

    if title:
        st.markdown(f"### {title}")

    if path is None:
        st.warning(f"Graph not available: {filename}")
        return False

    st.image(path, use_container_width=True)
    return True


def metric_value(df, possible_columns, default="—", fmt=".2f"):
    for col in possible_columns:
        if col in df.columns:
            try:
                value = df[col].mean()
                return format(value, fmt)
            except Exception:
                return default
    return default


def get_success_rate(df):
    if df is None or "Success" not in df.columns:
        return 0.0
    return float(df["Success"].mean() * 100)


def get_collision_rate(df):
    if df is None or "Collision" not in df.columns:
        return 0.0
    return float(df["Collision"].mean() * 100)


# ============================================================
# MANUAL SIMULATOR
# ============================================================

def create_simulator():
    from environment import RobotPathPlanningEnv

    env = RobotPathPlanningEnv(
        grid_size=20,
        max_steps=150,
        random_obstacles=False,
    )
    state, _ = env.reset(seed=42)
    return env, state


def init_simulator():
    # Initialize every simulator state variable independently.
    # This prevents AttributeError when Streamlit reruns the app
    # after navigation or when switching between simulator pages.
    if "sim_env" not in st.session_state:
        env, state = create_simulator()
        st.session_state.sim_env = env
        st.session_state.sim_state = state
        st.session_state.sim_total_reward = 0.0
        st.session_state.sim_last_reward = 0.0
        st.session_state.sim_done = False
        st.session_state.sim_collision = False
        st.session_state.sim_success = False
        st.session_state.sim_last_action = "Ready"
        st.session_state.sim_seed = 42
        st.session_state.sim_steps = 0
        st.session_state.sim_history = []
    else:
        # Backward-compatible initialization for sessions created by an
        # older version of this app.
        defaults = {
            "sim_steps": 0,
            "sim_history": [],
            "sim_total_reward": 0.0,
            "sim_last_reward": 0.0,
            "sim_done": False,
            "sim_collision": False,
            "sim_success": False,
            "sim_last_action": "Ready",
            "sim_seed": 42,
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value


def reset_simulator(seed=42):
    env = RobotPathPlanningEnv(
        grid_size=20,
        max_steps=150,
        random_obstacles=False,
    )
    state, info = env.reset(seed=seed)

    st.session_state.sim_env = env
    st.session_state.sim_state = state
    st.session_state.sim_info = info
    st.session_state.sim_total_reward = 0.0
    st.session_state.sim_last_reward = 0.0
    st.session_state.sim_steps = 0
    st.session_state.sim_done = False
    st.session_state.sim_collision = False
    st.session_state.sim_success = False
    st.session_state.sim_last_action = "Reset"
    st.session_state.sim_seed = seed
    st.session_state.sim_history = []


def move_robot(action, action_name):
    env = st.session_state.sim_env

    if st.session_state.sim_done:
        return

    next_state, reward, terminated, truncated, info = env.step(action)

    st.session_state.sim_state = next_state
    st.session_state.sim_total_reward += float(reward)
    st.session_state.sim_last_reward = float(reward)
    st.session_state.sim_last_action = action_name

    collision = bool(info.get("collision", False))
    success = bool(info.get("success", False))

    st.session_state.sim_collision = collision
    st.session_state.sim_success = success
    st.session_state.sim_done = bool(terminated or truncated)


def draw_environment(env):
    size = env.grid_size

    grid = [[0 for _ in range(size)] for _ in range(size)]

    for x, y in env.obstacles:
        if 0 <= x < size and 0 <= y < size:
            grid[x][y] = 1

    fig, ax = plt.subplots(figsize=(8, 8))

    cmap = ListedColormap(["#fbf8f2", "#28233a"])
    ax.imshow(grid, cmap=cmap, origin="upper", interpolation="none")

    # Path
    path = list(getattr(env, "path", []))
    if len(path) >= 2:
        xs = [p[1] for p in path]
        ys = [p[0] for p in path]
        ax.plot(xs, ys, linewidth=3.0, marker="o", markersize=3.5, alpha=0.80)

    # Start
    sx, sy = env.start_position
    ax.scatter(sy, sx, s=260, marker="s", edgecolors="white", linewidths=1.5, zorder=5)
    ax.text(sy, sx, "S", ha="center", va="center", color="white", fontweight="bold", zorder=6)

    # Goal
    gx, gy = env.goal_position
    ax.scatter(gy, gx, s=330, marker="*", edgecolors="white", linewidths=1.5, zorder=6)
    ax.text(gy, gx, "G", ha="center", va="center", color="white", fontweight="bold", zorder=7)

    # Robot
    rx, ry = env.robot_position
    ax.scatter(ry, rx, s=260, marker="o", edgecolors="white", linewidths=2, zorder=8)
    ax.text(ry, rx, "R", ha="center", va="center", color="white", fontweight="bold", zorder=9)

    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, linewidth=.35, alpha=.32)
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)
    ax.set_title(
        "20 × 20 Robot Navigation Environment",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )
    fig.tight_layout()

    return fig


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="drl-logo-box">
            <div class="drl-logo-icon">🤖</div>
            <div class="drl-logo-title">DRL LAB</div>
            <div class="drl-logo-subtitle">AUTONOMOUS ROBOT NAVIGATION</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="decor">✦ ❀ ✦</div>', unsafe_allow_html=True)

    st.markdown('<div class="nav-title">PROJECT NAVIGATION</div>', unsafe_allow_html=True)

    page = st.radio(
        "PROJECT NAVIGATION",
        [
            "🏠 Project Overview",
            "📊 Training Results",
            "🧠 DQN Analysis",
            "✨ DDQN Analysis",
            "⚖️ DQN vs DDQN",
            "🌿 Random Environment",
            "🎮 Live Simulator",
            "🤖 Robot Path",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown(
        """
        <div class="sidebar-card">
            <div class="sidebar-card-title">🎓 M.Sc. Artificial Intelligence</div>
            <div class="sidebar-card-text">
                Deep Reinforcement Learning<br>
                Robot Path Planning<br>
                Academic Year 2026–2027
            </div>
        </div>

        <div class="nav-title">MODEL STATUS</div>

        <div class="model-pill">
            <span class="model-dot"></span>
            DQN MODEL READY
        </div>

        <div class="model-pill">
            <span class="model-dot"></span>
            DDQN MODEL READY
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# PROJECT OVERVIEW
# ============================================================

if page == "🏠 Project Overview":
    st.markdown(
        """
        <div class="hero">
            <div class="hero-badge">🧠 DEEP REINFORCEMENT LEARNING LAB</div>
            <h1>🤖 Robot Path Planning</h1>
            <div class="hero-sub">
                An intelligent autonomous navigation system using
                <b>DQN</b> and <b>Double DQN</b> to learn obstacle-aware
                robot movement in a custom 20 × 20 environment.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">SYSTEM AT A GLANCE</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("GRID", "20 × 20")
    with c2:
        st.metric("STATE SPACE", "405")
    with c3:
        st.metric("ACTIONS", "4")
    with c4:
        st.metric("ALGORITHMS", "DQN + DDQN")

    st.markdown('<div class="section-label">INTELLIGENT NAVIGATION PIPELINE</div>', unsafe_allow_html=True)

    a1, a2, a3 = st.columns(3)

    with a1:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-icon">🗺️</div>
                <div class="info-title">Environment</div>
                <div class="info-text">
                    A custom Gymnasium 20 × 20 grid with a robot,
                    target position and obstacle configuration.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with a2:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-icon">🧠</div>
                <div class="info-title">Learning Agent</div>
                <div class="info-text">
                    DQN and DDQN neural networks learn action values
                    from rewards and replayed experiences.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with a3:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-icon">🎯</div>
                <div class="info-title">Objective</div>
                <div class="info-text">
                    Reach the goal efficiently while reducing collisions,
                    unnecessary movement and poor navigation decisions.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-label">TECHNOLOGY STACK</div>', unsafe_allow_html=True)

    t1, t2, t3, t4, t5 = st.columns(5)
    for col, icon, name in [
        (t1, "🐍", "Python"),
        (t2, "🔥", "PyTorch"),
        (t3, "🎮", "Gymnasium"),
        (t4, "📊", "Matplotlib"),
        (t5, "🌐", "Streamlit"),
    ]:
        with col:
            st.markdown(
                f"""
                <div class="glass-card" style="text-align:center;">
                    <div style="font-size:28px;">{icon}</div>
                    <div style="color:white;font-weight:800;margin-top:5px;">{name}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-label">PROJECT PIPELINE</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="glass-card" style="text-align:center;">
            <div style="font-size:16px;color:#eee6ff;font-weight:800;">
                Environment
                <span style="color:#c084fc;"> → </span>
                State
                <span style="color:#ec4899;"> → </span>
                Neural Network
                <span style="color:#2dd4bf;"> → </span>
                Action
                <span style="color:#f5c16c;"> → </span>
                Reward
                <span style="color:#c084fc;"> → </span>
                Replay Memory
                <span style="color:#ec4899;"> → </span>
                Learning
                <span style="color:#2dd4bf;"> → </span>
                Goal
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# TRAINING RESULTS
# ============================================================

elif page == "📊 Training Results":
    st.title("📊 Training Results")
    st.write("Learning behaviour and performance across 500 training episodes.")
    st.markdown('<div class="decor">✦ ❀ ✦ ❀ ✦</div>', unsafe_allow_html=True)

    dqn = load_csv("dqn_training_results.csv")
    ddqn = load_csv("ddqn_training_results.csv")

    if dqn is not None:
        st.header("🧠 DQN Training")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Episodes", len(dqn))
        with c2:
            st.metric("Success Rate", f"{get_success_rate(dqn):.2f}%")
        with c3:
            st.metric("Average Reward", metric_value(dqn, ["Cumulative_Reward"]))
        with c4:
            st.metric("Average Loss", metric_value(dqn, ["Training_Loss"], fmt=".4f"))

        show_graph("episode_vs_cumulative_reward.png", "Episode vs Cumulative Reward")
        show_graph("episode_vs_average_reward.png", "Episode vs Average Reward")
        show_graph("episode_vs_training_loss.png", "Episode vs Training Loss")
        show_graph("episode_vs_episode_length.png", "Episode vs Episode Length")
        show_graph("exploration_rate_vs_episode.png", "Exploration Rate vs Episode")

    st.divider()

    if ddqn is not None:
        st.header("✨ DDQN Training")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Episodes", len(ddqn))
        with c2:
            st.metric("Success Rate", f"{get_success_rate(ddqn):.2f}%")
        with c3:
            st.metric("Average Reward", metric_value(ddqn, ["Cumulative_Reward"]))
        with c4:
            st.metric("Average Loss", metric_value(ddqn, ["Training_Loss"], fmt=".4f"))

        show_graph("ddqn_episode_vs_cumulative_reward.png", "DDQN — Episode vs Cumulative Reward")
        show_graph("ddqn_episode_vs_average_reward.png", "DDQN — Episode vs Average Reward")
        show_graph("ddqn_episode_vs_training_loss.png", "DDQN — Episode vs Training Loss")
        show_graph("ddqn_episode_vs_episode_length.png", "DDQN — Episode vs Episode Length")
        show_graph("ddqn_exploration_rate_vs_episode.png", "DDQN — Exploration Rate vs Episode")

# ============================================================
# DQN ANALYSIS
# ============================================================

elif page == "🧠 DQN Analysis":
    st.title("🧠 DQN Analysis")
    st.write("Deep Q-Network training and fixed-environment testing analysis.")
    st.markdown('<div class="decor">✦ ❀ ✦</div>', unsafe_allow_html=True)

    dqn = load_csv("dqn_training_results.csv")
    test = load_csv("dqn_test_results.csv")

    if dqn is not None:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Training Episodes", len(dqn))
        with c2:
            st.metric("Training Success", f"{get_success_rate(dqn):.2f}%")
        with c3:
            st.metric("Average Reward", metric_value(dqn, ["Cumulative_Reward"]))
        with c4:
            st.metric("Final Epsilon", metric_value(dqn.tail(1), ["Epsilon"], fmt=".3f"))

    if test is not None:
        st.markdown('<div class="section-label">FIXED ENVIRONMENT TEST PERFORMANCE</div>', unsafe_allow_html=True)

        reward_col = "Cumulative Reward" if "Cumulative Reward" in test.columns else "Cumulative_Reward"
        length_col = "Episode Length" if "Episode Length" in test.columns else "Episode_Length"
        path_col = "Path Length" if "Path Length" in test.columns else "Path_Length"

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Test Success", f"{get_success_rate(test):.2f}%")
        with c2:
            st.metric("Collision Rate", f"{get_collision_rate(test):.2f}%")
        with c3:
            st.metric("Average Reward", f"{test[reward_col].mean():.2f}" if reward_col in test.columns else "—")
        with c4:
            st.metric("Episode Length", f"{test[length_col].mean():.0f}" if length_col in test.columns else "—")

        if path_col in test.columns:
            st.caption(f"Average path length: {test[path_col].mean():.0f} steps/positions.")

    st.markdown('<div class="section-label">DQN LEARNING CURVES</div>', unsafe_allow_html=True)
    show_graph("episode_vs_cumulative_reward.png")
    show_graph("episode_vs_average_reward.png")
    show_graph("episode_vs_training_loss.png")
    show_graph("episode_vs_episode_length.png")
    show_graph("exploration_rate_vs_episode.png")

# ============================================================
# DDQN ANALYSIS
# ============================================================

elif page == "✨ DDQN Analysis":
    st.title("✨ Double Deep Q-Network")
    st.write("DDQN training and fixed-environment testing analysis.")
    st.markdown('<div class="decor">✦ ❀ ✦</div>', unsafe_allow_html=True)

    ddqn = load_csv("ddqn_training_results.csv")
    test = load_csv("ddqn_test_results.csv")

    if ddqn is not None:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Training Episodes", len(ddqn))
        with c2:
            st.metric("Training Success", f"{get_success_rate(ddqn):.2f}%")
        with c3:
            st.metric("Average Reward", metric_value(ddqn, ["Cumulative_Reward"]))
        with c4:
            st.metric("Final Epsilon", metric_value(ddqn.tail(1), ["Epsilon"], fmt=".3f"))

    if test is not None:
        st.markdown('<div class="section-label">FIXED ENVIRONMENT TEST PERFORMANCE</div>', unsafe_allow_html=True)

        reward_col = "Cumulative Reward" if "Cumulative Reward" in test.columns else "Cumulative_Reward"
        length_col = "Episode Length" if "Episode Length" in test.columns else "Episode_Length"

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Test Success", f"{get_success_rate(test):.2f}%")
        with c2:
            st.metric("Collision Rate", f"{get_collision_rate(test):.2f}%")
        with c3:
            st.metric("Average Reward", f"{test[reward_col].mean():.2f}" if reward_col in test.columns else "—")
        with c4:
            st.metric("Episode Length", f"{test[length_col].mean():.0f}" if length_col in test.columns else "—")

    st.markdown('<div class="section-label">DDQN LEARNING CURVES</div>', unsafe_allow_html=True)
    show_graph("ddqn_episode_vs_cumulative_reward.png")
    show_graph("ddqn_episode_vs_average_reward.png")
    show_graph("ddqn_episode_vs_training_loss.png")
    show_graph("ddqn_episode_vs_episode_length.png")
    show_graph("ddqn_exploration_rate_vs_episode.png")

# ============================================================
# DQN VS DDQN
# ============================================================

elif page == "⚖️ DQN vs DDQN":
    st.title("⚖️ DQN vs DDQN")
    st.write("Comparative experimental evaluation of DQN and Double DQN.")
    st.markdown('<div class="decor">✦ ❀ ✦ ❀ ✦</div>', unsafe_allow_html=True)

    comparison = load_csv("dqn_vs_ddqn_comparison.csv")

    if comparison is not None:
        st.markdown('<div class="section-label">EXPERIMENTAL COMPARISON</div>', unsafe_allow_html=True)
        st.dataframe(comparison, use_container_width=True, hide_index=True)
    else:
        st.error("Comparison CSV not found. Run: python backend/compare_models.py")

    st.markdown('<div class="section-label">COMPARISON VISUALS</div>', unsafe_allow_html=True)

    show_graph("dqn_vs_ddqn_training_reward.png", "📈 Training Reward")
    show_graph("dqn_vs_ddqn_training_loss.png", "📉 Training Loss")
    show_graph("dqn_vs_ddqn_success_rate.png", "🎯 Success Rate")
    show_graph("dqn_vs_ddqn_test_performance.png", "🧪 Test Performance")

    st.info(
        "Research interpretation: DDQN separates action selection from action evaluation "
        "in the target calculation, reducing the tendency of standard DQN to overestimate Q-values."
    )

# ============================================================
# RANDOM ENVIRONMENT
# ============================================================

elif page == "🌿 Random Environment":
    st.title("🌿 Random Environment")
    st.write("Evaluation on unseen random obstacle configurations.")
    st.markdown('<div class="decor">✦ ❀ ✦ ❀ ✦</div>', unsafe_allow_html=True)

    comparison = load_csv("random_dqn_vs_ddqn_comparison.csv")

    if comparison is not None:
        st.markdown('<div class="section-label">GENERALIZATION RESULTS</div>', unsafe_allow_html=True)
        st.dataframe(comparison, use_container_width=True, hide_index=True)

    else:
        st.error("Random comparison file not found. Run: python backend/compare_random_models.py")

    st.markdown('<div class="section-label">RANDOM-ENVIRONMENT COMPARISON</div>', unsafe_allow_html=True)

    show_graph("random_dqn_vs_ddqn_training_reward.png", "📈 Training Reward")
    show_graph("random_dqn_vs_ddqn_training_loss.png", "📉 Training Loss")
    show_graph("random_dqn_vs_ddqn_success_rate.png", "🎯 Success Rate")
    show_graph("random_dqn_vs_ddqn_test_performance.png", "🧪 Test Performance")
    show_graph("random_dqn_vs_ddqn_test_reward.png", "💰 Test Reward")
    show_graph("random_dqn_vs_ddqn_final_distance.png", "📍 Final Distance")

    st.warning(
        "Note: these random-environment results evaluate generalization to unseen obstacle layouts. "
        "The current trained random-map models show that higher obstacle density remains challenging."
    )

# ============================================================
# LIVE MANUAL SIMULATOR
# ============================================================

elif page == "🎮 Live Simulator":
    st.markdown(
        """
        <div class="hero">
            <div class="hero-badge">🎮 INTERACTIVE ENVIRONMENT</div>
            <h1>🤖 Live Robot Simulator</h1>
            <div class="hero-sub">
                Manually control the robot through the same custom 20 × 20
                navigation environment used by the project.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    init_simulator()
    env = st.session_state.sim_env

    st.markdown('<div class="section-label">SIMULATION STATUS</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric("ROBOT", str(env.robot_position))
    with c2:
        st.metric("GOAL", str(env.goal_position))
    with c3:
        st.metric("STEPS", env.steps)
    with c4:
        st.metric("LAST REWARD", f"{st.session_state.sim_last_reward:.2f}")
    with c5:
        st.metric("TOTAL REWARD", f"{st.session_state.sim_total_reward:.2f}")

    left, right = st.columns([1.45, 1])

    with left:
        st.pyplot(draw_environment(env), use_container_width=True)
        plt.close("all")

    with right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="sim-title">CONTROL PANEL</div>', unsafe_allow_html=True)
        st.markdown('<div class="sim-subtitle">Use the directional controls to move the robot</div>', unsafe_allow_html=True)

        if st.session_state.sim_success:
            st.markdown('<div class="status-success">🎯 GOAL REACHED — SUCCESS!</div>', unsafe_allow_html=True)
        elif st.session_state.sim_collision:
            st.markdown('<div class="status-collision">⚠️ COLLISION DETECTED</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="status-ready">● Status: {st.session_state.sim_last_action}</div>',
                unsafe_allow_html=True,
            )

        st.write("")

        col_a, col_b, col_c = st.columns(3)
        with col_b:
            if st.button("⬆️ UP", use_container_width=True, disabled=st.session_state.sim_done):
                move_robot(0, "Up")
                st.rerun()

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("⬅️ LEFT", use_container_width=True, disabled=st.session_state.sim_done):
                move_robot(2, "Left")
                st.rerun()
        with col_b:
            if st.button("⬇️ DOWN", use_container_width=True, disabled=st.session_state.sim_done):
                move_robot(1, "Down")
                st.rerun()
        with col_c:
            if st.button("➡️ RIGHT", use_container_width=True, disabled=st.session_state.sim_done):
                move_robot(3, "Right")
                st.rerun()

        st.write("")

        if st.button("🔄 RESET SIMULATION", use_container_width=True):
            reset_simulator(42)
            st.rerun()

        st.markdown(
            """
            <br>
            <div class="sidebar-card">
                <div class="sidebar-card-title">Legend</div>
                <div class="sidebar-card-text">
                    🟣 R = Robot<br>
                    ⭐ G = Goal<br>
                    ■ = Obstacle<br>
                    ─ = Robot Path
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-label">SIMULATION INFORMATION</div>', unsafe_allow_html=True)

    i1, i2, i3 = st.columns(3)
    with i1:
        st.markdown(
            '<div class="glass-card"><b>Action Space</b><br><span style="color:#aaa5b8;">Up • Down • Left • Right</span></div>',
            unsafe_allow_html=True,
        )
    with i2:
        st.markdown(
            '<div class="glass-card"><b>Environment</b><br><span style="color:#aaa5b8;">20 × 20 • Fixed Obstacles</span></div>',
            unsafe_allow_html=True,
        )
    with i3:
        st.markdown(
            '<div class="glass-card"><b>Maximum Steps</b><br><span style="color:#aaa5b8;">150 steps per episode</span></div>',
            unsafe_allow_html=True,
        )

# ============================================================
# ROBOT PATH + LIVE MANUAL SIMULATOR
# ============================================================

elif page == "🤖 Robot Path":

    # --------------------------------------------------------
    # BACKEND IMPORT
    # --------------------------------------------------------
    import sys
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    BACKEND_DIR = os.path.join(BASE_DIR, "backend")
    if BACKEND_DIR not in sys.path:
        sys.path.insert(0, BACKEND_DIR)

    try:
        from environment import RobotPathPlanningEnv
        ENV_AVAILABLE = True
    except Exception as e:
        ENV_AVAILABLE = False
        ENV_ERROR = str(e)

    # --------------------------------------------------------
    # PAGE HEADER
    # --------------------------------------------------------
    st.title("🤖 Robot Navigation Studio")

    st.markdown(
        """
        <div class="glass">
            <h2 style="margin:0; color:#ffffff;">🎮 Live Robot Path Planning Simulator</h2>
            <p style="margin:8px 0 0 0;">
                Control the robot manually inside the trained environment,
                observe rewards in real time, and visualize the learned DQN path.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='decor'>✦ ❀ ✦ ❀ ✦</div>",
        unsafe_allow_html=True
    )

    if not ENV_AVAILABLE:
        st.error("Environment could not be loaded.")
        st.code(ENV_ERROR)
    else:

        # ----------------------------------------------------
        # SESSION STATE
        # ----------------------------------------------------
        if "sim_env" not in st.session_state:
            st.session_state.sim_env = RobotPathPlanningEnv(
                grid_size=20,
                max_steps=150,
                random_obstacles=False
            )
            state, info = st.session_state.sim_env.reset(seed=42)
            st.session_state.sim_state = state
            st.session_state.sim_info = info
            st.session_state.sim_total_reward = 0.0
            st.session_state.sim_steps = 0
            st.session_state.sim_done = False
            st.session_state.sim_collision = False
            st.session_state.sim_success = False
            st.session_state.sim_last_action = "Ready"
            st.session_state.sim_history = []

        env = st.session_state.sim_env

        # ----------------------------------------------------
        # RESET
        # ----------------------------------------------------
        def robot_path_reset_simulator():
            state, info = env.reset(seed=42)
            st.session_state.sim_state = state
            st.session_state.sim_info = info
            st.session_state.sim_total_reward = 0.0
            st.session_state.sim_steps = 0
            st.session_state.sim_done = False
            st.session_state.sim_collision = False
            st.session_state.sim_success = False
            st.session_state.sim_last_action = "Ready"
            st.session_state.sim_history = []

        def robot_path_perform_action(action, action_name):
            if st.session_state.sim_done:
                return

            next_state, reward, terminated, truncated, info = env.step(action)

            st.session_state.sim_state = next_state
            st.session_state.sim_info = info
            st.session_state.sim_total_reward += float(reward)
            st.session_state.sim_steps += 1
            st.session_state.sim_last_action = action_name

            collision = bool(info.get("collision", False))
            success = bool(info.get("success", False))

            st.session_state.sim_collision = collision
            st.session_state.sim_success = success
            st.session_state.sim_done = bool(terminated or truncated)

            st.session_state.sim_history.append(
                {
                    "Step": st.session_state.sim_steps,
                    "Action": action_name,
                    "Reward": float(reward),
                    "Total Reward": st.session_state.sim_total_reward,
                    "Position": str(env.robot_position),
                    "Collision": collision,
                    "Success": success
                }
            )

        # ----------------------------------------------------
        # TOP STATUS CARDS
        # ----------------------------------------------------
        robot_pos = getattr(env, "robot_position", (0, 0))
        goal_pos = getattr(env, "goal_position", (19, 19))

        m1, m2, m3, m4, m5 = st.columns(5)

        with m1:
            st.metric("🤖 Robot", f"{robot_pos}")

        with m2:
            st.metric("🎯 Goal", f"{goal_pos}")

        with m3:
            st.metric("👣 Steps", st.session_state.sim_steps)

        with m4:
            st.metric(
                "💰 Total Reward",
                f"{st.session_state.sim_total_reward:.2f}"
            )

        with m5:
            distance = abs(robot_pos[0] - goal_pos[0]) + abs(
                robot_pos[1] - goal_pos[1]
            )
            st.metric("📍 Distance", distance)

        st.markdown("<br>", unsafe_allow_html=True)

        # ----------------------------------------------------
        # SIMULATOR LAYOUT
        # ----------------------------------------------------
        left, right = st.columns([1.65, 1], gap="large")

        with left:

            st.markdown(
                """
                <div class="glass">
                    <h3 style="margin-top:0;">🗺️ Interactive 20 × 20 Grid</h3>
                    <p>
                        <b>Green</b> = Start &nbsp; • &nbsp;
                        <b>Red</b> = Goal &nbsp; • &nbsp;
                        <b>Blue</b> = Robot Path &nbsp; • &nbsp;
                        <b>Black</b> = Obstacles
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Build grid image.
            grid_size = env.grid_size
            grid = np.zeros((grid_size, grid_size), dtype=int)

            for obstacle in env.obstacles:
                r, c = obstacle
                if 0 <= r < grid_size and 0 <= c < grid_size:
                    grid[r, c] = 1

            fig, ax = plt.subplots(figsize=(8.2, 8.2))

            cmap = ListedColormap([
                "#F7F4EF",   # open
                "#24213A"    # obstacle
            ])

            ax.imshow(
                grid,
                cmap=cmap,
                origin="upper",
                interpolation="none"
            )

            # Path.
            if getattr(env, "path", None):
                path = list(env.path)
                ys = [p[1] for p in path]
                xs = [p[0] for p in path]

                ax.plot(
                    ys,
                    xs,
                    linewidth=3.0,
                    marker="o",
                    markersize=3.5,
                    color="#7C3AED",
                    alpha=0.80
                )

            # Start.
            ax.scatter(
                env.start_position[1],
                env.start_position[0],
                s=170,
                marker="o",
                color="#22C55E",
                edgecolors="white",
                linewidths=1.5,
                zorder=5
            )

            # Goal.
            ax.scatter(
                goal_pos[1],
                goal_pos[0],
                s=230,
                marker="*",
                color="#EF4444",
                edgecolors="white",
                linewidths=1.5,
                zorder=6
            )

            # Robot.
            ax.scatter(
                robot_pos[1],
                robot_pos[0],
                s=220,
                marker="o",
                color="#06B6D4",
                edgecolors="white",
                linewidths=2,
                zorder=7
            )

            ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
            ax.grid(
                which="minor",
                linewidth=0.45,
                alpha=0.30
            )

            ax.set_xticks(range(0, grid_size, 2))
            ax.set_yticks(range(0, grid_size, 2))
            ax.set_xlabel("Grid Column")
            ax.set_ylabel("Grid Row")
            ax.set_title(
                "LIVE ROBOT NAVIGATION",
                fontsize=15,
                fontweight="bold",
                pad=12
            )

            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with right:

            st.markdown(
                """
                <div class="glass">
                    <h3 style="margin-top:0;">🎮 Manual Control Panel</h3>
                    <p>
                        Use the controls to move the robot one cell at a time.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Direction pad.
            a, b, c = st.columns(3)

            with b:
                if st.button(
                    "⬆️ UP",
                    use_container_width=True,
                    disabled=st.session_state.sim_done,
                    key="sim_up"
                ):
                    robot_path_perform_action(0, "UP")
                    st.rerun()

            a, b, c = st.columns(3)

            with a:
                if st.button(
                    "⬅️ LEFT",
                    use_container_width=True,
                    disabled=st.session_state.sim_done,
                    key="sim_left"
                ):
                    robot_path_perform_action(2, "LEFT")
                    st.rerun()

            with b:
                if st.button(
                    "⏺️ STAY",
                    use_container_width=True,
                    disabled=st.session_state.sim_done,
                    key="sim_stay"
                ):
                    # There is no native STAY action in the 4-action
                    # environment, so this button is intentionally disabled
                    # semantically by not taking a step.
                    st.session_state.sim_last_action = "STAY"
                    st.rerun()

            with c:
                if st.button(
                    "➡️ RIGHT",
                    use_container_width=True,
                    disabled=st.session_state.sim_done,
                    key="sim_right"
                ):
                    robot_path_perform_action(3, "RIGHT")
                    st.rerun()

            a, b, c = st.columns(3)

            with b:
                if st.button(
                    "⬇️ DOWN",
                    use_container_width=True,
                    disabled=st.session_state.sim_done,
                    key="sim_down"
                ):
                    robot_path_perform_action(1, "DOWN")
                    st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)

            if st.button(
                "🔄 RESET SIMULATOR",
                use_container_width=True,
                key="sim_reset"
            ):
                robot_path_reset_simulator()
                st.rerun()

            # Current action status.
            st.markdown(
                f"""
                <div class="glass">
                    <h4 style="margin:0;">⚡ Current Status</h4>
                    <p style="margin:8px 0;">
                        Last Action:
                        <b>{st.session_state.sim_last_action}</b>
                    </p>
                    <p style="margin:8px 0;">
                        Position:
                        <b>{robot_pos}</b>
                    </p>
                    <p style="margin:8px 0;">
                        Goal:
                        <b>{goal_pos}</b>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            if st.session_state.sim_success:
                st.success(
                    "🎉 GOAL REACHED! The robot successfully reached the target."
                )
            elif st.session_state.sim_collision:
                st.warning(
                    "⚠️ Collision detected. The robot can continue from its current state."
                )
            elif st.session_state.sim_done:
                st.error(
                    "⏱️ Maximum episode steps reached. Reset and try another path."
                )
            else:
                st.info(
                    "🟢 Simulation running — choose a direction to move the robot."
                )

            # Reward explanation.
            st.markdown(
                """
                <div class="glass">
                    <h4 style="margin:0;">💡 Reward Feedback</h4>
                    <p style="margin:8px 0;">
                        The environment rewards progress toward the goal and
                        penalizes collisions, revisits and inefficient movement.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ----------------------------------------------------
        # ACTION HISTORY
        # ----------------------------------------------------
        if st.session_state.sim_history:

            st.divider()

            st.subheader("📜 Simulation History")

            history_df = pd.DataFrame(
                st.session_state.sim_history
            )

            st.dataframe(
                history_df,
                use_container_width=True,
                hide_index=True
            )

        # ----------------------------------------------------
        # LEARNED DQN PATH
        # ----------------------------------------------------
        st.divider()

        st.subheader("🧠 Learned DQN Path")

        st.write(
            "The trained DQN agent's greedy navigation path is shown below "
            "for comparison with the manual simulation."
        )

        learned_path = find_file("dqn_greedy_path.png")

        if learned_path is not None:
            st.image(
                learned_path,
                use_container_width=True
            )
        else:
            st.warning(
                "DQN path visualization is not available."
            )



# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <div style="font-size:18px;color:#d9b5ff;letter-spacing:7px;">✦ ❀ ✦ ❀ ✦</div>
        <b>DEEP REINFORCEMENT LEARNING • ROBOT PATH PLANNING</b><br>
        M.Sc. Artificial Intelligence • Mini Project • Academic Year 2026–2027
    </div>
    """,
    unsafe_allow_html=True,
)
