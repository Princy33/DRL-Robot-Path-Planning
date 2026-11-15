import os
import sys
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

BACKEND_DIR = os.path.join(
    BASE_DIR,
    "backend"
)

sys.path.append(BACKEND_DIR)

from environment import RobotPathPlanningEnv


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Robot Path Planning Simulator",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if "env" not in st.session_state:

    st.session_state.env = RobotPathPlanningEnv(
        grid_size=20,
        max_steps=150
    )

    state, info = st.session_state.env.reset(
        seed=42
    )

    st.session_state.state = state
    st.session_state.total_reward = 0.0
    st.session_state.done = False
    st.session_state.last_reward = 0.0
    st.session_state.collision = False
    st.session_state.success = False


env = st.session_state.env


# ============================================================
# FUNCTIONS
# ============================================================

def reset_environment():

    state, info = env.reset(
        seed=42
    )

    st.session_state.state = state
    st.session_state.total_reward = 0.0
    st.session_state.done = False
    st.session_state.last_reward = 0.0
    st.session_state.collision = False
    st.session_state.success = False


def move_robot(action):

    if st.session_state.done:
        return

    (
        next_state,
        reward,
        terminated,
        truncated,
        info
    ) = env.step(action)

    st.session_state.state = next_state

    st.session_state.total_reward += reward

    st.session_state.last_reward = reward

    st.session_state.collision = info.get(
        "collision",
        False
    )

    st.session_state.success = info.get(
        "success",
        False
    )

    if terminated or truncated:

        st.session_state.done = True


def create_grid():

    grid = [
        [
            0 for _ in range(
                env.grid_size
            )
        ]
        for _ in range(
            env.grid_size
        )
    ]

    # Obstacles
    for obstacle in env.obstacles:

        row, col = obstacle

        grid[row][col] = 1

    return grid


# ============================================================
# HEADER
# ============================================================

st.title(
    "🤖 Interactive Robot Path Planning Simulator"
)

st.write(
    """
    Use the directional buttons to move the robot through
    the 20 × 20 environment while avoiding obstacles and
    reaching the target goal.
    """
)

st.divider()


# ============================================================
# MAIN LAYOUT
# ============================================================

left_col, right_col = st.columns(
    [2, 1]
)


# ============================================================
# GRID VISUALIZATION
# ============================================================

with left_col:

    st.subheader(
        "🗺️ Environment"
    )

    grid = create_grid()

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    ax.imshow(
        grid,
        cmap="Greys",
        origin="upper"
    )

    # --------------------------------------------------------
    # DRAW ROBOT PATH
    # --------------------------------------------------------

    if hasattr(
        env,
        "path"
    ):

        if len(env.path) > 1:

            rows = [
                p[0]
                for p in env.path
            ]

            cols = [
                p[1]
                for p in env.path
            ]

            ax.plot(
                cols,
                rows,
                linewidth=2,
                marker="o",
                markersize=3
            )

    # --------------------------------------------------------
    # START POSITION
    # --------------------------------------------------------

    start_row = 0
    start_col = 0

    ax.scatter(
        start_col,
        start_row,
        s=200,
        marker="s",
        label="Start"
    )

    # --------------------------------------------------------
    # GOAL POSITION
    # --------------------------------------------------------

    goal_row, goal_col = (
        env.goal_position
    )

    ax.scatter(
        goal_col,
        goal_row,
        s=250,
        marker="*",
        label="Goal"
    )

    # --------------------------------------------------------
    # ROBOT POSITION
    # --------------------------------------------------------

    robot_row, robot_col = (
        env.robot_position
    )

    ax.scatter(
        robot_col,
        robot_row,
        s=160,
        marker="o",
        label="Robot"
    )

    # --------------------------------------------------------
    # GRID SETTINGS
    # --------------------------------------------------------

    ax.set_xticks(
        range(env.grid_size)
    )

    ax.set_yticks(
        range(env.grid_size)
    )

    ax.set_xlabel(
        "Grid Column"
    )

    ax.set_ylabel(
        "Grid Row"
    )

    ax.set_title(
        "20 × 20 Robot Navigation Environment"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# ============================================================
# CONTROL PANEL
# ============================================================

with right_col:

    st.subheader(
        "🎮 Robot Controls"
    )

    st.write(
        "Move the robot:"
    )

    # --------------------------------------------------------
    # UP
    # --------------------------------------------------------

    empty1, up_col, empty2 = st.columns(
        [1, 1, 1]
    )

    with up_col:

        if st.button(
            "⬆️ UP",
            use_container_width=True
        ):

            move_robot(0)

            st.rerun()

    # --------------------------------------------------------
    # LEFT / DOWN / RIGHT
    # --------------------------------------------------------

    left_button, down_button, right_button = st.columns(
        3
    )

    with left_button:

        if st.button(
            "⬅️ LEFT",
            use_container_width=True
        ):

            move_robot(2)

            st.rerun()

    with down_button:

        if st.button(
            "⬇️ DOWN",
            use_container_width=True
        ):

            move_robot(1)

            st.rerun()

    with right_button:

        if st.button(
            "➡️ RIGHT",
            use_container_width=True
        ):

            move_robot(3)

            st.rerun()

    st.divider()

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    if st.button(
        "🔄 Reset Environment",
        use_container_width=True
    ):

        reset_environment()

        st.rerun()

    st.divider()

    # ========================================================
    # METRICS
    # ========================================================

    st.subheader(
        "📊 Current Performance"
    )

    metric1, metric2 = st.columns(2)

    with metric1:

        st.metric(
            "Steps",
            env.steps
        )

    with metric2:

        st.metric(
            "Last Reward",
            f"{st.session_state.last_reward:.2f}"
        )

    metric3, metric4 = st.columns(2)

    with metric3:

        st.metric(
            "Total Reward",
            f"{st.session_state.total_reward:.2f}"
        )

    with metric4:

        st.metric(
            "Path Length",
            len(env.path)
        )

    st.write(
        f"**Robot Position:** "
        f"{env.robot_position}"
    )

    st.write(
        f"**Goal Position:** "
        f"{env.goal_position}"
    )

    # ========================================================
    # STATUS
    # ========================================================

    if st.session_state.success:

        st.success(
            "🎯 Goal Reached Successfully!"
        )

    elif st.session_state.collision:

        st.error(
            "⚠️ Collision Detected!"
        )

    elif st.session_state.done:

        st.warning(
            "Episode Finished."
        )

    else:

        st.info(
            "🟢 Simulation Running"
        )


# ============================================================
# INFORMATION
# ============================================================

st.divider()

st.subheader(
    "ℹ️ Environment Information"
)

info_col1, info_col2, info_col3, info_col4 = st.columns(
    4
)

with info_col1:

    st.write(
        "**Grid Size**"
    )

    st.write(
        "20 × 20"
    )

with info_col2:

    st.write(
        "**Action Space**"
    )

    st.write(
        "4 Actions"
    )

with info_col3:

    st.write(
        "**State Space**"
    )

    st.write(
        "405 Features"
    )

with info_col4:

    st.write(
        "**Maximum Steps**"
    )

    st.write(
        "150"
    )


st.caption(
    "Deep Reinforcement Learning Based Robot Path Planning | "
    "Interactive Environment"
)