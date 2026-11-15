import gymnasium as gym
from gymnasium import spaces
import numpy as np


class RobotPathPlanningEnv(gym.Env):
    """
    Custom Grid-Based Robot Path Planning Environment.

    Supports:
    - Fixed obstacle maps
    - Random obstacle maps
    - Guaranteed-solvable random maps
    - Reward shaping for easier DRL training
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        grid_size=20,
        max_steps=None,
        random_obstacles=False,
        obstacle_density=0.18,
        render_mode=None
    ):

        super().__init__()

        # =====================================================
        # BASIC PARAMETERS
        # =====================================================

        self.grid_size = grid_size

        if max_steps is None:
            self.max_steps = grid_size * grid_size
        else:
            self.max_steps = max_steps

        self.random_obstacles = random_obstacles
        self.obstacle_density = obstacle_density
        self.render_mode = render_mode

        # =====================================================
        # ACTION SPACE
        # =====================================================

        # 0 = Up
        # 1 = Down
        # 2 = Left
        # 3 = Right

        self.action_space = spaces.Discrete(4)

        # =====================================================
        # OBSERVATION SPACE
        # =====================================================

        observation_size = (
            grid_size * grid_size
            + 2
            + 2
            + 1
        )

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(observation_size,),
            dtype=np.float32
        )

        # =====================================================
        # START AND GOAL
        # =====================================================

        self.start_position = (0, 0)

        self.goal_position = (
            grid_size - 1,
            grid_size - 1
        )

        # =====================================================
        # EPISODE VARIABLES
        # =====================================================

        self.robot_position = self.start_position

        self.obstacles = set()

        self.path = []

        self.visited = set()

        self.steps = 0

        # =====================================================
        # INITIAL MAP
        # =====================================================

        if self.random_obstacles:

            self.obstacles = (
                self._create_random_obstacles()
            )

        else:

            self.obstacles = (
                self._create_obstacles()
            )

    # =========================================================
    # FIXED OBSTACLE MAP
    # =========================================================

    def _create_obstacles(self):

        obstacles = set()

        obstacle_coordinates = [

            # Row 2
            (2, 2), (2, 3), (2, 4), (2, 5),
            (2, 6), (2, 7), (2, 8), (2, 9),
            (2, 10), (2, 11), (2, 12), (2, 13),
            (2, 14), (2, 15), (2, 16), (2, 17),

            # Row 5
            (5, 0), (5, 1), (5, 2), (5, 3),
            (5, 4), (5, 5), (5, 6), (5, 7),
            (5, 8), (5, 9), (5, 10), (5, 11),
            (5, 12), (5, 13), (5, 14),

            # Row 8
            (8, 5), (8, 6), (8, 7), (8, 8),
            (8, 9), (8, 10), (8, 11), (8, 12),
            (8, 13), (8, 14), (8, 15), (8, 16),
            (8, 17), (8, 18), (8, 19),

            # Row 11
            (11, 0), (11, 1), (11, 2), (11, 3),
            (11, 4), (11, 5), (11, 6), (11, 7),
            (11, 8), (11, 9), (11, 10), (11, 11),
            (11, 12), (11, 13), (11, 14),

            # Row 14
            (14, 5), (14, 6), (14, 7), (14, 8),
            (14, 9), (14, 10), (14, 11), (14, 12),
            (14, 13), (14, 14), (14, 15), (14, 16),
            (14, 17), (14, 18), (14, 19),

            # Row 17
            (17, 0), (17, 1), (17, 2), (17, 3),
            (17, 4), (17, 5), (17, 6), (17, 7),
            (17, 8), (17, 9), (17, 10), (17, 11),
            (17, 12), (17, 13), (17, 14)
        ]

        for position in obstacle_coordinates:

            if position != self.start_position:
                if position != self.goal_position:

                    obstacles.add(position)

        return obstacles

    # =========================================================
    # RANDOM OBSTACLES
    # =========================================================

    def _create_random_obstacles(self):

        total_cells = (
            self.grid_size *
            self.grid_size
        )

        number_of_obstacles = int(
            total_cells *
            self.obstacle_density
        )

        # -----------------------------------------------------
        # Random solvable maps
        # -----------------------------------------------------

        for _ in range(300):

            obstacles = set()

            while len(obstacles) < number_of_obstacles:

                row = int(
                    self.np_random.integers(
                        0,
                        self.grid_size
                    )
                )

                col = int(
                    self.np_random.integers(
                        0,
                        self.grid_size
                    )
                )

                position = (row, col)

                if position == self.start_position:
                    continue

                if position == self.goal_position:
                    continue

                obstacles.add(position)

            if self._is_path_available(obstacles):

                return obstacles

        # -----------------------------------------------------
        # Fallback corridor
        # -----------------------------------------------------

        obstacles = set()

        guaranteed_path = set()

        # Top row
        for col in range(self.grid_size):

            guaranteed_path.add(
                (0, col)
            )

        # Rightmost column
        for row in range(self.grid_size):

            guaranteed_path.add(
                (row, self.grid_size - 1)
            )

        while len(obstacles) < number_of_obstacles:

            row = int(
                self.np_random.integers(
                    0,
                    self.grid_size
                )
            )

            col = int(
                self.np_random.integers(
                    0,
                    self.grid_size
                )
            )

            position = (row, col)

            if position in guaranteed_path:
                continue

            if position == self.start_position:
                continue

            if position == self.goal_position:
                continue

            obstacles.add(position)

        return obstacles

    # =========================================================
    # BFS SOLVABILITY CHECK
    # =========================================================

    def _is_path_available(self, obstacles):

        start = self.start_position
        goal = self.goal_position

        queue = [start]

        visited = {start}

        while queue:

            current = queue.pop(0)

            if current == goal:

                return True

            row, col = current

            neighbours = [

                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1)

            ]

            for neighbour in neighbours:

                n_row, n_col = neighbour

                if n_row < 0:
                    continue

                if n_row >= self.grid_size:
                    continue

                if n_col < 0:
                    continue

                if n_col >= self.grid_size:
                    continue

                if neighbour in obstacles:
                    continue

                if neighbour in visited:
                    continue

                visited.add(neighbour)

                queue.append(neighbour)

        return False

    # =========================================================
    # RESET
    # =========================================================

    def reset(
        self,
        seed=None,
        options=None
    ):

        super().reset(seed=seed)

        # -----------------------------------------------------
        # Create map
        # -----------------------------------------------------

        if self.random_obstacles:

            self.obstacles = (
                self._create_random_obstacles()
            )

        else:

            self.obstacles = (
                self._create_obstacles()
            )

        # -----------------------------------------------------
        # Robot
        # -----------------------------------------------------

        self.robot_position = (
            self.start_position
        )

        # -----------------------------------------------------
        # Counters
        # -----------------------------------------------------

        self.steps = 0

        self.path = [
            self.start_position
        ]

        self.visited = {
            self.start_position
        }

        # -----------------------------------------------------
        # Observation
        # -----------------------------------------------------

        observation = (
            self._get_observation()
        )

        # -----------------------------------------------------
        # Info
        # -----------------------------------------------------

        info = {

            "robot_position":
                self.robot_position,

            "goal_position":
                self.goal_position,

            "distance_to_goal":
                self._manhattan_distance(
                    self.robot_position,
                    self.goal_position
                ),

            "steps":
                0,

            "success":
                False,

            "collision":
                False,

            "path_length":
                1
        }

        return observation, info

    # =========================================================
    # STEP
    # =========================================================

    def step(self, action):

        self.steps += 1

        old_position = (
            self.robot_position
        )

        old_distance = (
            self._manhattan_distance(
                self.robot_position,
                self.goal_position
            )
        )

        row, col = (
            self.robot_position
        )

        # =====================================================
        # ACTION
        # =====================================================

        if action == 0:

            new_position = (
                row - 1,
                col
            )

        elif action == 1:

            new_position = (
                row + 1,
                col
            )

        elif action == 2:

            new_position = (
                row,
                col - 1
            )

        elif action == 3:

            new_position = (
                row,
                col + 1
            )

        else:

            new_position = (
                row,
                col
            )

        # =====================================================
        # DEFAULT
        # =====================================================

        reward = -0.10

        collision = False

        terminated = False

        truncated = False

        # =====================================================
        # BOUNDARY COLLISION
        # =====================================================

        if (
            new_position[0] < 0
            or
            new_position[0] >= self.grid_size
            or
            new_position[1] < 0
            or
            new_position[1] >= self.grid_size
        ):

            reward = -2.0

            collision = True

            new_position = (
                old_position
            )

        # =====================================================
        # OBSTACLE COLLISION
        # =====================================================

        elif new_position in self.obstacles:

            reward = -4.0

            collision = True

            new_position = (
                old_position
            )

        # =====================================================
        # VALID MOVEMENT
        # =====================================================

        else:

            self.robot_position = (
                new_position
            )

            new_distance = (
                self._manhattan_distance(
                    self.robot_position,
                    self.goal_position
                )
            )

            # -------------------------------------------------
            # Distance based reward
            # -------------------------------------------------

            distance_change = (
                old_distance
                - new_distance
            )

            reward += (
                distance_change
                * 1.5
            )

            # -------------------------------------------------
            # Additional reward for progress
            # -------------------------------------------------

            if distance_change > 0:

                reward += 0.5

            elif distance_change < 0:

                reward -= 0.3

        # =====================================================
        # UPDATE POSITION
        # =====================================================

        self.robot_position = (
            new_position
        )

        # =====================================================
        # REVISIT PENALTY
        # =====================================================

        if (
            self.robot_position
            in self.visited
        ):

            if self.steps > 1:

                reward -= 0.5

        else:

            self.visited.add(
                self.robot_position
            )

        # =====================================================
        # PATH
        # =====================================================

        self.path.append(
            self.robot_position
        )

        # =====================================================
        # GOAL
        # =====================================================

        if (
            self.robot_position
            ==
            self.goal_position
        ):

            reward = 50.0

            terminated = True

        # =====================================================
        # MAXIMUM STEPS
        # =====================================================

        if (
            self.steps >=
            self.max_steps
        ):

            if not terminated:

                truncated = True

        # =====================================================
        # OBSERVATION
        # =====================================================

        observation = (
            self._get_observation()
        )

        # =====================================================
        # INFO
        # =====================================================

        info = {

            "robot_position":
                self.robot_position,

            "goal_position":
                self.goal_position,

            "distance_to_goal":
                self._manhattan_distance(
                    self.robot_position,
                    self.goal_position
                ),

            "steps":
                self.steps,

            "collision":
                collision,

            "success":
                (
                    self.robot_position
                    ==
                    self.goal_position
                ),

            "path_length":
                len(self.path)
        }

        return (
            observation,
            reward,
            terminated,
            truncated,
            info
        )

    # =========================================================
    # MANHATTAN DISTANCE
    # =========================================================

    def _manhattan_distance(
        self,
        position_a,
        position_b
    ):

        return (
            abs(
                position_a[0]
                -
                position_b[0]
            )
            +
            abs(
                position_a[1]
                -
                position_b[1]
            )
        )

    # =========================================================
    # OBSERVATION
    # =========================================================

    def _get_observation(self):

        grid = np.zeros(
            (
                self.grid_size,
                self.grid_size
            ),
            dtype=np.float32
        )

        # -----------------------------------------------------
        # Obstacles
        # -----------------------------------------------------

        for obstacle in self.obstacles:

            row, col = obstacle

            grid[row, col] = 1.0

        # -----------------------------------------------------
        # Robot
        # -----------------------------------------------------

        robot_row, robot_col = (
            self.robot_position
        )

        grid[
            robot_row,
            robot_col
        ] = 2.0

        # -----------------------------------------------------
        # Goal
        # -----------------------------------------------------

        goal_row, goal_col = (
            self.goal_position
        )

        grid[
            goal_row,
            goal_col
        ] = 3.0

        # -----------------------------------------------------
        # Normalize
        # -----------------------------------------------------

        grid = grid / 3.0

        grid_flat = (
            grid.flatten()
        )

        # =====================================================
        # ROBOT COORDINATES
        # =====================================================

        robot_coordinates = np.array(
            [
                robot_row /
                (self.grid_size - 1),

                robot_col /
                (self.grid_size - 1)
            ],
            dtype=np.float32
        )

        # =====================================================
        # GOAL COORDINATES
        # =====================================================

        goal_coordinates = np.array(
            [
                goal_row /
                (self.grid_size - 1),

                goal_col /
                (self.grid_size - 1)
            ],
            dtype=np.float32
        )

        # =====================================================
        # NORMALIZED DISTANCE
        # =====================================================

        max_distance = (
            2 *
            (self.grid_size - 1)
        )

        distance = (
            self._manhattan_distance(
                self.robot_position,
                self.goal_position
            )
        )

        normalized_distance = np.array(
            [
                distance /
                max_distance
            ],
            dtype=np.float32
        )

        # =====================================================
        # FINAL OBSERVATION
        # =====================================================

        observation = np.concatenate(
            [
                grid_flat,
                robot_coordinates,
                goal_coordinates,
                normalized_distance
            ]
        ).astype(
            np.float32
        )

        return observation

    # =========================================================
    # RENDER
    # =========================================================

    def render(self):

        print()

        print(
            "=" * 60
        )

        for row in range(
            self.grid_size
        ):

            row_string = ""

            for col in range(
                self.grid_size
            ):

                position = (
                    row,
                    col
                )

                if (
                    position
                    ==
                    self.robot_position
                ):

                    row_string += " R "

                elif (
                    position
                    ==
                    self.goal_position
                ):

                    row_string += " G "

                elif (
                    position
                    in
                    self.obstacles
                ):

                    row_string += " # "

                else:

                    row_string += " . "

            print(row_string)

        print(
            "=" * 60
        )

        print(
            "Robot Position :",
            self.robot_position
        )

        print(
            "Goal Position  :",
            self.goal_position
        )

        print(
            "Distance       :",
            self._manhattan_distance(
                self.robot_position,
                self.goal_position
            )
        )

        print(
            "Steps          :",
            self.steps
        )

    # =========================================================
    # CLOSE
    # =========================================================

    def close(self):
        pass