from environment import RobotPathPlanningEnv
import time


def main():

    # Create the environment
    env = RobotPathPlanningEnv(
        grid_size=20,
        max_steps=100,
        render_mode="human"
    )

    # Reset environment
    state, info = env.reset(seed=42)

    print("=" * 50)
    print("DRL ROBOT PATH PLANNING - ENVIRONMENT TEST")
    print("=" * 50)

    print("\nEnvironment Information:")
    print("Grid Size       :", env.grid_size, "x", env.grid_size)
    print("Start Position  :", env.start_position)
    print("Goal Position   :", env.goal_position)
    print("Number of       :", len(env.obstacles), "obstacles")
    print("Action Space    :", env.action_space)
    print("State Shape     :", state.shape)

    print("\nInitial State Information:")
    print("Robot Position  :", env.robot_position)
    print("Distance to Goal:", info["distance_to_goal"])

    print("\nStarting random movement test...")
    print("-" * 50)

    # Test 20 random actions
    for step in range(20):

        # Select a random action
        action = env.action_space.sample()

        # Perform action
        state, reward, terminated, truncated, info = env.step(action)

        print(
            f"Step: {step + 1:02d} | "
            f"Action: {action} | "
            f"Position: {env.robot_position} | "
            f"Reward: {reward:6.2f} | "
            f"Distance: {info['distance_to_goal']}"
        )

        # Display environment
        env.render()

        time.sleep(0.1)

        # Stop if episode ends
        if terminated or truncated:

            print("\nEpisode ended.")

            if info["success"]:
                print("SUCCESS: Robot reached the goal!")

            elif info["collision"]:
                print("Collision detected.")

            else:
                print("Maximum steps reached.")

            break

    print("\nTest completed successfully!")

    env.close()


if __name__ == "__main__":
    main()
    