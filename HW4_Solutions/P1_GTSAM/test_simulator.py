"""
Use the simulator code and plot an example.

Author: Daniel Neamati
"""

import simulator
import numpy as np

# Set up the 2D Ground Truth Map to generate the simulated data for the robot
# Make 10 landmarks to mimic the corners of a building floor interior
# with a hallway wrapping around two office rooms.
landmarks = [
    (-1, -1),
    (-1, 1),
    (5, 1),
    (7, 1),
    (10, 1),
    (12, -1),
    (12, 6),
    (10, 4),
    (7, 4),
    (5, 4),
    (-1, 4),
    (-1, 6)
]
print(landmarks)
# landmarks = [(0, 0), (1, 0), (0, 1), (1, 1)]
ground_truth_map = simulator.GroundTruthMap(landmarks)

# Plot the landmarks
fig, ax = ground_truth_map.plot(block=False)

# Make a robot that goes left, up, then right through the building in 10 steps
# Initialize at origin
max_range=3
robot = simulator.StepWiseRobot(initial_pose=np.array([0, 0, 0]),
                                max_sensor_range=max_range)

# Get initial sensor data
robot.get_sensor_data(ground_truth_map)

# 1D speed and
commands = [(3, 0) for _ in range(3)] + \
            [(3, np.pi/2)] + \
            [(2.5, 0) for _ in range(1)] + \
            [(2.5, np.pi/2)] + \
            [(3, 0) for _ in range(4)]

# Simulate the robot moving through the building
for command in commands:
    robot.next_step(*command)
    robot.get_sensor_data(ground_truth_map)

# Check same length
print(f"Length of pose history: {len(robot.pose_history)}")
print(f"Length of measurement history: {len(robot.measurement_history)}")
print(f"Length of command history: {len(robot.command_history)}")
assert len(robot.pose_history) == len(robot.measurement_history)
assert len(robot.pose_history) == (len(robot.command_history) + 1)

# # Print the pose history
# for p, y in zip(robot.pose_history, robot.measurement_history):
#     print(f"At pose {p}")
#     print("Received the following sensor data:")
#     print(y)

fig, ax = robot.plot(fig=fig, ax=ax, show=False, block=True)
fig.savefig(f"GTSAM/robot_path_{max_range}.png", dpi=300)

# Separately plot the sensor data
fig, ax = robot.plot_sensor_data(show=False, block=True)
fig.savefig(f"GTSAM/sensor_data_{max_range}.png", dpi=300)


# Save the history to a CSV file
robot.save_history(f"GTSAM/robot_history_{max_range}.csv")
