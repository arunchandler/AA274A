# Set up the 2D Ground Truth Map to generate the simulated data for the robot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class GroundTruthMap:
    """
    Ground truth map for the simulator to easily get range and bearing
    measurements, given a robot pose.
    """

    def __init__(self, landmarks):
        """
        Initialize the ground truth map with the given landmarks.
        """
        self.landmarks = np.array(landmarks)

    def get_range_bearing(self, robot_pose, max_sensor_range=4.0):
        """
        Get the range and bearing to each landmark from the robot pose.
        Vectorize with numpy
        """
        # Make sure the robot pose is a numpy array
        # pose is implied as (x, y, theta)
        robot_pose = np.array(robot_pose)

        # Calculate the deltas to each landmark
        # delta x, delta y
        deltas = self.landmarks - robot_pose[:2]

        # Calculate the range
        ranges = np.linalg.norm(deltas, axis=1)

        # Calculate the bearing
        # arctan2(y, x) - theta
        bearings = np.arctan2(deltas[:, 1], deltas[:, 0]) - robot_pose[2]

        # Wrap the bearings to [-pi, pi]
        # Modulo 2pi would be [0, 2pi], so offset by pi before and after
        bearings = (bearings + np.pi) % (2 * np.pi) - np.pi

        # Filter out landmarks that are too far away and replace with NaN
        invalid_landmarks = ranges > max_sensor_range
        ranges[invalid_landmarks] = np.nan
        bearings[invalid_landmarks] = np.nan

        return ranges, bearings


    def plot(self, fig=None, ax=None, show=False, block=False):
        """
        Plot the landmarks.
        """
        # If no plotting axes are provided, create a new figure and axes
        if fig is None or ax is None:
            fig, ax = plt.subplots()

        ax.scatter(*zip(*self.landmarks), c='r', marker='x')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title('Landmarks')

        # Ensure equal axes so box is square
        ax.axis('equal')

        # Subtle grid
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        if show:
            # Allow non blocking plots
            plt.show(block=block)

        return fig, ax


class StepWiseRobot:
    """
    Robot model that takes long-duration velocity and turn commands to update
    the pose.
    """
    def __init__(self,
                 initial_pose: np.ndarray = np.array([0, 0, 0]),
                 max_sensor_range: float = 2.75):
        """
        Initialize the robot with the given initial pose.
        """
        self.pose = np.array(initial_pose)
        self.pose_history = [self.pose]

        self.max_sensor_range = max_sensor_range
        self.measurement_history = []

        self.command_history = []

    def next_step(self, speed: float, delta_theta: float):
        """
        Take the speed and turn commands to update the pose.
        """
        # Unpack the pose
        x, y, theta = self.pose

        # Update the position
        x += speed * np.cos(theta)
        y += speed * np.sin(theta)

        # Update the angle at the END of the step
        theta += delta_theta

        # Wrap the angle to [-pi, pi]
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        # Pack the pose back up
        self.pose = np.array([x, y, theta])

        # Append the pose to the history
        self.pose_history.append(self.pose)

        # Append the command to the history
        self.command_history.append((speed, delta_theta))

    def get_sensor_data(self, ground_truth_map: GroundTruthMap):
        """
        Get the sensor data from the ground truth map.
        """
        ranges, bearings =  ground_truth_map.get_range_bearing(
            self.pose, self.max_sensor_range)

        measurements = np.stack([ranges, bearings], axis=1)

        self.measurement_history.append(measurements)

    def get_full_history(self):
        """
        Get the full history of the robot's pose and sensor data formatted
        together as:

        [time, x, y, theta, range_0, ... range_n, bearing_0, ... bearing_n]
        """
        # Stack the history to directly get the desired shape
        # (n_time_steps x (1 + 3 + 2*n_landmarks))

        # Time history is just the index
        t_history = np.arange(len(self.pose_history)).reshape(-1, 1)

        # Pose history is a list of numpy arrays
        # n_time_steps x 3 -> (n_time_steps x 3)
        pose_history_array = np.stack(self.pose_history, axis=0)

        # Measurement history is a list of numpy arrays ->
        # n_time_steps x (n_landmarks x 2) -> (n_time_steps x n_landmarks x 2)
        measurement_history_array = np.stack(self.measurement_history, axis=0)

        # Separate the ranges and bearings
        ranges_history_array = measurement_history_array[:, :, 0]
        bearings_history_array = measurement_history_array[:, :, 1]

        # Convert the bearings to degrees
        bearings_history_array = np.rad2deg(bearings_history_array)

        # Command history is a list of tuples
        # n_time_steps x 2 -> (n_time_steps x 2)
        command_history_array = np.array(self.command_history)

        # Add zeros at the last command to match the length of the pose history
        command_history_array = np.vstack([command_history_array, [0, 0]])

        # Concatenate the pose and measurement history
        history_array = np.concatenate([
            t_history,
            pose_history_array,
            command_history_array,
            ranges_history_array,
            bearings_history_array],
            axis=1)

        # print(f"Shape of history array: {history_array.shape}")
        # print(history_array)

        return history_array

    def save_history(self, file_path):
        """
        Save the pose and measurement history to a CSV file:

        [time, x, y, theta, range_0, ... range_n, bearing_0, ... bearing_n]
        """
        history_array = self.get_full_history()
        # history_array has shape (n_time_steps x (1 + 3 + 2 + 2*n_landmarks))

        n_separate = 1 + 3 + 2
        n_landmarks = (history_array.shape[1] - n_separate) // 2

        ranges_header = ','.join([f'range_{i}' for i in range(n_landmarks)])
        bearings_header = ','.join([f'bearing_{i}' for i in range(n_landmarks)])

        header = 'time,x,y,theta,speed,dtheta,' + ranges_header + ',' + bearings_header
        print(header)

        np.savetxt(file_path, history_array, delimiter=',',
                   header=header)


    def plot(self, fig=None, ax=None, show=False, block=False,
             plot_sensor_range=True):
        """
        Plot the robot's pose history.
        """
        # If no plotting axes are provided, create a new figure and axes
        if fig is None or ax is None:
            fig, ax = plt.subplots()

        # Plot the path
        # The pose is (x, y, theta), so plot (x, y) and orient the triangle
        # marker in the direction of theta

        # Plot each point separately to allow for different rotation of the
        # marker.
        # If there was a vectorized way to plot with different rotations, that
        # would be preferred.
        theta_offset = -90  # Offset for the triangle marker
        for x, y, theta in self.pose_history:
            ax.scatter(x, y, c='b',
                       marker=(3, 0, theta_offset + np.rad2deg(theta)))

            if plot_sensor_range:
                # Plot a circle with radius of sensor range for sanity
                # Dotted border to ease with visualization
                circle = patches.Circle((x, y), self.max_sensor_range,
                                        color='b', fill=False,
                                        linestyle=':', linewidth=0.5)

                ax.add_patch(circle)

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title('Robot Path')

        # Ensure equal axes so box is square
        ax.axis('equal')

        # Subtle grid
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        if show:
            # Allow non blocking plots
            plt.show(block=block)

        return fig, ax

    def plot_sensor_data(self, show=True, block=False):
        """
        Plot the sensor data accounting for the NaN values. Each line is one
        landmark. The x-axis is the index (i.e., time stamp) and the y-axis is
        the range (top subplot) or bearing (bottom subplot).
        """
        fig, axs = plt.subplots(2, 1, sharex=True, layout='constrained')

        # Measurment history is a list of numpy arrays (n_landmarks x 2)
        # Stack the history to directly get the desired shape
        # (n_landmarks x 2 x n_time_steps)
        measurement_history_array = np.stack(self.measurement_history, axis=2)
        assert measurement_history_array.shape[1] == 2

        time_index = np.arange(len(self.measurement_history))
        # print(time_index)

        for i, landmark_data in enumerate(measurement_history_array):
            ranges, bearings = landmark_data
            # print(f"Landmark {i} ranges: {ranges}")
            # print(f"Landmark {i} bearings: {bearings}")

            # Plot the range
            axs[0].plot(time_index, ranges, label=f'Landmark {i}', marker='o')

            # Plot the bearing
            axs[1].plot(time_index, np.rad2deg(bearings), marker='o')

        axs[0].set_ylabel('Range')
        axs[0].set_title('Sensor Data')
        axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)
        # Set the y-axis limits to 0 to max_sensor_range for the ranges
        axs[0].set_ylim(0, self.max_sensor_range)

        axs[1].set_ylabel('Bearing')
        axs[1].set_xlabel('Time Step')
        # Set the y-axis limits to -pi to pi for the bearings
        # axs[1].set_ylim(-np.pi, np.pi)
        axs[1].set_ylim(-180, 180)
        axs[1].set_yticks(np.linspace(-180, 180, 9))
        axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)

        plt.figlegend(loc='outside center right')
        # plt.tight_layout()

        if show:
            plt.show(block=block)

        return fig, axs
