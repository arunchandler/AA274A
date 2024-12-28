#!/usr/bin/env python3
import scipy.interpolate
from asl_tb3_lib.grids import StochOccupancyGrid2D
import rclpy
import typing as T
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# from utils import plot_line_segments
from asl_tb3_lib.navigation import BaseNavigator, TrajectoryPlan
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_lib.tf_utils import quaternion_to_yaw
from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState

class Navigator(BaseNavigator):
    def __init__(self, kpx: float, kpy: float, kdx: float, kdy: float):
        super().__init__()
        self.kp = 2.0
        self.V_PREV_THRES = 0.0001

        self.t_prev = 0.0
        self.V_prev = 0.0

        # self.kpx = kpx
        # self.kpy = kpy
        # self.kdx = kdx
        # self.kdy = kdy
        self.kpx = 2.0
        self.kpy = 2.0
        self.kdx = 2.0
        self.kdy = 2.0
        self.get_logger().info("Navigator Initialized")

    def compute_heading_control(self, s_cur: TurtleBotState, s_des: TurtleBotState)->TurtleBotControl:
        err = wrap_angle(s_des.theta - s_cur.theta)
        om = self.kp*err
        control_message = TurtleBotControl()
        control_message.omega = om
        return control_message
    
    def compute_trajectory_tracking_control(self, state: TurtleBotState, plan: TrajectoryPlan, t: float) -> TurtleBotControl:

        dt = t - self.t_prev
        x_d = scipy.interpolate.splev(t, plan.path_x_spline, 0)
        y_d = scipy.interpolate.splev(t, plan.path_y_spline, 0)
        xd_d = scipy.interpolate.splev(t, plan.path_x_spline, 1)
        yd_d = scipy.interpolate.splev(t, plan.path_y_spline, 1)
        xdd_d = scipy.interpolate.splev(t, plan.path_x_spline, 2)
        ydd_d = scipy.interpolate.splev(t, plan.path_y_spline, 2)

        u1 = xdd_d + self.kpx*(x_d-state.x) + self.kdx*(xd_d-self.V_prev*np.cos(state.theta))
        u2 = ydd_d + self.kpy*(y_d-state.y) + self.kdy*(yd_d-self.V_prev*np.sin(state.theta))
        V = self.V_prev+ ((u1*np.cos(state.theta))+(u2*np.sin(state.theta)))*dt
        if V<self.V_PREV_THRES: V = self.V_PREV_THRES
        om = (u2*np.cos(state.theta)-u1*np.sin(state.theta))/V

        self.t_prev = t
        self.V_prev = V
        self.om_prev = om

        ctrl = TurtleBotControl()
        ctrl.v = V
        ctrl.omega = om

        return ctrl
    
    def compute_trajectory_plan(self, state: TurtleBotState, goal: TurtleBotState, occupancy: StochOccupancyGrid2D, resolution: float, horizon: float) -> TrajectoryPlan | None:
        v_desired = 0.15
        spline_alpha = 0.15
        self.get_logger().info("Beginning AStar")
        astar = AStar((state.x-horizon, state.y-horizon), (state.x+horizon, state.y+horizon), [state.x,state.y], [goal.x,goal.y], occupancy, resolution)
        if not astar.solve() or len(astar.path)<4:
            self.get_logger().info("No Path Found")
            return None

        self.get_logger().info("Path Found!")
        path = np.asarray(astar.path)

        self.V_prev = self.V_PREV_THRES
        self.t_prev = 0

        ts = np.array([0])
        tot_time = 0
        for index in range(1,len(path)):
            dist = np.sqrt((path[index][1]-path[index-1][1])**2+(path[index][0]-path[index-1][0])**2)
            temp_time = dist/v_desired
            tot_time += temp_time
            ts = np.append(ts, tot_time)

        path_x_spline = scipy.interpolate.splrep(ts, path[:,0], s=spline_alpha)
        path_y_spline = scipy.interpolate.splrep(ts, path[:,1], s=spline_alpha)
        self.get_logger().info("Splines Calculated")
        return TrajectoryPlan(
            path=path,
            path_x_spline=path_x_spline,
            path_y_spline=path_y_spline,
            duration=ts[-1],
        )        

class AStar(object):
    """Represents a motion planning problem to be solved using A*"""

    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy, resolution=1):
        self.statespace_lo = statespace_lo         # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = statespace_hi         # state space upper bound (e.g., [5, 5])
        self.occupancy = occupancy                 # occupancy grid (a DetOccupancyGrid2D object)
        self.resolution = resolution               # resolution of the discretization of state space (cell/m)
        self.x_offset = x_init                     
        self.x_init = self.snap_to_grid(x_init)    # initial state
        self.x_goal = self.snap_to_grid(x_goal)    # goal state

        self.closed_set = set()    # the set containing the states that have been visited
        self.open_set = set()      # the set containing the states that are condidate for future expension

        self.est_cost_through = {}  # dictionary of the estimated cost from start to goal passing through state (often called f score)
        self.cost_to_arrive = {}    # dictionary of the cost-to-arrive at state from start (often called g score)
        self.came_from = {}         # dictionary keeping track of each state's parent to reconstruct the path

        self.open_set.add(self.x_init)
        self.cost_to_arrive[self.x_init] = 0
        self.est_cost_through[self.x_init] = self.distance(self.x_init,self.x_goal)

        self.path = None        # the final path as a list of states

    def is_free(self, x):
        """
        Checks if a give state x is free, meaning it is inside the bounds of the map and
        is not inside any obstacle.
        Inputs:
            x: state tuple
        Output:
            Boolean True/False
        Hint: self.occupancy is a DetOccupancyGrid2D object, take a look at its methods for what might be
              useful here
        """
        ########## Code starts here ##########
        if x[0]<=self.statespace_lo[0] or x[1]<=self.statespace_lo[1] or x[0]>=self.statespace_hi[0] or x[1]>=self.statespace_hi[1]:
            return False
        elif self.occupancy.is_free(np.asarray(x)) == True:
            return True
        else:
            return False
        ########## Code ends here ##########

    def distance(self, x1, x2):
        """
        Computes the Euclidean distance between two states.
        Inputs:
            x1: First state tuple
            x2: Second state tuple
        Output:
            Float Euclidean distance

        HINT: This should take one line. Tuples can be converted to numpy arrays using np.array().
        """
        ########## Code starts here ##########
        return np.sqrt((x2[0]-x1[0])**2+(x2[1]-x1[1])**2)
        ########## Code ends here ##########

    def snap_to_grid(self, x):
        """ Returns the closest point on a discrete state grid
        Input:
            x: tuple state
        Output:
            A tuple that represents the closest point to x on the discrete state grid
        """
        return (
            self.resolution * round((x[0] - self.x_offset[0]) / self.resolution) + self.x_offset[0],
            self.resolution * round((x[1] - self.x_offset[1]) / self.resolution) + self.x_offset[1],
        )

    def get_neighbors(self, x):
        """
        Gets the FREE neighbor states of a given state x. Assumes a motion model
        where we can move up, down, left, right, or along the diagonals by an
        amount equal to self.resolution.
        Input:
            x: tuple state
        Ouput:
            List of neighbors that are free, as a list of TUPLES

        HINTS: Use self.is_free to check whether a given state is indeed free.
               Use self.snap_to_grid (see above) to ensure that the neighbors
               you compute are actually on the discrete grid, i.e., if you were
               to compute neighbors by adding/subtracting self.resolution from x,
               numerical errors could creep in over the course of many additions
               and cause grid point equality checks to fail. To remedy this, you
               should make sure that every neighbor is snapped to the grid as it
               is computed.
        """
        neighbors = []
        ########## Code starts here ##########
        if self.is_free((x[0],x[1]+self.resolution)):
            neighbors.append(self.snap_to_grid((x[0],x[1]+self.resolution)))
        if self.is_free((x[0]+self.resolution,x[1]+self.resolution)):
            neighbors.append(self.snap_to_grid((x[0]+self.resolution,x[1]+self.resolution)))
        if self.is_free((x[0]+self.resolution,x[1])):
            neighbors.append(self.snap_to_grid((x[0]+self.resolution,x[1])))
        if self.is_free((x[0]+self.resolution,x[1]-self.resolution)):
            neighbors.append(self.snap_to_grid((x[0]+self.resolution,x[1]-self.resolution)))
        if self.is_free((x[0],x[1]-self.resolution)):
            neighbors.append(self.snap_to_grid((x[0],x[1]-self.resolution)))
        if self.is_free((x[0]-self.resolution,x[1]-self.resolution)):
            neighbors.append(self.snap_to_grid((x[0]-self.resolution,x[1]-self.resolution)))
        if self.is_free((x[0]-self.resolution,x[1])):
            neighbors.append(self.snap_to_grid((x[0]-self.resolution,x[1])))
        if self.is_free((x[0]-self.resolution,x[1]+self.resolution)):
            neighbors.append(self.snap_to_grid((x[0]-self.resolution,x[1]+self.resolution)))
        ########## Code ends here ##########
        return neighbors

    def find_best_est_cost_through(self):
        """
        Gets the state in open_set that has the lowest est_cost_through
        Output: A tuple, the state found in open_set that has the lowest est_cost_through
        """
        return min(self.open_set, key=lambda x: self.est_cost_through[x])

    def reconstruct_path(self):
        """
        Use the came_from map to reconstruct a path from the initial location to
        the goal location
        Output:
            A list of tuples, which is a list of the states that go from start to goal
        """
        path = [self.x_goal]
        current = path[-1]
        while current != self.x_init:
            path.append(self.came_from[current])
            current = path[-1]
        return list(reversed(path))

    def plot_path(self, fig_num=0, show_init_label=True):
        """Plots the path found in self.path and the obstacles"""
        if not self.path:
            return

        self.occupancy.plot(fig_num)

        solution_path = np.asarray(self.path)
        plt.plot(solution_path[:,0],solution_path[:,1], color="green", linewidth=2, label="A* solution path", zorder=10)
        plt.scatter([self.x_init[0], self.x_goal[0]], [self.x_init[1], self.x_goal[1]], color="green", s=30, zorder=10)
        if show_init_label:
            plt.annotate(r"$x_{init}$", np.array(self.x_init) + np.array([.2, .2]), fontsize=16)
        plt.annotate(r"$x_{goal}$", np.array(self.x_goal) + np.array([.2, .2]), fontsize=16)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)

        plt.axis([0, self.occupancy.width, 0, self.occupancy.height])

    def solve(self):
        """
        Solves the planning problem using the A* search algorithm. It places
        the solution as a list of tuples (each representing a state) that go
        from self.x_init to self.x_goal inside the variable self.path
        Input:
            None
        Output:
            Boolean, True if a solution from x_init to x_goal was found

        HINTS:  We're representing the open and closed sets using python's built-in
                set() class. This allows easily adding and removing items using
                .add(item) and .remove(item) respectively, as well as checking for
                set membership efficiently using the syntax "if item in set".
        """
        ########## Code starts here ##########
        self.open_set.clear()
        self.open_set.add(self.x_init)
        self.closed_set.clear()
        self.cost_to_arrive[self.x_init] = 0
        self.est_cost_through[self.x_init] = self.distance(self.x_init,self.x_goal)
        while len(self.open_set)>0:
            x_cur = self.find_best_est_cost_through()
            if x_cur == self.x_goal:
                self.path = self.reconstruct_path()
                return True
            self.open_set.remove(x_cur)
            self.closed_set.add(x_cur)
            for x_neigh in self.get_neighbors(x_cur):
                if x_neigh in self.closed_set:
                    continue
                tentative_CTA = self.cost_to_arrive[x_cur]+self.distance(x_cur,x_neigh)
                if x_neigh not in self.open_set:
                    self.open_set.add(x_neigh)
                elif tentative_CTA > self.cost_to_arrive[x_neigh]:
                    continue
                self.came_from[x_neigh] = x_cur
                self.cost_to_arrive[x_neigh] = tentative_CTA
                self.est_cost_through[x_neigh] = tentative_CTA + self.distance(x_neigh,self.x_goal)
        return False
        ########## Code ends here ##########

class DetOccupancyGrid2D(object):
    """
    A 2D state space grid with a set of rectangular obstacles. The grid is
    fully deterministic
    """
    def __init__(self, width, height, obstacles):
        self.width = width
        self.height = height
        self.obstacles = obstacles

    def is_free(self, x):
        """Verifies that point is not inside any obstacles by some margin"""
        for obs in self.obstacles:
            if x[0] >= obs[0][0] - self.width * .01 and \
               x[0] <= obs[1][0] + self.width * .01 and \
               x[1] >= obs[0][1] - self.height * .01 and \
               x[1] <= obs[1][1] + self.height * .01:
                return False
        return True

    def plot(self, fig_num=0):
        """Plots the space and its obstacles"""
        fig = plt.figure(fig_num)
        ax = fig.add_subplot(111, aspect='equal')
        for obs in self.obstacles:
            ax.add_patch(
            patches.Rectangle(
            obs[0],
            obs[1][0]-obs[0][0],
            obs[1][1]-obs[0][1],))
        ax.set(xlim=(0,self.width), ylim=(0,self.height))
        
if __name__ == "__main__":
    print("Running Main")
    rclpy.init()
    ctrl = Navigator(2.0, 2.0, 2.0, 2.0)
    rclpy.spin(ctrl)
    rclpy.shutdown()
