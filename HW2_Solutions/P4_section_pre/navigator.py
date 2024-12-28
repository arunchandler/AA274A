#!/usr/bin/env python3

import typing as T
import numpy as np
from scipy.interpolate import splev

import rclpy
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState
from asl_tb3_lib.grids import StochOccupancyGrid2D
from asl_tb3_lib.navigation import BaseNavigator, TrajectoryPlan
from asl_tb3_lib.math_utils import wrap_angle

from A_star import AStar, compute_smooth_plan


class Navigator(BaseNavigator):
    def __init__(self, kpx = 1.0, kpy = 1.0, kdx = 1.0, kdy = 1.0, kp = 2.0, V_PREV_THRESH=0.001):
        super().__init__("navigator")
        self.kp = kp
        self.V_PREV_THRES = V_PREV_THRESH
        self.kpx = kpx
        self.kpy = kpy
        self.kdx = kdx
        self.kdy = kdy

    def reset(self) -> None:
        self.V_prev = 0.
        self.om_prev = 0.
        self.t_prev = 0.

    def compute_heading_control(self, state: TurtleBotState, goal: TurtleBotState) -> TurtleBotControl:
        heading_error = goal.theta - state.theta
        heading_error = wrap_angle(heading_error)

        # using the proportional control formula
        omega_value = self.kp * heading_error
        new_message = TurtleBotControl()
        new_message.omega = omega_value

        return new_message
    
    def compute_trajectory_tracking_control(self, state: TurtleBotState, plan: TrajectoryPlan, t: float) -> TurtleBotControl:

        path_x_spline = plan.path_x_spline
        path_y_spline = plan.path_y_spline

        dt = t - self.t_prev
        x_d = float(splev(t, path_x_spline, der = 0))
        xd_d = float(splev(t, path_x_spline, der = 1))
        xdd_d = float(splev(t, path_x_spline, der = 2))
        y_d = float(splev(t, path_y_spline, der = 0))
        yd_d = float(splev(t, path_y_spline, der = 1))
        ydd_d = float(splev(t, path_y_spline, der = 2))
       
        if abs(self.V_prev)<self.V_PREV_THRES:
            self.V_prev = self.V_PREV_THRES
        v = self.V_prev

        th = state.theta

        Vx = v*np.cos(th)
        Vy = v*np.sin(th)

        J = np.array([[np.cos(th), -Vy], 
                      [np.sin(th), Vx]])
        
        virtual_control = np.zeros((2,))
        virtual_control[0] = xdd_d + self.kdx*(xd_d-Vx) + self.kpx*(x_d-state.x)
        virtual_control[1] = ydd_d + self.kdy*(yd_d-Vy) + self.kpy*(y_d-state.y)
        controls = np.linalg.solve(J,  virtual_control)

        acceleration = controls[0]
        v_new = acceleration*dt + v

        if abs(v_new)<self.V_PREV_THRES:
            V = self.V_PREV_THRES
        else:
            V = v_new
        om = controls[1]

        # save the commands that were applied and the time
        self.t_prev = t
        self.V_prev = V
        self.om_prev = om

        turtle_bot_control = TurtleBotControl()
        turtle_bot_control.v = V
        turtle_bot_control.omega = om

        return turtle_bot_control

    
    def compute_trajectory_plan(self, state: TurtleBotState, goal: TurtleBotState, occupancy: StochOccupancyGrid2D, resolution: float, horizon: float) -> TrajectoryPlan | None:
        grid_xy_lower = (state.x - horizon, state.y - horizon)
        grid_xy_upper = (state.x + horizon, state.y + horizon)
        init_state = (state.x, state.y)
        goal_state = (goal.x, goal.y)
        astar = AStar(grid_xy_lower, grid_xy_upper, init_state, goal_state, occupancy, resolution)

        if not astar.solve():
            return None
        
        astar_path = astar.path
        if len(astar_path) < 4:
            return None

        self.reset()
        smooth_traj = compute_smooth_plan(astar_path)

        return smooth_traj



if __name__ == "__main__":
    rclpy.init()
    navigator = Navigator()
    rclpy.spin(navigator)
    rclpy.shutdown()
