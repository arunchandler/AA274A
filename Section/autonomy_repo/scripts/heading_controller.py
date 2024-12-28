#!/usr/bin/env python3

import numpy, rclpy
from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState

class HeadingController(BaseHeadingController):
    def __init__(self):
        BaseHeadingController.__init__(self)
        #self.kp = 2.0
        self.declare_parameter('kp', 2.0)
        
    @property
    def kp(self)->float:
        return self.get_parameter('kp').value
        
    def compute_control_with_goal(self, current_state: TurtleBotState, goal_state: TurtleBotState) -> TurtleBotControl:
        heading_error = goal_state.theta - current_state.theta
        heading_error = wrap_angle(heading_error)
        omega_p = self.kp * heading_error
        return TurtleBotControl(omega= omega_p)
    
if __name__ == "__main__":
    rclpy.init()
    ctrl = HeadingController()
    rclpy.spin(ctrl)
    rclpy.shutdown()