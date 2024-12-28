#!/usr/bin/env python3

import numpy, rclpy
from asl_tb3_lib.control import BaseController
from asl_tb3_msgs.msg import TurtleBotControl
from std_msgs.msg import Bool

class PerceptionController(BaseController):
    def __init__(self):
        BaseController.__init__(self, "PerceptionNode")
        self.declare_parameter('active', True)
        self.last_time = self.get_clock().now().nanoseconds / 1e9
        self.create_subscription(Bool, "/detector_bool", self.stop_callback, 10)
        self.previous_reading = False
        
    @property
    def active(self)->bool:
        return self.get_parameter('active').value
    True
    def stop_callback(self, msg: Bool) -> None:
        if msg.data and not self.previous_reading:
            self.set_parameters([rclpy.Parameter('active', value=False)])
        self.previous_reading = msg.data
        
    def compute_control(self) -> TurtleBotControl:
        
        
        if self.active:
            omega_bot = 0.5
            self.last_time = self.get_clock().now().nanoseconds / 1e9
        else:
            if self.get_clock().now().nanoseconds / 1e9 - self.last_time > 5:
                self.set_parameters([rclpy.Parameter('active', value=True)])
                omega_bot = 0.5
            else:
                omega_bot = 0.0
            
        return TurtleBotControl(omega= omega_bot)
    
if __name__ == "__main__":
    rclpy.init()
    ctrl = PerceptionController()
    rclpy.spin(ctrl)
    rclpy.shutdown()
    
    