#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

# import the message type to use
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist


class Control(Node):
    def __init__(self) -> None:
				# initialize base class (must happen before everything else)
        super().__init__("control")
				
       
        # create publisher with: self.create_publisher(<msg type>, <topic>, <qos>)
        self.control_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        
        # create a timer with: self.create_timer(<second>, <callback>)
        self.control_timer = self.create_timer(0.2, self.control_message)
        
        self.kill_sub = self.create_subscription(Bool, '/kill', self.end_movement, 10)

    def control_message(self) -> None:
        """
        Heartbeat callback triggered by the timer
        """
        # construct heartbeat messagesending constant control..."
        msg = Twist()
        msg.linear.x = 10.0
        msg.angular.z = 10.0

        # publish heartbeat counter
        self.control_pub.publish(msg)
        
    def end_movement(self, msg:Bool) -> None:
        if msg.data:            
            self.destroy_timer(self.control_timer)
            
            msg = Twist()

            self.control_pub.publish(msg)
            
        
        


if __name__ == "__main__":
    rclpy.init()        # initialize ROS2 context (must run before any other rclpy call)
    node = Control()  # instantiate the heartbeat node
    rclpy.spin(node)    # Use ROS2 built-in schedular for executing the node
    rclpy.shutdown()    # cleanly shutdown ROS2 context