#!/usr/bin/env python3
import sys, tty, termios, select, threading
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

LIN_STEP = 0.05
ANG_STEP = 0.02

KEY_MAP = {
  'i': ('lin',  LIN_STEP, 0.0),
  'k': ('lin', -LIN_STEP, 0.0),
  'j': ('lin', 0.0,  LIN_STEP),
  'l': ('lin', 0.0, -LIN_STEP),
  'q': ('ang',  ANG_STEP, 0.0),
  'e': ('ang', -ANG_STEP, 0.0),
  ' ': ('stop', 0.0, 0.0),
}

class Teleop(Node):
  def __init__(self):
    super().__init__('ik_teleop')
    self.pub = self.create_publisher(Twist, '/ik_delta', 10)
    self.cmd = Twist()
    self.create_timer(0.05, self._publish)

  def _publish(self):
    self.pub.publish(self.cmd)

def key_reader(node):
  fd = sys.stdin.fileno()
  old = termios.tcgetattr(fd)
  tty.setcbreak(fd)
  try:
    print("Keys: i/k forward/back, j/l left/right, q/e yaw, space stop, Ctrl-C exit")
    while True:
      if select.select([sys.stdin], [], [], 0.1)[0]:
        c = sys.stdin.read(1)
        if c == '\x03': break
        if c in KEY_MAP:
          t = KEY_MAP[c]
          if t[0] == 'lin':
            node.cmd.linear.x = t[1]; node.cmd.linear.y = t[2]
          elif t[0] == 'ang':
            node.cmd.angular.z = t[1]
          else:
            node.cmd = Twist()
  finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old)

def main():
  rclpy.init()
  node = Teleop()
  th = threading.Thread(target=key_reader, args=(node,), daemon=True)
  th.start()
  try: rclpy.spin(node)
  except KeyboardInterrupt: pass
  node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()