#!/usr/bin/env python3
import sys, tty, termios, select, threading
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

LIN_STEP = 0.05
ANG_STEP = 0.02

# 12 keys: increase/decrease for linear x,y,z and angular x,y,z
# Mapping: key -> (type, axis_index, sign)
# type: 'lin' or 'ang'; axis_index: 0=x,1=y,2=z; sign: +1 or -1
KEY_MAP = {
  'i': ('lin', 0, +1),  # +lin x
  'k': ('lin', 0, -1),  # -lin x
  'j': ('lin', 1, +1),  # +lin y
  'l': ('lin', 1, -1),  # -lin y
  'u': ('lin', 2, +1),  # +lin z
  'o': ('lin', 2, -1),  # -lin z
  'q': ('ang', 0, +1),  # +ang x (roll)
  'e': ('ang', 0, -1),  # -ang x
  'z': ('ang', 1, +1),  # +ang y (pitch)
  'x': ('ang', 1, -1),  # -ang y
  'c': ('ang', 2, +1),  # +ang z (yaw)
  'v': ('ang', 2, -1),  # -ang z
  ' ': ('stop', 0, 0),  # optional: publish zero
}

class Teleop(Node):
  def __init__(self):
    super().__init__('ik_teleop')
    self.pub = self.create_publisher(Twist, '/ik_delta', 10)

  def publish_once(self, msg: Twist):
    # one-shot publish
    self.pub.publish(msg)


def key_reader(node: Teleop):
  fd = sys.stdin.fileno()
  old = termios.tcgetattr(fd)
  tty.setcbreak(fd)
  try:
    print("Keys: i/k +x/-x, j/l +y/-y, u/o +z/-z, q/e +roll/-roll, z/x +pitch/-pitch, c/v +yaw/-yaw, space=stop, Ctrl-C exit")
    while True:
      if select.select([sys.stdin], [], [], 0.1)[0]:
        c = sys.stdin.read(1)
        if c == '\x03':
          break
        if c in KEY_MAP:
          entry = KEY_MAP[c]
          if entry[0] == 'stop':
            msg = Twist()  # zero
            node.publish_once(msg)
            print('published: stop')
            continue
          typ, axis, sign = entry
          msg = Twist()
          if typ == 'lin':
            val = sign * LIN_STEP
            if axis == 0:
              msg.linear.x = val
            elif axis == 1:
              msg.linear.y = val
            else:
              msg.linear.z = val
          else:
            val = sign * ANG_STEP
            if axis == 0:
              msg.angular.x = val
            elif axis == 1:
              msg.angular.y = val
            else:
              msg.angular.z = val
          node.publish_once(msg)
          print(f'published: {c} -> {msg}')
  finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main():
  rclpy.init()
  node = Teleop()
  th = threading.Thread(target=key_reader, args=(node,), daemon=True)
  th.start()
  try:
    rclpy.spin(node)
  except KeyboardInterrupt:
    pass
  node.destroy_node()
  rclpy.shutdown()

if __name__ == '__main__':
  main()