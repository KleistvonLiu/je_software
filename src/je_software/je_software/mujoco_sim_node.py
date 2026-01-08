#!/usr/bin/env python3
"""ROS2 node to run a lightweight MuJoCo visualization simulation and bridge data for IK.

- Loads a MuJoCo XML model (configurable)
- Runs a simulation loop at control_frequency
- Publishes `/sim/joint_states` (sensor_msgs/JointState)
- Subscribes to `/joint_command` (sensor_msgs/JointState) to accept position targets

This file is intentionally standalone and uses the MuJoCo python bindings directly
so it does not require the project follower class to be importable.
"""

import sys
import time
import threading
from typing import List, Dict

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

import datetime
import json

try:
    import mujoco
except Exception as e:
    mujoco = None

# try to import mujoco.viewer and glfw (zongzhuang implementation imports these explicitly)
try:
    import mujoco.viewer as mujoco_viewer
except Exception:
    mujoco_viewer = None

try:
    import glfw
except Exception:
    glfw = None


class MujocoSimNode(Node):
    def __init__(self):
        super().__init__('mujoco_sim_node')

        # Parameters
        self.declare_parameter('model_xml', '/home/agx/price/issac-sim/zongzhuang2/meshes/zongzhuang2.xml')
        self.declare_parameter('joint_names', ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7'])
        self.declare_parameter('control_frequency', 50.0)
        self.declare_parameter('use_viewer', False)
        # control_mode: 'actuator' (default) uses actuator.ctrl; 'direct' writes qpos directly
        self.declare_parameter('control_mode', 'direct')
        # topic used to publish simulated joint states (make configurable so it can publish to /joint_states)
        self.declare_parameter('state_topic', '/joint_states')

        # --- Logging: enable saving received joint_command messages to file
        self.declare_parameter('log_joint_commands', False)
        self.declare_parameter('joint_command_log_file', '/home/agx/price/je_software/log/mujoco_joint_commands.log')

        # --- Joint states periodic logging (for observing drift when no commands)
        self.declare_parameter('log_joint_states', False)
        self.declare_parameter('joint_states_log_file', '/home/agx/price/je_software/log/mujoco_joint_states.log')
        self.declare_parameter('joint_states_log_frequency', 1.0)

        model_xml = self.get_parameter('model_xml').get_parameter_value().string_value
        self.joint_names: List[str] = list(self.get_parameter('joint_names').get_parameter_value().string_array_value)
        self.control_frequency = float(self.get_parameter('control_frequency').get_parameter_value().double_value)
        self.use_viewer = bool(self.get_parameter('use_viewer').get_parameter_value().bool_value)
        self.control_mode = str(self.get_parameter('control_mode').get_parameter_value().string_value)
        # read the state topic parameter
        self.state_topic = self.get_parameter('state_topic').get_parameter_value().string_value

        # read logging parameters
        try:
            self.log_joint_commands = bool(self.get_parameter('log_joint_commands').get_parameter_value().bool_value)
        except Exception:
            self.log_joint_commands = True
        try:
            self.joint_command_log_file = self.get_parameter('joint_command_log_file').get_parameter_value().string_value
        except Exception:
            self.joint_command_log_file = '/tmp/mujoco_joint_commands.log'

        # read joint_states logging parameters
        try:
            self.log_joint_states = bool(self.get_parameter('log_joint_states').get_parameter_value().bool_value)
        except Exception:
            self.log_joint_states = False
        try:
            self.joint_states_log_file = self.get_parameter('joint_states_log_file').get_parameter_value().string_value
        except Exception:
            self.joint_states_log_file = '/tmp/mujoco_joint_states.log'
        try:
            self.joint_states_log_frequency = float(self.get_parameter('joint_states_log_frequency').get_parameter_value().double_value)
        except Exception:
            self.joint_states_log_frequency = 1.0

        if mujoco is None:
            self.get_logger().error('mujoco python bindings not available. Install mujoco or adjust PYTHONPATH.')
            raise RuntimeError('mujoco missing')

        self.get_logger().info(f'Loading MuJoCo model from: {model_xml}')
        self.model = mujoco.MjModel.from_xml_path(model_xml)
        self.data = mujoco.MjData(self.model)

        # optionally start an interactive passive viewer for manual inspection
        self.viewer = None
        self._glfw_window = None
        if self.use_viewer:
            try:
                # try to create an invisible OpenGL context (helps some mujoco viewer backends)
                if glfw is not None:
                    try:
                        if not glfw.init():
                            self.get_logger().debug('glfw.init() returned False')
                        else:
                            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
                            # small invisible window
                            self._glfw_window = glfw.create_window(1, 1, 'mj_offscreen', None, None)
                            if self._glfw_window is not None:
                                glfw.make_context_current(self._glfw_window)
                    except Exception:
                        self.get_logger().debug('Failed to create offscreen GLFW context for viewer')

                # prefer mujoco.viewer submodule if available (zongzhuang imports mujoco.viewer explicitly)
                viewer_launcher = None
                if mujoco_viewer is not None and hasattr(mujoco_viewer, 'launch_passive'):
                    viewer_launcher = getattr(mujoco_viewer, 'launch_passive')
                elif hasattr(mujoco, 'viewer') and hasattr(mujoco.viewer, 'launch_passive'):
                    viewer_launcher = mujoco.viewer.launch_passive

                if viewer_launcher is None:
                    raise RuntimeError('No mujoco.viewer.launch_passive available')

                self.get_logger().info('Launching passive MuJoCo viewer (interactive)')
                self.viewer = viewer_launcher(self.model, self.data)

                try:
                    self.viewer.cam.lookat[:] = [0.0, 0.0, 1.0]
                    self.viewer.cam.distance = 2.0
                    self.viewer.cam.azimuth = 0.0
                    self.viewer.cam.elevation = -30.0
                except Exception:
                    pass
            except Exception as e:
                self.get_logger().warning(f'Failed to launch passive viewer: {e}')

        # publishers/subscribers
        # use configured topic (default '/sim/joint_states') so users can set it to '/joint_states' if needed
        self.pub_js = self.create_publisher(JointState, self.state_topic, 10)
        self.sub_cmd = self.create_subscription(JointState, '/joint_command', self._on_joint_command, 10)

        self._target_positions: Dict[int, float] = {}
        self._lock = threading.Lock()

        # Precompute joint qpos addresses if possible
        self._joint_qposadr = []
        for name in self.joint_names:
            try:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid >= 0:
                    adr = int(self.model.jnt_qposadr[jid])
                    self._joint_qposadr.append(adr)
                else:
                    self._joint_qposadr.append(None)
            except Exception:
                self._joint_qposadr.append(None)

        # === actuator mapping: map joints -> actuator indices (pos_jointN) ===
        self._actuator_name_to_idx = {}
        self._actuator_ctrlrange = {}
        try:
            nact = int(getattr(self.model, 'nactuator'))
        except Exception:
            # fallback: scan ids
            nact = 0
            for i in range(1024):
                if not mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i):
                    break
                nact += 1
        for ai in range(nact):
            aname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, ai)
            if not aname:
                continue
            self._actuator_name_to_idx[aname] = ai
            try:
                self._actuator_ctrlrange[ai] = (float(self.model.actuator_ctrlrange[ai][0]), float(self.model.actuator_ctrlrange[ai][1]))
            except Exception:
                pass

        # Build direct mapping from joint index -> actuator index if actuator names follow pos_jointN or pos_<jointname>
        self._joint_to_actuator = [None] * len(self.joint_names)
        for jidx, jname in enumerate(self.joint_names):
            candidates = [f'pos_{jname}', f'pos_joint{jidx+1}', f'pos_joint{jidx}']
            for cand in candidates:
                if cand in self._actuator_name_to_idx:
                    self._joint_to_actuator[jidx] = self._actuator_name_to_idx[cand]
                    break

        # Initialize previous ctrl values for smoothing
        try:
            self._prev_ctrls = [float(self.data.ctrl[i]) for i in range(nact)]
        except Exception:
            self._prev_ctrls = [0.0] * nact

        # --- New: allow setting initial joint positions at startup via parameter
        self.declare_parameter('initial_positions', [0.947, 0.367, 0.148, 1.16, 0.148, 0.97, -0.769])
        try:
            initial_positions = list(self.get_parameter('initial_positions').get_parameter_value().double_array_value)
        except Exception:
            initial_positions = []

        if initial_positions:
            # apply initial qpos values where joint qpos addresses are known
            for jidx, val in enumerate(initial_positions):
                try:
                    if jidx < len(self._joint_qposadr):
                        adr = self._joint_qposadr[jidx]
                        if adr is not None:
                            self.data.qpos[adr] = float(val)
                        else:
                            # fallback: try set via data.joint(name).qpos if available
                            try:
                                self.data.joint(self.joint_names[jidx]).qpos = float(val)
                            except Exception:
                                pass
                except Exception:
                    pass
            # update forward kinematics after setting qpos
            try:
                mujoco.mj_forward(self.model, self.data)
            except Exception:
                pass

        # timer
        period = 1.0 / max(1.0, float(self.control_frequency))
        self._timer = self.create_timer(period, self._loop)

        # optional periodic joint_states logger
        if getattr(self, 'log_joint_states', False):
            try:
                js_period = max(1e-3, 1.0 / float(self.joint_states_log_frequency))
            except Exception:
                js_period = 1.0
            try:
                self._js_log_timer = self.create_timer(js_period, self._log_joint_states)
            except Exception as e:
                self.get_logger().warning(f'Failed to create joint_states log timer: {e}')

        self.get_logger().info('MujocoSimNode initialized')

    def _on_joint_command(self, msg: JointState):
        # Accept incoming JointState commands; map by name
        with self._lock:
            for i, name in enumerate(msg.name):
                try:
                    pos = float(msg.position[i]) if i < len(msg.position) else None
                except Exception:
                    pos = None
                if pos is None:
                    continue
                # accept both 'joint1' or 'joint_1' or 'joint1' etc. Try to find matching index
                try:
                    # direct match
                    idx = self.joint_names.index(name) if name in self.joint_names else None
                except Exception:
                    idx = None
                if idx is None:
                    # try stripping prefixes/suffixes
                    bare = name.replace('_', '').replace('-', '')
                    for k, jn in enumerate(self.joint_names):
                        if jn.replace('_', '') in bare or bare in jn.replace('_', ''):
                            idx = k
                            break
                if idx is not None:
                    self._target_positions[idx] = pos

            # Log the received joint_command message and publishers to a file if enabled
            if getattr(self, 'log_joint_commands', False):
                try:
                    # snapshot publishers for the topic (best-effort; not per-message identity)
                    pubs = []
                    try:
                        infos = self.get_publishers_info_by_topic('/joint_command')
                        for p in infos:
                            node_name = getattr(p, 'node_name', None)
                            node_ns = getattr(p, 'node_namespace', None)
                            if node_name is None and node_ns is None:
                                continue
                            pubs.append(f"{node_name}@{node_ns}")
                    except Exception:
                        pubs = []

                    payload = {
                        'ts': datetime.datetime.utcnow().isoformat() + 'Z',
                        'publishers': pubs,
                        'msg_header': {
                            'stamp': getattr(getattr(msg, 'header', None), 'stamp', None),
                            'frame_id': getattr(getattr(msg, 'header', None), 'frame_id', None),
                        },
                        'names': list(msg.name),
                        'positions': list(msg.position),
                        'mapped_targets': {str(k): v for k, v in self._target_positions.items()},
                    }
                    # append as one JSON line
                    try:
                        with open(self.joint_command_log_file, 'a') as f:
                            f.write(json.dumps(payload, default=str) + '\n')
                    except Exception as e:
                        self.get_logger().warning(f'Failed to write joint_command log file: {e}')
                except Exception as e:
                    self.get_logger().warning(f'Failed to prepare joint_command log entry: {e}')

    def _apply_targets(self):
        # Apply target positions to actuator controls when available (positional actuators)
        with self._lock:
            # If control_mode is 'direct', write target into qpos directly and advance kinematics
            if getattr(self, 'control_mode', 'actuator') == 'direct':
                for idx, pos in list(self._target_positions.items()):
                    if idx < 0 or idx >= len(self.joint_names):
                        continue
                    try:
                        adr = self._joint_qposadr[idx] if idx < len(self._joint_qposadr) else None
                        if adr is not None:
                            self.data.qpos[adr] = float(pos)
                        else:
                            # fallback to joint accessor
                            try:
                                self.data.joint(self.joint_names[idx]).qpos = float(pos)
                            except Exception:
                                pass
                    except Exception:
                        self.get_logger().debug(f'Failed to set direct qpos for joint idx={idx}')
                # update kinematics so frame placements reflect new qpos immediately
                try:
                    mujoco.mj_forward(self.model, self.data)
                except Exception:
                    pass
                return
            # smoothing step per control publish
            step_max = 0.2
            for idx, pos in self._target_positions.items():
                if idx < 0 or idx >= len(self.joint_names):
                    continue
                act_idx = None
                # get mapped actuator index
                if idx < len(self._joint_to_actuator):
                    act_idx = self._joint_to_actuator[idx]
                # fallback: try common actuator name patterns
                if act_idx is None:
                    cand1 = f'pos_joint{idx+1}'
                    cand2 = f'pos_{self.joint_names[idx]}'
                    if cand1 in self._actuator_name_to_idx:
                        act_idx = self._actuator_name_to_idx[cand1]
                    elif cand2 in self._actuator_name_to_idx:
                        act_idx = self._actuator_name_to_idx[cand2]
                if act_idx is None:
                    # nothing to do for this joint (no actuator mapped)
                    continue

                # clamp to ctrlrange if available
                low, high = None, None
                try:
                    low, high = self._actuator_ctrlrange.get(act_idx, (pos, pos))
                except Exception:
                    low, high = pos, pos
                tgt = float(pos)
                tgt_clamped = max(min(tgt, high), low) if (low is not None and high is not None) else tgt

                # smoothing
                prev = float(self._prev_ctrls[act_idx]) if act_idx < len(self._prev_ctrls) else 0.0
                delta = tgt_clamped - prev
                if abs(delta) > step_max:
                    tgt_clamped = prev + step_max * (1 if delta > 0 else -1)

                # final clamp
                if low is not None and tgt_clamped < low:
                    tgt_clamped = low
                if high is not None and tgt_clamped > high:
                    tgt_clamped = high

                # write into control buffer
                try:
                    self.data.ctrl[act_idx] = float(tgt_clamped)
                    if act_idx < len(self._prev_ctrls):
                        self._prev_ctrls[act_idx] = float(tgt_clamped)
                except Exception:
                    self.get_logger().debug(f'Failed to set ctrl for actuator idx={act_idx}')

    def _publish_joint_states(self):
        names = []
        positions = []
        for i, jn in enumerate(self.joint_names):
            names.append(jn)
            try:
                adr = self._joint_qposadr[i]
                if adr is not None:
                    positions.append(float(self.data.qpos[adr]))
                else:
                    positions.append(float(self.data.joint(jn).qpos))
            except Exception:
                positions.append(0.0)
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = names
        msg.position = positions
        self.pub_js.publish(msg)

    def _log_joint_states(self):
        # Periodically write current joint_states (names + positions) to a file as JSON lines
        if not getattr(self, 'log_joint_states', False):
            return
        try:
            # snapshot publishers for the state topic (best-effort)
            pubs = []
            try:
                infos = self.get_publishers_info_by_topic(self.state_topic)
                for p in infos:
                    node_name = getattr(p, 'node_name', None)
                    node_ns = getattr(p, 'node_namespace', None)
                    if node_name is None and node_ns is None:
                        continue
                    pubs.append(f"{node_name}@{node_ns}")
            except Exception:
                pubs = []

            # capture current joint positions under lock to avoid races with _apply_targets
            names = []
            positions = []
            try:
                with self._lock:
                    for i, jn in enumerate(self.joint_names):
                        names.append(jn)
                        try:
                            adr = self._joint_qposadr[i]
                            if adr is not None:
                                positions.append(float(self.data.qpos[adr]))
                            else:
                                positions.append(float(self.data.joint(jn).qpos))
                        except Exception:
                            positions.append(0.0)
            except Exception:
                # fallback, try without lock
                for i, jn in enumerate(self.joint_names):
                    names.append(jn)
                    try:
                        adr = self._joint_qposadr[i]
                        if adr is not None:
                            positions.append(float(self.data.qpos[adr]))
                        else:
                            positions.append(float(self.data.joint(jn).qpos))
                    except Exception:
                        positions.append(0.0)

            payload = {
                'ts': datetime.datetime.utcnow().isoformat() + 'Z',
                'publishers': pubs,
                'names': names,
                'positions': positions,
            }
            try:
                with open(self.joint_states_log_file, 'a') as f:
                    f.write(json.dumps(payload, default=str) + '\n')
            except Exception as e:
                self.get_logger().warning(f'Failed to write joint_states log file: {e}')
        except Exception as e:
            self.get_logger().warning(f'Error while logging joint_states: {e}')

    def _loop(self):
        # apply requested targets
        self._apply_targets()
        # step physics a small amount so actuators take effect
        try:
            mujoco.mj_step(self.model, self.data)
        except Exception:
            # fallback to forward kinematics if stepping unsupported in this environment
            try:
                mujoco.mj_forward(self.model, self.data)
            except Exception:
                pass
        # publish state
        self._publish_joint_states()
        # drive passive viewer if running so it updates with simulation
        try:
            if getattr(self, 'viewer', None) is not None:
                try:
                    self.viewer.sync()
                except Exception:
                    pass
        except Exception:
            pass

    def _close_viewer_and_glfw(self):
        # close mujoco viewer if present
        try:
            if getattr(self, 'viewer', None) is not None:
                try:
                    self.viewer.close()
                except Exception:
                    pass
                self.viewer = None
        except Exception:
            pass
        # destroy glfw window if created
        try:
            if getattr(self, '_glfw_window', None) is not None and glfw is not None:
                try:
                    glfw.destroy_window(self._glfw_window)
                except Exception:
                    pass
                try:
                    glfw.terminate()
                except Exception:
                    pass
                self._glfw_window = None
        except Exception:
            pass

    def destroy_node(self):
        # try to close passive viewer and glfw context if present
        try:
            self._close_viewer_and_glfw()
        except Exception:
            pass
        return super().destroy_node()

    def render_viewer(self):
        # compatible helper similar to zongzhuang implementation
        try:
            if getattr(self, 'viewer', None) is None:
                # attempt to (re)launch
                if mujoco_viewer is not None and hasattr(mujoco_viewer, 'launch_passive'):
                    self.viewer = mujoco_viewer.launch_passive(self.model, self.data)
                elif hasattr(mujoco, 'viewer') and hasattr(mujoco.viewer, 'launch_passive'):
                    self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                else:
                    self.get_logger().warning('No mujoco viewer available to render')
                    return
            try:
                self.viewer.cam.lookat[:] = [0.0, 0.0, 1.0]
                self.viewer.cam.distance = 2
                self.viewer.cam.azimuth = 0
                self.viewer.cam.elevation = -45
            except Exception:
                pass
        except Exception as e:
            self.get_logger().warning(f'render_viewer failed: {e}')

    def viewer_step(self):
        try:
            if getattr(self, 'viewer', None) is not None:
                self.viewer.sync()
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = MujocoSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
