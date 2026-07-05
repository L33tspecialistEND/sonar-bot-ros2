#!/usr/bin/env python3
"""
sonar_to_scan — reconstruct a dense LaserScan from the 8-beam ultrasonic vector.

Subscribes : /sparse_scan  (std_msgs/Float32MultiArray) — 8 ranges in metres,
                            one per ultrasonic beam steering angle.
Publishes  : /scan         (sensor_msgs/LaserScan)      — 50-point dense scan.

Model/data interface (verified empirically, not assumed):
  input  'sparse_input'  [batch, 8]  float32, metres
  output 'dense_output'  [batch, 50] float32, metres
  angles gt_angles_deg = -49:2:49  -> 50 pts, -49 deg .. +49 deg, ascending
  order  positive angle = left (CCW, ROS +y)   [rests on castRays.m via prior chat]
  no input/output scaling (raw metres in, raw metres out)
Range handling: Option A — clip to [range_min, range_max] before publishing.
"""

import os
import math

import numpy as np
import onnxruntime as ort

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import LaserScan


class SonarToScan(Node):
    def __init__(self):
        super().__init__('sonar_to_scan')

        # --- Parameters (defaults are the confirmed values; nothing hidden) ---
        self.declare_parameter('model_path', '')        # '' -> package share dir
        self.declare_parameter('input_topic', '/sparse_scan')
        self.declare_parameter('output_topic', '/scan')
        self.declare_parameter('frame_id', 'sensor_frame')
        self.declare_parameter('num_beams_in', 8)
        self.declare_parameter('angle_min_deg', -49.0)
        self.declare_parameter('angle_max_deg', 49.0)
        self.declare_parameter('range_min', 0.25)
        self.declare_parameter('range_max', 4.0)
        self.declare_parameter('norm_max_range', 4.0)   # equals training MAX_RANGE
        self.declare_parameter('scan_time', 0.1)        # ~10 Hz sparse input (see note)

        p = self.get_parameter
        self._input_topic = p('input_topic').value
        self._output_topic = p('output_topic').value
        self._frame_id = p('frame_id').value
        self._num_in = int(p('num_beams_in').value)
        self._range_min = float(p('range_min').value)
        self._range_max = float(p('range_max').value)
        self._norm_max = float(p('norm_max_range').value)
        self._scan_time = float(p('scan_time').value)
        angle_min_deg = float(p('angle_min_deg').value)
        angle_max_deg = float(p('angle_max_deg').value)

        # --- Locate and load the ONNX model ---
        model_path = p('model_path').value
        if not model_path:
            model_path = os.path.join(
                get_package_share_directory('sonar_to_scan'), 'cnn_model.onnx')
        if not os.path.isfile(model_path):
            self.get_logger().error(f"ONNX model not found: {model_path}")
            raise FileNotFoundError(model_path)

        self._session = ort.InferenceSession(
            model_path, providers=['CPUExecutionProvider'])
        self._in_name = self._session.get_inputs()[0].name
        self._out_name = self._session.get_outputs()[0].name
        self.get_logger().info(
            f"Loaded model '{os.path.basename(model_path)}' "
            f"(in='{self._in_name}', out='{self._out_name}').")

        # --- Confirm input width against the model, read output width from it ---
        in_shape = self._session.get_inputs()[0].shape    # e.g. ['batch_size', 8]
        out_shape = self._session.get_outputs()[0].shape  # e.g. ['batch_size', 50]
        if (len(in_shape) >= 2 and isinstance(in_shape[1], int)
                and in_shape[1] != self._num_in):
            self.get_logger().warn(
                f"Model expects {in_shape[1]} inputs but num_beams_in="
                f"{self._num_in}; using model value {in_shape[1]}.")
            self._num_in = in_shape[1]
        if len(out_shape) >= 2 and isinstance(out_shape[1], int):
            self._num_out = out_shape[1]
        else:
            self._num_out = 50
            self.get_logger().warn("Could not read output width; assuming 50.")

        # --- Derive scan geometry from the (now final) output width ---
        self._angle_min = math.radians(angle_min_deg)
        self._angle_max = math.radians(angle_max_deg)
        self._angle_increment = (
            (self._angle_max - self._angle_min) / (self._num_out - 1)
            if self._num_out > 1 else 0.0)

        # --- ROS interfaces ---
        self._pub = self.create_publisher(LaserScan, self._output_topic, 10)
        self._sub = self.create_subscription(
            Float32MultiArray, self._input_topic, self._on_sparse_scan, 10)

        self.get_logger().info(
            f"Ready: '{self._input_topic}' ({self._num_in} beams) -> "
            f"'{self._output_topic}' ({self._num_out}-pt LaserScan, "
            f"{math.degrees(self._angle_increment):.3f} deg step, "
            f"frame '{self._frame_id}').")

    def _on_sparse_scan(self, msg: Float32MultiArray):
        data = list(msg.data)
        if len(data) != self._num_in:
            self.get_logger().warn(
                f"Ignoring /sparse_scan with {len(data)} values "
                f"(expected {self._num_in}).")
            return

        try:
            x = np.asarray(data, dtype=np.float32).reshape(1, self._num_in) / self._norm_max
            y = self._session.run([self._out_name], {self._in_name: x})[0]
        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")
            return

        ranges = np.asarray(y, dtype=np.float32).reshape(-1) * self._norm_max
        # Option A: clamp to the declared sensor range (after de-normalising).
        ranges = np.clip(ranges, self._range_min, self._range_max)

        scan = LaserScan()
        scan.header.stamp = self.get_clock().now().to_msg()
        scan.header.frame_id = self._frame_id
        scan.angle_min = float(self._angle_min)
        scan.angle_max = float(self._angle_max)
        scan.angle_increment = float(self._angle_increment)
        scan.time_increment = 0.0      # 8 beams captured together, not swept
        scan.scan_time = float(self._scan_time)
        scan.range_min = float(self._range_min)
        scan.range_max = float(self._range_max)
        scan.ranges = ranges.tolist()
        scan.intensities = []          # no intensity data from this sensor

        self._pub.publish(scan)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = SonarToScan()
        rclpy.spin(node)
    except (KeyboardInterrupt, FileNotFoundError):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()