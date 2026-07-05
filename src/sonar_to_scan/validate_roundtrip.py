#!/usr/bin/env python3
"""
validate_roundtrip.py — end-to-end validation of the sonar_to_scan node.

Replays real (sparse_input, ground_truth) pairs through the LIVE pipeline:
  publish /sparse_scan -> sonar_to_scan node -> /scan
then compares each received /scan against the stored ground_truth.

Confirms, on real in-distribution data through the actual node:
  - the ONNX model loads and runs on the Pi via ROS
  - input/output scaling is correct (a scale bug -> metre-scale error)
  - the node preserves array order (output[i] -> scan.ranges[i])
  - reconstruction error matches the model's known cm-scale behaviour

Does NOT establish physical left/right orientation in the ROS frame
(both arrays share the MATLAB angle convention) — that rests on castRays.m.

Run (ROS sourced) with the sonar_to_scan node already running:
  python3 validate_roundtrip.py --data validation_batch_01.npz --n 5000
"""
import argparse
import time
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import LaserScan


class RoundTripValidator(Node):
    def __init__(self, data_path, num_samples, near, miss_pred):
        super().__init__('roundtrip_validator')
        blob = np.load(data_path)
        self._sparse = blob['sparse_input'].astype(np.float32)   # (N,8)
        self._gt = blob['ground_truth'].astype(np.float32)       # (N,50)
        total = self._sparse.shape[0]
        self._n = total if num_samples <= 0 else min(num_samples, total)
        self._near = near
        self._miss_pred = miss_pred

        self.get_logger().info(
            f"Loaded {total} samples; validating {self._n}. "
            f"sparse={self._sparse.shape} gt={self._gt.shape}")

        self._pub = self.create_publisher(Float32MultiArray, '/sparse_scan', 10)
        self._sub = self.create_subscription(LaserScan, '/scan', self._on_scan, 10)

        self._i = 0
        self._all_abs = []
        self._z = {'0-0.5': [], '0.5-1.0': [], '1.0-2.0': [], '2.0-4.0': []}
        self._missed = self._false = 0
        self._miss_denom = self._false_denom = 0
        self._last_progress = time.time()
        self._done = False
        self._started = False
        self._warned = False

        self._start_timer = self.create_timer(0.25, self._try_start)
        self._watchdog = self.create_timer(1.0, self._check_watchdog)

    def _try_start(self):
        if self._pub.get_subscription_count() < 1:
            if not self._warned:
                self.get_logger().warn(
                    "Waiting for sonar_to_scan on /sparse_scan (is it running?)...")
                self._warned = True
            return
        self._start_timer.cancel()
        self._started = True
        self.get_logger().info("Node detected. Starting replay.")
        self._publish(0)

    def _publish(self, idx):
        m = Float32MultiArray()
        m.data = self._sparse[idx].tolist()
        self._pub.publish(m)

    def _on_scan(self, scan: LaserScan):
        if self._done or not self._started:
            return
        pred = np.asarray(scan.ranges, dtype=np.float32)
        gt = self._gt[self._i]
        if pred.shape != gt.shape:
            self.get_logger().error(
                f"Length mismatch: scan={pred.shape[0]} gt={gt.shape[0]}. Aborting.")
            self._finish()
            return

        abserr = np.abs(pred - gt)
        self._all_abs.append(abserr)
        for lo, hi, key in [(0, 0.5, '0-0.5'), (0.5, 1.0, '0.5-1.0'),
                            (1.0, 2.0, '1.0-2.0'), (2.0, 4.0, '2.0-4.0')]:
            mask = (gt >= lo) & (gt < hi)
            if mask.any():
                self._z[key].append(abserr[mask])

        near_true = gt < self._near
        far_true = gt > self._miss_pred
        self._miss_denom += int(near_true.sum())
        self._false_denom += int(far_true.sum())
        self._missed += int(((pred > self._miss_pred) & near_true).sum())
        self._false += int(((pred < self._near) & far_true).sum())

        self._last_progress = time.time()
        self._i += 1
        if self._i % 1000 == 0:
            self.get_logger().info(f"  {self._i}/{self._n} ...")
        if self._i >= self._n:
            self._finish()
        else:
            self._publish(self._i)

    def _check_watchdog(self):
        if self._done or not self._started:
            return
        if time.time() - self._last_progress > 5.0:
            self.get_logger().error(
                f"Stalled at sample {self._i} (no /scan for 5 s). Partial results.")
            self._finish()

    def _finish(self):
        if self._done:
            return
        self._done = True
        if not self._all_abs:
            self.get_logger().error("No samples processed.")
            rclpy.shutdown()
            return
        allabs = np.concatenate(self._all_abs)
        cm = lambda x: 100.0 * float(x)
        print("\n" + "=" * 58)
        print(f"ROUND-TRIP VALIDATION — {self._i} samples  /sparse_scan -> /scan")
        print("=" * 58)
        print(f"Global MAE      : {cm(allabs.mean()):6.2f} cm")
        print(f"Median |err|    : {cm(np.median(allabs)):6.2f} cm")
        print(f"95th pct |err|  : {cm(np.percentile(allabs, 95)):6.2f} cm")
        print(f"Max |err|       : {cm(allabs.max()):6.2f} cm")
        print("-" * 58)
        print("MAE by TRUE range zone:")
        for key in ['0-0.5', '0.5-1.0', '1.0-2.0', '2.0-4.0']:
            if self._z[key]:
                v = np.concatenate(self._z[key])
                print(f"  {key:9} m : {cm(v.mean()):6.2f} cm   (points: {v.size})")
        print("-" * 58)
        mr = (100.0 * self._missed / self._miss_denom) if self._miss_denom else float('nan')
        fr = (100.0 * self._false / self._false_denom) if self._false_denom else float('nan')
        print(f"Missed-obstacle (true<{self._near}m, pred>{self._miss_pred}m): "
              f"{mr:.3f}%  ({self._missed}/{self._miss_denom})")
        print(f"False-obstacle  (true>{self._miss_pred}m, pred<{self._near}m): "
              f"{fr:.3f}%  ({self._false}/{self._false_denom})")
        print("=" * 58)
        print("PASS if: cm-scale errors AND near zone tighter than far zone.")
        print("Metre-scale error => scaling/variable/order bug.")
        print("Does NOT test physical L/R orientation (see castRays.m).")
        print("=" * 58 + "\n")
        rclpy.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--n', type=int, default=5000, help='samples (<=0 = all)')
    ap.add_argument('--near', type=float, default=1.0)
    ap.add_argument('--miss-pred', type=float, default=1.5)
    a = ap.parse_args()
    rclpy.init()
    node = RoundTripValidator(a.data, a.n, a.near, a.miss_pred)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()