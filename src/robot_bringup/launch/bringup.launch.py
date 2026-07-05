import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import xacro


def generate_launch_description():
    # --- Robot description ---
    xacro_path = os.path.join(
        get_package_share_directory('robot_description'),
        'urdf',
        'robot.urdf.xacro'
    )
    robot_description_xml = xacro.process_file(xacro_path).toxml()

    # --- EKF config ---
    ekf_config = os.path.join(
        get_package_share_directory('robot_bringup'),
        'config',
        'ekf.yaml'
    )

    return LaunchDescription([
        # Publish URDF to /robot_description and broadcast fixed TFs
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{
                'robot_description': robot_description_xml,
            }],
            output='screen',
        ),

        # Fuse wheel_odom + IMU into /odometry/filtered, broadcast odom -> base_footprint
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            parameters=[ekf_config],
            output='screen',
        ),

        # Reconstruct a dense LaserScan from the 8-beam sparse vector.
        # Uses the node's built-in defaults (model from package share,
        # norm_max_range=4.0, angles -49..+49, range [0.25, 4.0]).
        Node(
            package='sonar_to_scan',
            executable='sonar_to_scan_node',
            name='sonar_to_scan',
            output='screen',
        ),
    ])