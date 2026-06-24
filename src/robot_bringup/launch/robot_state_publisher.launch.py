import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import xacro


def generate_launch_description():
    # Locate the xacro file from the installed robot_description package
    xacro_path = os.path.join(
        get_package_share_directory('robot_description'),
        'urdf',
        'robot.urdf.xacro'
    )

    # Process xacro into raw URDF XML string
    robot_description_xml = xacro.process_file(xacro_path).toxml()

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{
                'robot_description': robot_description_xml,
            }],
            output='screen',
        ),
    ])