from setuptools import find_packages, setup

package_name = 'sonar_to_scan'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'cnn_model.onnx', 'cnn_model.onnx.data']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robotbrain',
    maintainer_email='robotbrain@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'sonar_to_scan_node = sonar_to_scan.sonar_to_scan_node:main',
        ],
    },
)
