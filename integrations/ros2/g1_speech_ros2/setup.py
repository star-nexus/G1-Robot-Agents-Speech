from glob import glob

from setuptools import find_packages, setup


package_name = "g1_speech_ros2"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Coldmooon",
    maintainer_email="howtoliy@gmail.com",
    description="ROS 2 lifecycle wrapper for G1 Speech Service",
    license="Unspecified",
    entry_points={
        "console_scripts": [
            "speech_lifecycle_node = g1_speech_ros2.lifecycle_node:main",
        ],
    },
)
