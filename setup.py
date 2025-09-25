from setuptools import find_packages, setup

package_name = 'je_software'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
        (f'share/{package_name}/launch', ['launch/orbbec.launch.py']),
        (f'share/{package_name}/config', ['config/orbbec.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kleist',
    maintainer_email='gmvonkleist@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'camera_node = je_software.camera_node:main',
        ],
    },
)
