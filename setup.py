from setuptools import setup, find_packages

package_name = 'nr_diff_flat'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='egmc',
    maintainer_email='egmc@todo.todo',
    description='Differential-flatness based Newton-Raphson controller for quadrotor trajectory tracking',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'run_node = nr_diff_flat.run_node:main',
        ],
    },
)
