from setuptools import setup

setup(
    name='aerial_robotics',
    packages=['flightsim', 'algorithms'],
    version='0.1',
    python_requires='>=3.8,<3.9',
    install_requires=[
            'PyYAML==6.0.3',
            'opencv-python==5.0.0.93',
            'cvxopt==1.3.2',
            'matplotlib==3.2.2',
            'numpy==1.23.5',
            'scipy==1.10.1',
            'timeout_decorator==0.5.0'])
