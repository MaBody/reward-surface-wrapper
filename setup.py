from setuptools import setup

setup(
    name="reward-surfaces-wrapper",
    version="1.0",
    description="Implements a simple wrapper for RL agents to visualize filter-normalized reward surfaces.",
    author="Mattis Bodynek",
    author_email="mattisbodynek@gmail.com",
    packages=["wrapper"],  # same as name
    install_requires=[
        "pytorch",
        "numpy",
        "plotly",
    ],
)
