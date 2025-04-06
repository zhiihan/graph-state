from setuptools import find_packages, setup

with open("./README.md", "r") as f:
    long_description = f.read()

setup(
    name="ClusterSim",
    version="0.0.1",
    description="A browser-based interactive cluster state simulator.",
    license="MIT",
    author="zhiihan",
    author_email="zhihuahan72@gmail.com",
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
