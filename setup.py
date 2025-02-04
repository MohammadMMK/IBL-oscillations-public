from setuptools import setup, find_packages

setup(
    name="IBL_oscillations",  # Replace with your desired package name
    version="0.1.0",
    description="IBL oscillations analysis",
    author="mohammad keshtkar",
    packages=find_packages(include=["_analyses", "_analyses.*"]),  # Include your modules and submodules,
)
