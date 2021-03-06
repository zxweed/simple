from setuptools import setup
import os

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


setup(
    name="simple",
    version="0.4",
    author="zxweed",
    author_email="zxweed@gmail.com",
    description="A package that implements some simple trading functions",
    long_description_content_type="text/markdown",
    url="https://github.com/zxweed/simple",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6'
)