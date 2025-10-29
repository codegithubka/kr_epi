from setuptools import setup, find_packages

setup(
    name="kr_epi",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
    ],
    python_requires=">=3.8",
)
