from setuptools import find_packages, setup

setup(
    name="svdlab",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-image>=0.20.0",
        "Pillow>=10.0.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
    ],
    python_requires=">=3.10",
)
