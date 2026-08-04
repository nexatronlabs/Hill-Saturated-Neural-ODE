from setuptools import setup, find_packages

setup(
    name="nexatron",
    version="1.1.0",
    description="Surgical Hill-Kinetics Field Saturation for Neural ODEs (NXL-2026-01)",
    author="Nexatron Labs",
    author_email="research@nexatronlabs.org",
    url="https://github.com/nexatronlabs/Hill-Saturated-Neural-ODE",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)