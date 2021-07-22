from setuptools import setup, find_packages

setup(
    name="oak",
    version="0.1.1",
    packages=find_packages(),
    python_requires=">=3.7",
    setup_requires=["setuptools_scm"],
    install_requires=[
        "depthai>=2.5.0.0",
        "opencv-python>=4.5.*",
        "numpy>=1.19.5",
        "typer>=0.3.2",
        "matplotlib>=3.4.2",
        "pandas>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pip-tools",
            "pytest>=6.2.3",
            "black>=20.8b1",
        ]
    },
)
