import setuptools

setuptools.setup(
    name="pythonabm",
    version="0.1.7",
    author="Jack Toppen",
    author_email="jtoppen3@gatech.edu",
    description="Framework for building Agent-Based Models (ABMs) in Python",
    packages=["pythonabm"],
    python_requires=">=3.7",
    install_requires=[
        "matplotlib",
        "numpy>=1.11",
        "numba",
        "opencv-python",
        "psutil",
        "python-igraph",
        "pyyaml",
        "scipy",
    ],
)
