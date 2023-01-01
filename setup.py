import setuptools

with open('README.md') as f:
    long_description = f.read()

setuptools.setup(
    name="pythonabm",
    version="0.3.1",
    author="Jack Toppen",
    author_email="jtoppen3@gatech.edu",
    description="Framework for building Agent-Based Models (ABMs) in Python",
    long_description=long_description,
    long_description_content_type='text/markdown',
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
