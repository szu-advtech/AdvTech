from setuptools import setup,find_packages
import os
path = os.path.join(os.path.dirname(__file__),"python")

with open(os.path.join(path, "JSJQY/__init__.py"), "r", encoding='utf8') as fh:
    for line in fh:
        if line.startswith('__version__'):
            version = line.split("'")[1]
            break
    else:
        raise RuntimeError("Unable to find version string.")

setup(
    name="JSJQY",
    version=version,
    author="JSJQY Group",
    python_requires='>=3.7',
    packages=find_packages(path),
    package_dir={'': "python"},
    install_requires=[
        "shapely",
        "pyyaml",
        "numpy",
        "tqdm",
        "pillow",
        "astunparse",
        "jittor",
        "tensorboardX",
        "opencv-python",
        "tqdm",
        "terminaltables",
    ],
)