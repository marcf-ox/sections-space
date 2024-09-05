from setuptools import setup, find_packages
from sections_space import __version__

setuptools.setup(
    name='sections-space',
    version=__version__,
    author='Marcf',
    license='MIT',
    install_requires=['numpy>=1.24.2'],
    python_requires=">=3.8.0",
    packages=find_packages(),
)