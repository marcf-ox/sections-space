from setuptools import setup, find_packages

setuptools.setup(
    name='sections-space',
    version="0.0.0",
    author='Marcf',
    license='MIT',
    install_requires=['numpy>=1.23','scipy>=1.10.0','cfractions>=2.2.0','networkx>=2.6.2','matplotlib>=3.3.4'],
    python_requires=">=3.8.0",
    packages=find_packages(),
)
