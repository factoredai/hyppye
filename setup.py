from distutils.core import setup
from setuptools import setup, find_packages

setup(
    name='hyppye',
    version='0.1',
    description='Hyperbolic Python Embeddings',
    url='https://github.com/lacunafellow/hyperbolicEmbeddings_team4',
    author='Factored AI',
    author_email='info@factored.ai',
    packages=find_packages(),
    install_requires=[
        'mpmath>=1.1.0',
        'networkx>=2.3',
        'numpy>=1.16.3',
        'pandas>=0.24.2',
        'scipy>=1.2.1',
        'progress==1.5'
    ],
    entry_points="""
        [console_scripts]
        hyppye=hyppye.main:main
    """
)
