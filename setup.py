from setuptools import setup, find_packages

setup(
    name='DD-Ranking',  # Package name
    version='0.1.0',  # Version number
    description='DD-Ranking: A unified benchmark for dataset distillation method evaluation',
    long_description=open('README.md').read(),  # Use your README as the long description
    long_description_content_type='text/markdown',
    author='DD-Ranking Team',
    author_email='lizekai@u.nus.edu',
    url={
        'homepage': 'https://github.com/NUS-HPC-AI-Lab/DD-Ranking',
        'repository': 'https://github.com/NUS-HPC-AI-Lab/DD-Ranking',
        'documentation': 'https://github.com/NUS-HPC-AI-Lab/DD-Ranking'
    },
    license='MIT',  # License type
    packages=find_packages(),  # Automatically discover submodules
    install_requires=[
        'torch',
        'numpy',
        'optuna',
        'torchvision',
        'tqdm',
        'scipy',
        'kornia',
        'timm'
    ],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
