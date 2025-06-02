from setuptools import setup, find_packages

packages = find_packages()

setup(
    name='ddranking',  # Package name
    version='0.2.0',  # Version number
    description='DD-Ranking: Rethinking the Evaluation of Dataset Distillation',
    long_description=open('README.md').read(),  # Use your README as the long description
    long_description_content_type='text/markdown',
    author='DD-Ranking Team',
    author_email='lizekai@u.nus.edu',
    include_dirs=['ddranking', 'configs', 'static'],
    include_package_data=True,
    packages=packages,  # Automatically discover submodules
    install_requires=[
        'torch',
        'numpy',
        'torchvision',
        'tqdm',
        'scipy<1.16.0',
        'pandas',
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
