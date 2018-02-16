from setuptools import setup, find_packages

setup(name='roispy2',
      version='0.0.1',
      description='Analyze 2P calcium imaging data from mouse retina ganglion cells.',
      author='Ziwei Huang',
      author_email='huang-ziwei@outlook.com',
      url='https://github.com/huangziwei/RoisPy',
      packages=find_packages(),
      install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "matplotlib-scalebar",
        "tifffile",
        "scikit-fmm",
        "scikit-image",
        "h5py",
        "seaborn",
        "opencv-python",
        "astropy",
        "shapely",
      ],
     )

