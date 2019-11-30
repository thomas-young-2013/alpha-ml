from setuptools import setup, find_packages

requires = [
    'scikit-learn',
    'numpy >= 1.7',
    'pandas',
    'smac'
    ]

setup(name='alpha-ml',
      version='0.1.0',
      description='AutoML toolkit',
      author='DAIM',
      author_email='liyang.cs@pku.edu.cn',
      url='https://github.com/thomas-young-2013/',
      keywords='AutoML',
      packages=find_packages(),
      license='LICENSE.txt',
      test_suite='nose.collector',
      include_package_data=True,
      install_requires=requires)
