from setuptools import setup, find_packages

requirements = ['numpy==1.16.4',
'six==1.11.0',
'scipy==1.3.0',
'pandas==0.23.0',
'scikit-learn==0.21.3',
'tqdm==4.34.0',
'hyperopt==0.1.2',
'ConfigSpace==0.4.11',
'smac==0.11.1']

setup(name='alphaml',
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
      install_requires=requirements)
