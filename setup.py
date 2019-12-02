from setuptools import setup, find_packages

with open('requirements.txt') as fh:
    requirements = fh.read()
requirements = requirements.split('\n')
requirements = [requirement.strip() for requirement in requirements]

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
