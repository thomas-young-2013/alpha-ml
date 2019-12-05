# -*- encoding: utf-8 -*-
import os
import sys

from alphaml.utils import dependencies
from alphaml.utils.logging_utils import setup_logging

__version__ = '0.1.0'
__MANDATORY_PACKAGES__ = '''
tqdm
xgboost
hyperopt
pynisher
pyrfr
smac==0.8.0
ConfigSpace
pyparsing
lockfile
pyyaml
liac-arff
psutil
sobol_seq
joblib
scikit-learn
setuptools
nose
scipy
Cython
pandas
numpy
'''

dependencies.verify_packages(__MANDATORY_PACKAGES__)

if os.name != 'posix':
    raise ValueError(
        'Detected unsupported operating system: %s. Please check '
        'the compability information of auto-sklearn: http://automl.github.io'
        '/auto-sklearn/stable/installation.html#windows-osx-compability' %
        sys.platform
    )

if sys.version_info < (3, 5):
    raise ValueError(
        'Unsupported python version %s found. Auto-sklearn requires Python '
        '3.5 or higher.' % sys.version_info
    )

setup_logging()
