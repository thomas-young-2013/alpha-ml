# -*- encoding: utf-8 -*-
import os
import sys

from alphaml.utils import dependencies
from alphaml.utils.logging_utils import setup_logging

__MANDATORY_PACKAGES__ = '''
numpy==1.16.4
six==1.11.0
scipy==1.3.0
pandas==0.23.0
scikit-learn==0.21.3
tqdm==4.34.0
hyperopt==0.1.2
ConfigSpace==0.4.11
smac==0.11.1
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
