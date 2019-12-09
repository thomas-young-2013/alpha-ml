"""Constants used in TASK TYPE.
"""
CLASSIFICATION = 0
BINARY_CLS = 1
MULTICLASS_CLS = 2
REGRESSION = 3
IMG_CLS = 4

REG_TASKS = [REGRESSION]
CLS_TASKS = [BINARY_CLS, MULTICLASS_CLS]

SUPPORTED_TASK_TYPES = REG_TASKS + CLS_TASKS

TASK_TYPE2STR = {BINARY_CLS: "binary",
                 MULTICLASS_CLS: "multiclass",
                 REGRESSION: "regression"}

TASK_STR2TYPE = {"binary": BINARY_CLS,
                 "multiclass": MULTICLASS_CLS,
                 "regression": REGRESSION}

DENSE = 5
SPARSE = 6
PREDICTIONS = 7
INPUT = 8

SIGNED_DATA = 9
UNSIGNED_DATA = 10

"""Constants used in Integer.
"""
MAX_INT = 999999999
# Classification uses - metric, Regression uses metric
FAILED = 2147483647.0

"""Constants used in feature engineering
"""
feature_types = ['discrete', 'numerical', 'ordinal', 'categorical', 'text']
DISCRETE = 'discrete'
NUMERICAL = 'numerical'
TEXT = 'text'
CATEGORICAL = 'categorical'
ORDINAL = 'ordinal'

FEATURE_TYPES = [DISCRETE, NUMERICAL, TEXT, CATEGORICAL, ORDINAL]
