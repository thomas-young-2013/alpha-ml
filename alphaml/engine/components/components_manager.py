from collections import OrderedDict
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter


class ComponentsManager(object):
    def __init__(self):
        self.model_dict = OrderedDict()

    def get_hyperparameter_search_space(self, task_type, optimizer='smac', include=None, exclude=None):
        if task_type in ['binary', 'multiclass']:
            from alphaml.engine.components.models.classification import _classifiers, _addons
            self.model_dict.update(_classifiers)
            self.model_dict.update(_addons.components)
            builtin_models = self.model_dict.keys()
            builtin_estimators = self.model_dict.copy()
        elif task_type in ['continuous']:
            from alphaml.engine.components.models.regression import _regressors, _addons
            self.model_dict.update(_regressors)
            self.model_dict.update(_addons.components)
            builtin_models = self.model_dict.keys()
            builtin_estimators = self.model_dict.copy()
        elif task_type in ['img_binary', 'img_multiclass', 'img_multilabel-indicator']:
            from alphaml.engine.components.models.image_classification import _img_classifiers
            builtin_models = _img_classifiers.keys()
            builtin_estimators = _img_classifiers
        else:
            raise ValueError('Undefined Task Type: %s' % task_type)

        model_candidates = set()
        if include is not None:
            for model in include:
                if model in builtin_models:
                    model_candidates.add(model)
                else:
                    raise ValueError("The estimator %s is NOT available in alpha-ml!" % str(model))
        else:
            model_candidates = set(builtin_models)

        if exclude is not None:
            for model in exclude:
                if model in model_candidates:
                    model_candidates.remove(model)

        return self.get_configuration_space(builtin_estimators, list(model_candidates), optimizer=optimizer)

    def get_configuration_space(self, builtin_estimators, model_candidates, optimizer='smac'):
        config_dict = dict()
        for model_item in model_candidates:
            sub_configuration_space = builtin_estimators[model_item].get_hyperparameter_search_space(
                optimizer=optimizer)
            config_dict[model_item] = sub_configuration_space
        return config_dict

    @staticmethod
    def build_hierarchical_configspace(config_dict):
        """
        Reference: pipeline/base=325, classification/__init__=121
        """
        cs = ConfigurationSpace()
        candidates = list(config_dict.keys())
        # TODO: set the default model.
        model_option = CategoricalHyperparameter("estimator", candidates, default_value=candidates[0])
        cs.add_hyperparameter(model_option)

        for model_item in candidates:
            sub_configuration_space = config_dict[model_item]
            parent_hyperparameter = {'parent': model_option,
                                     'value': model_item}
            cs.add_configuration_space(model_item, sub_configuration_space, parent_hyperparameter=parent_hyperparameter)
        return cs
