# TODO: Scorer
def get_metric(metricstr):
    # Metrics for classification
    if metricstr in ["accuracy", "acc"]:
        from sklearn.metrics import accuracy_score
        return accuracy_score
    elif metricstr == 'f1':
        from sklearn.metrics import f1_score
        return f1_score
    elif metricstr == 'precision':
        from sklearn.metrics import precision_score
        return precision_score
    elif metricstr == 'recall':
        from sklearn.metrics import recall_score
        return recall_score
    elif metricstr == "auc":
        from sklearn.metrics import roc_auc_score
        return roc_auc_score

    # Metrics for regression
    elif metricstr in ["mean_squared_error", "mse"]:
        from sklearn.metrics import mean_squared_error
        return mean_squared_error
    elif metricstr in ['mean_squared_log_error', "msle"]:
        from sklearn.metrics import mean_squared_log_error
        return mean_squared_log_error
    elif metricstr == "evs":
        from sklearn.metrics import explained_variance_score
        return explained_variance_score
    elif metricstr == "r2":
        from sklearn.metrics import r2_score
        return r2_score
    elif metricstr in ["mean_absolute_error", "mae"]:
        from sklearn.metrics import mean_absolute_error
        return mean_absolute_error
    elif callable(metricstr):
        return metricstr
    else:
        raise ValueError("Given", metricstr, ". Expected valid metric string like 'acc' or callable metric function!")
