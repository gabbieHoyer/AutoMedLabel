
from src.finetuning.engine.metrics import T1rhoMetric, T2Metric, TissueHeightMetric, TissueVolumeMetric, CartilageThicknessMetric
# Import other metric classes as needed

def metric_factory(metric_name, **kwargs):
    if metric_name.lower() == 't1rho':
        return T1rhoMetric(**kwargs)
    elif metric_name.lower() == 't2':
        return T2Metric(**kwargs)
    elif metric_name.lower() == 'tissueheight':
        return TissueHeightMetric(**kwargs)
    if metric_name.lower() == 'tissuevolume':
        return TissueVolumeMetric(**kwargs)
    elif metric_name.lower() == 'cartilagethickness':
        return CartilageThicknessMetric(**kwargs)
    # Add more metrics as needed
    else:
        raise ValueError(f"Unknown metric type: {metric_name}")

def load_metrics(config, class_names=None, tissue_labels=None):
    metrics = []
    metric_params = {
        'reduction': 'mean_batch',
        'class_names': class_names,
        'tissue_labels': tissue_labels
    }

    for metric_name, use_metric in config.items():
        if use_metric:
            metrics.append(metric_factory(metric_name, **metric_params))

    return metrics