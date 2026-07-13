import numpy as np
import tensorflow as tf
from scipy.stats import pearsonr

METRIC_NAMES = ["Balance", "Power", "Technique", "Defence"]

def mc_dropout_predict(model, input_tensor, n_passes: int = 30, scale_to_100: bool = True):
    input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.float32)
    
    # If input is single sample without batch dim, add it
    if len(input_tensor.shape) == 2:
        input_tensor = tf.expand_dims(input_tensor, axis=0)
        
    batch_size = input_tensor.shape[0]
    raw_passes = np.zeros((n_passes, batch_size, 4), dtype=np.float32)
    
    for i in range(n_passes):
        pred = model(input_tensor, training=True)
        if isinstance(pred, tuple) or isinstance(pred, list):
            pred = pred[0]
        raw_passes[i] = pred.numpy()

    if scale_to_100:
        raw_passes = raw_passes * 100.0

    # Calculate across n_passes (axis 0)
    mean_scores = raw_passes.mean(axis=0)
    std_scores = raw_passes.std(axis=0)
    
    # If single batch, return 1D
    if batch_size == 1:
        mean_scores = mean_scores[0]
        std_scores = std_scores[0]
        raw_passes = raw_passes[:, 0, :]
    
    return mean_scores, std_scores, raw_passes

def uncertainty_error_correlation(mean_scores_list, std_scores_list, ground_truth_list):
    means = np.array(mean_scores_list)      # (n_videos, 4)
    stds = np.array(std_scores_list)        # (n_videos, 4)
    truths = np.array(ground_truth_list)    # (n_videos, 4)

    abs_errors = np.abs(means - truths)     # (n_videos, 4)
    correlations = {}
    for idx, name in enumerate(METRIC_NAMES):
        if stds[:, idx].std() == 0 or abs_errors[:, idx].std() == 0:
            correlations[name] = (float("nan"), float("nan"))
            continue
        corr, p_value = pearsonr(stds[:, idx], abs_errors[:, idx])
        correlations[name] = (float(corr), float(p_value))

    return correlations
