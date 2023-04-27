"""_summary_line = "Utility functions for stattools"

"""
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd

from packaging import version
from typing import List, Tuple, Dict, Union, Optional, Any, Union
from pandas.api.typing import FrameOrSeries
try:
    from typing import Literal
except:
    from typing_extensions import Literal
if version.parse(np.version.version) >= version.parse('1.20.0'):
    from numpy.typing import ArrayLike, DTypeLike
else:
    ArrayLike = Union[
        np.ndarray,
        List[float],
        List[List[float]],
        Tuple[float, ...],
        Tuple[Tuple[float, ...], ...],
        float,
        int,
        # Add more types as needed
    ]
    DTypeLike = Union[
        "np.dtype",
        type,
        Literal["float64"],
        # Add more types or string literals as needed
    ]


def calculate_conflation(p1: float, p2: float) -> float:
    """calculates the conflation between two probabilities

    Parameters
    ----------
    p1 : float
        probability 1
    p2 : float
        probability 2

    Returns
    -------
    float
        conflation between p1 and p2
    """
    return (p1*p2)/(p1*p2 + (1 - p1)*(1 - p2))


def get_confusion_matrix(y_true: ArrayLike,
                         y_pred: ArrayLike,
                         inconclusive_mask: Optional[ArrayLike] = None,
                         threshold: float = 0.5) -> ArrayLike:
    """Calculates the confusion matrix from the true and predicted labels.
    The predicted labels can be either probabilities or binary labels ['fit', 'non-fit', 'inconclusive'].
    If the predicted labels are probabilities, then the threshold is used to 
    convert them to binary labels.
    If you want to include inconclusive while using the probabilities, the instances 
    corresponding to inconclusive must be replaced by np.nan
    Parameters
    ----------
    y_true : ArrayLike
        the true labels
    y_pred : ArrayLike
        the predicted labels or probabilities
    inconclusive_mask : Optional[ArrayLike], optional
        the mask for inconclusive instances, by default None
    Returns
    -------
    ArrayLike
        the confusion matrix as a 3x3 numpy array
    """
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    assert np.shape(y_true) == np.shape(y_pred), "true and predicted labels must be the same shape"
    # check if the predicted labels are probabilities or binary labels
    if isinstance(y_pred[0], float):
        predicted_labels = np.where(y_pred >= threshold, 'fit', 'non-fit')
        if inconclusive_mask is not None:
            predicted_labels[inconclusive_mask] = 'inconclusive'
        y_pred = predicted_labels
    conf_mat = confusion_matrix(y_true, y_pred, labels=[
                                'non-fit', 'inconclusive', 'fit'])
    return conf_mat


def evalute_cross_val(df: FrameOrSeries, 
                      inconclusive_mask: Optional[ArrayLike] = None, 
                      threshold: float = 0.5) -> Dict[str, Dict[str, float]]:
    """Evaluates the cross validation results from the dataframe

    Parameters
    ----------
    df : FrameOrSeries
        Pandas dataframe with the cross validation results
    inconclusive_mask : Optional[ArrayLike], optional
        a numpy mask that corresponds to inconvlusive results, by default None
    threshold : float, optional
        Threshold for probability, by default 0.5

    Returns
    -------
    Dict[str, Dict[str, float]]
        Metrics as a dictionary of dictionaries with the keys being the metric 
        names and the values being the mean and standard deviation
    """
    metrics = []
    for cross_val in df.cross_val_idx.unique():
        idx = df.cross_val_idx == cross_val
        for c in df.columns:
            if c == 'cross_val_idx':
                continue
            if c == 'ground_truth':
                y_true = df[c][idx].values
            else:
                y_pred = df[c][idx].values
        conf_mat = get_confusion_matrix(
            y_true, y_pred, inconclusive_mask, threshold)
        metrics.append(calculate_metrics(conf_mat))
    ret = {}
    keys = list(metrics[0].keys())
    if inconclusive_mask is None:
        keys.remove('INR')
        keys.remove('IPR')
    for mtr in keys:
        ret[mtr] = {'mean': np.mean([m[mtr] for m in metrics]),
                    'std': np.std([m[mtr] for m in metrics])}
    return ret


def calculate_metrics(conf_mat: ArrayLike) -> Dict:
    """Calculates the metrics from the confusion matrix
    The metrics are:
    1. True Positive Rate (TPR)
    2. True Negative Rate (TNR)
    3. False Negative Rate (FNR)
    4. False Positive Rate (FPR)
    5. Inconclusive Negative Rate (INR)
    6. Inconclusive Positive Rate (IPR)
    7. Accuracy (ACC)

    Parameters
    ----------
    conf_mat : np.ndarray
        a 3x3 confusion matrix

    Returns
    -------
    Dict
        The metrics as a dictionary
    """
    conf_mat = np.array(conf_mat)
    assert conf_mat.shape == (3, 3), "confusion matrix must be 3x3"
    TPR = conf_mat[2, 2]/(conf_mat[2, 2] + conf_mat[2, 0])
    TNR = conf_mat[0, 0]/(conf_mat[0, 0] + conf_mat[0, 2])
    FNR = conf_mat[2, 0]/(conf_mat[2, 0] + conf_mat[2, 2])
    FPR = conf_mat[0, 2]/(conf_mat[0, 2] + conf_mat[0, 0])
    INR = conf_mat[0, 1]/(conf_mat[0, 0] + conf_mat[0, 1] + conf_mat[0, 2])
    IPR = conf_mat[2, 1]/(conf_mat[0, 2] + conf_mat[0, 1] + conf_mat[2, 2])
    ACC = (conf_mat[2, 2]+conf_mat[0, 0])/conf_mat.sum()
    return dict(TPR=TPR, TNR=TNR,
                FNR=FNR, FPR=FPR,
                INR=INR, IPR=IPR,
                ACC=ACC)
