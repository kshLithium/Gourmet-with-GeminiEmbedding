import numpy as np


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_true = y_true != 0
    if np.sum(non_zero_true) == 0:
        return 0.0  # 모든 y_true가 0인 경우 MAPE는 0으로 처리
    return (
        np.mean(
            np.abs(
                (y_true[non_zero_true] - y_pred[non_zero_true]) / y_true[non_zero_true]
            )
        )
        * 100
    )
