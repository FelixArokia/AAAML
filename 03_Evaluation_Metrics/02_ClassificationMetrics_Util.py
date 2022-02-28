from sklearn import metrics
import pandas as pd
import numpy as np


def confusion_matrix_util(y_true, y_predictions):
    cm_df = pd.DataFrame()
    for thresh in np.arange(0, 1, 0.001):
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_predictions).ravel()
        cm_df = cm_df.append({
            'thresh': thresh,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp
        }, ignore_index=True)

    cm_df['recall'] = cm_df['tp'] / (cm_df['tp'] + cm_df['fn'])
    cm_df['precision'] = cm_df['tp'] / (cm_df['tp'] + cm_df['fp'])
    cm_df['f1'] = 2 * cm_df['recall'] * cm_df['precision'] / (cm_df['recall'] + cm_df['precision'])
    return cm_df


cm_df = confusion_matrix_util(y_true=[0, 1, 0, 0, 0, 1, 1, 1, 0, 1], y_predictions=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
print(f'cm_df : {cm_df.tail(1000)}')
