from sklearn.metrics import f1_score
import numpy as np

def get_f1_score(y_true, y_pred):
    y_true = np.array(y_true).astype(int)
    if np.array(y_pred).dtype.kind in 'fc':
        y_pred_bin = (np.array(y_pred) > 0.5).astype(int)
        y_pred_final = np.where(y_pred_bin == 1,1,2)
    else:
        y_pred_final = y_pred
    return f1_score(y_true,y_pred_final, average='macro',zero_division=0)