from tensorflow.keras import backend as K

def dice_coef(y_true, y_pred):
    y_true_f = K.cast(K.flatten(y_true), 'float32')
    y_pred_f = K.cast(K.flatten(y_pred), 'float32')
    inter = K.sum(y_true_f * y_pred_f)
    return (2.*inter + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

def iou_metric(y_true, y_pred):
    y_true_f = K.cast(K.flatten(y_true), 'float32')
    y_pred_f = K.cast(K.flatten(y_pred), 'float32')
    inter = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - inter
    return (inter + K.epsilon()) / (union + K.epsilon())

def precision_metric(y_true, y_pred):
    y_true_f = K.cast(K.flatten(y_true), 'float32')
    y_pred_f = K.cast(K.flatten(y_pred), 'float32')
    TP = K.sum(y_true_f * y_pred_f)
    FP = K.sum((1-y_true_f) * y_pred_f)
    return TP / (TP + FP + K.epsilon())

def recall_metric(y_true, y_pred):
    y_true_f = K.cast(K.flatten(y_true), 'float32')
    y_pred_f = K.cast(K.flatten(y_pred), 'float32')
    TP = K.sum(y_true_f * y_pred_f)
    FN = K.sum(y_true_f * (1-y_pred_f))
    return TP / (TP + FN + K.epsilon())

def f1_score_metric(y_true, y_pred):
    precision = precision_metric(y_true, y_pred)
    recall = recall_metric(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall + K.epsilon())

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    return K.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
