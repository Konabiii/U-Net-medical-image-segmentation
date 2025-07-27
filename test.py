import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from skimage.io import imshow
from data_utils import preprocess_image, mask_formation, path_generation
from metrics import dice_coef, iou_metric, precision_metric, recall_metric, f1_score_metric

model_path = 'C:\\Users\\admin\\Documents\\unet\\model\\unet.h5'
model = load_model(
    model_path,
    custom_objects={
        'dice_coef': dice_coef,
        'iou_metric': iou_metric,
        'precision_metric': precision_metric,
        'recall_metric': recall_metric,
        'f1_score_metric': f1_score_metric
    }
)

train_path = 'C:\\Users\\admin\\Documents\\unet\\stage1_train'
xTrain, yTrain = path_generation(train_path)

ix = 100
img = preprocess_image(xTrain[ix])
mask = mask_formation(yTrain[ix])
preds = model.predict(np.expand_dims(img, axis=0))
preds_threshold = (preds > 0.5).astype(np.uint8)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
imshow(img)
plt.title("Original Image")

plt.subplot(1, 3, 2)
imshow(np.squeeze(mask))
plt.title("Ground Truth Mask")

plt.subplot(1, 3, 3)
imshow(np.squeeze(preds_threshold))
plt.title("Predicted Mask")

plt.tight_layout()
plt.show()
