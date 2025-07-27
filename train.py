from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
from config import trainPath, batchSize, epochs, model_path
from data_utils import pathGeneration, split_data, dataGen
from model_unet import unet_model
from metrics import bce_dice_loss, precision_metric, recall_metric, f1_score_metric, dice_coef, iou_metric
import matplotlib.pyplot as plt

xTrain, yTrain = pathGeneration(trainPath)
xTrain_split, yTrain_split, xVal_split, yVal_split = split_data(xTrain, yTrain)

train_gen = dataGen(xTrain_split, yTrain_split, batchSize)
val_gen   = dataGen(xVal_split, yVal_split, batchSize)

model = unet_model()
model.compile(optimizer=Adam(1e-4),
              loss=bce_dice_loss,
              metrics=['accuracy', precision_metric, recall_metric, f1_score_metric, dice_coef, iou_metric])

earlystopper = EarlyStopping(patience=5, verbose=1)
checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    steps_per_epoch=len(xTrain_split)//batchSize,
    validation_steps=len(xVal_split)//batchSize,
    callbacks=[checkpoint,earlystopper]
)

results = model.evaluate(val_gen, steps=len(xVal_split)//batchSize, verbose=1)
print("\n--- Validation Metrics ---")
for name, val in zip(model.metrics_names, results):
    print(f"{name}: {val:.4f}")

# plt.figure(figsize=(12, 4))

# # Loss
# plt.subplot(1, 2, 1)
# plt.plot(history.history['loss'], label='Loss')
# plt.plot(history.history['val_loss'], label='Val Loss')
# plt.title('Training & Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()

# # Accuracy
# plt.subplot(1, 2, 2)
# plt.plot(history.history['accuracy'], label='Accuracy')
# plt.plot(history.history['val_accuracy'], label='Val Accuracy')
# plt.title('Training & Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()

# plt.tight_layout()
# plt.show()