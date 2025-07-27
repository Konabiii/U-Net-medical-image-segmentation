import os

trainPath = 'C:\\Users\\admin\\Documents\\unet\\stage1_train'

img_height, img_width = 128, 128

batchSize = 16
epochs = 50

os.makedirs('C:\\Users\\admin\\Documents\\unet\\model', exist_ok=True)
model_path = 'C:\\Users\\admin\\Documents\\unet\\model\\unet.h5'
