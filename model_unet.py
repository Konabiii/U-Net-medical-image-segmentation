from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, concatenate, Input
from tensorflow.keras.models import Model
from config import img_height, img_width

def contraction_block(x, f):
    c = Conv2D(f, 3, activation='relu', padding='same')(x)
    c = Conv2D(f, 3, activation='relu', padding='same')(c)
    p = MaxPooling2D()(c)
    return c, p

def expansion_block(x, f, res):
    u = Conv2DTranspose(f, 3, strides=2, padding='same')(x)
    m = concatenate([u, res])
    c = Conv2D(f, 3, activation='relu', padding='same')(m)
    c = Conv2D(f, 3, activation='relu', padding='same')(c)
    return c

def unet_model():
    inp = Input((img_height, img_width, 3))
    c1,p1 = contraction_block(inp, 64)
    c2,p2 = contraction_block(p1, 128)
    c3,p3 = contraction_block(p2, 256)
    c4,p4 = contraction_block(p3, 512)
    c4 = Dropout(0.5)(c4)
    c5,_  = contraction_block(p4, 1024)
    e6    = expansion_block(c5, 512, c4)
    e7    = expansion_block(e6, 256, c3)
    e8    = expansion_block(e7, 128, c2)
    e9    = expansion_block(e8, 64, c1)
    out   = Conv2D(1, 1, activation='sigmoid')(e9)
    return Model(inp, out)
