# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 21:17:32 2021

@author: amrut
"""

import keras
from keras.models import load_model
from scipy import misc, spatial
from PIL import Image
import numpy as np
import imageio
from matplotlib.pyplot import imread


model = load_model("C:/Users/amrut/Desktop/notes 2020-2021/S6 SUBJECTS/AI/project/final_model.h5")

def predict(InputImg):
    image = imageio.imread(InputImg,pilmode="L")
    image = np.invert(image)
    image = np.resize(image,(28,28))
    image = image.reshape(1,28,28,1)

    return model.predict(image)[0].tolist().index(0)