import os
import warnings
import pickle
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.models import load_model
from scipy.misc import imsave, imread, imresize

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

warnings.filterwarnings("ignore")

path = os.path.dirname(os.path.realpath(__file__))
img_path = path + '/target_images'

mapper = pickle.load(open(path+'/mapper.p', 'rb'))
model = load_model(path+'/character_recognition_model.h5')

for f in os.listdir(img_path):
    try:
        img = imread(os.path.join(img_path,f), mode='L')
        img = np.invert(img)
        img = imresize(img,(28, 28))
        img = img.reshape(1,28,28,1)
        img = img.astype('float32')
        img = img / 255
        predictions = model.predict(img)
        print(f+":", chr(mapper[(int(np.argmax(predictions, axis=1)[0]))]))
    except(OSError):
        print('Files in the target folder must be images.')