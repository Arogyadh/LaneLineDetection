
import numpy as np
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.vgg16 import preprocess_input
from numpy.linalg import norm
from tensorflow.keras.models import Model
from PIL import Image
import os
from tqdm import tqdm
import pickle

model = tensorflow.keras.models.load_model('cnnmodel_6convlayer_final.h5')
print(model.summary())

layer_names = ['conv2d', 'max_pooling2d', 'conv2d_1', 'max_pooling2d_1', 'conv2d_2','max_pooling2d_2','conv2d_3','max_pooling2d_3','conv2d_4','max_pooling2d_4','conv2d_5','max_pooling2d_5']
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=[model.get_layer(layer_name).output for layer_name in layer_names])


def extract_features(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

filenames = []

for file in os.listdir('images_model'):
    filenames.append(os.path.join('images_model',file))

feature_list = []

for file in tqdm(filenames):
    feature_list.append(extract_features(file,model))

pickle.dump(feature_list,open('embeddingscnnmodelfinal(6layer).pkl','wb'))
pickle.dump(filenames,open('filenamescnnmodelfinal(6layer).pkl','wb'))