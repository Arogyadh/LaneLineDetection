import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.densenet import DenseNet121,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from tensorflow.keras.models import Model



feature_list = np.array(pickle.load(open('embeddingscnnmodelfinal(6layer).pkl','rb')))
filenames = pickle.load(open('filenamescnnmodelfinal(6layer).pkl','rb'))

model = tensorflow.keras.models.load_model('cnnmodel_6convlayer_final.h5')
layer_names = ['conv2d', 'max_pooling2d', 'conv2d_1', 'max_pooling2d_1', 'conv2d_2','max_pooling2d_2','conv2d_3','max_pooling2d_3','conv2d_4','max_pooling2d_4','conv2d_5','max_pooling2d_5']

intermediate_layer_model = Model(inputs=model.input,
                                 outputs=[model.get_layer(layer_name).output for layer_name in layer_names])


st.title('Reverse Image Search')

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0


def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices


uploaded_file = st.file_uploader("choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        #display the file
        display_image = Image.open(uploaded_file)
        st.image(display_image)

        features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
        #st.text(features)
        indices = recommend(features,feature_list)

        col1,col2,col3,col4,col5 = st.columns(5)

        with col1:
            st.image(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
        with col4:
            st.image(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4]])
    else:
        st.header("Some error occoured")