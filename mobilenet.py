from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications import MobileNet
import numpy as np
import os
import pickle
from tqdm import tqdm

def extract_features(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / np.linalg.norm(result)

    return normalized_result

# List of image file paths
filenames = []

for file in os.listdir('/content/images'):
    filenames.append(os.path.join('/content/images',file))

# Pre-trained model
model = MobileNet(weights='imagenet', include_top=False, pooling='max')

feature_list = []

for file in tqdm(filenames):
    feature_list.append(extract_features(file,model))

# Save the extracted features and their corresponding filenames
pickle.dump(feature_list,open('embeddingsmobilenet.pkl','wb'))
pickle.dump(filenames,open('filenamesmobilenet.pkl','wb'))