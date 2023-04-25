import time
import numpy as np
import os
import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg16_preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet50_preprocess_input
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input as mobilenet_preprocess_input
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input as densenet_preprocess_input

# Define models and preprocess functions
vgg16_model = VGG16(weights='imagenet', include_top=False)
resnet50_model = ResNet50(weights='imagenet', include_top=False)
mobilenet_model = MobileNet(weights='imagenet', include_top=False)
densenet_model = DenseNet121(weights='imagenet', include_top=False)
model = tensorflow.keras.models.load_model('cnnmodel_6convlayer_final.h5')
layer_names = ['conv2d', 'max_pooling2d', 'conv2d_1', 'max_pooling2d_1', 'conv2d_2','max_pooling2d_2','conv2d_3','max_pooling2d_3','conv2d_4','max_pooling2d_4','conv2d_5','max_pooling2d_5']
new_model = Model(inputs=model.input,
                  outputs=[model.get_layer(layer_name).output for layer_name in layer_names])


vgg16_preprocess = vgg16_preprocess_input
resnet50_preprocess = resnet50_preprocess_input
mobilenet_preprocess = mobilenet_preprocess_input
densenet_preprocess = densenet_preprocess_input
new_model_preprocess = mobilenet_preprocess_input

# Load images
filenames = []
for file in os.listdir('images'):
    if file.endswith('.jpg'):
        filenames.append(os.path.join('images', file))
filenames = filenames[3000:3100]

images = np.zeros((len(filenames), 224, 224, 3))
for i, filename in enumerate(filenames):
    img = image.load_img(filename, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = new_model_preprocess(x)
    images[i] = x

# Benchmark VGG16
start_time = time.time()
features = vgg16_model.predict(vgg16_preprocess(images))
end_time = time.time()
print(f'Time taken by VGG16: {end_time - start_time:.2f} seconds')

# Benchmark ResNet50
start_time = time.time()
features = resnet50_model.predict(resnet50_preprocess(images))
end_time = time.time()
print(f'Time taken by ResNet50: {end_time - start_time:.2f} seconds')

# Benchmark MobileNet
start_time = time.time()
features = mobilenet_model.predict(mobilenet_preprocess(images))
end_time = time.time()
print(f'Time taken by MobileNet: {end_time - start_time:.2f} seconds')

# Benchmark DenseNet
start_time = time.time()
features = densenet_model.predict(densenet_preprocess(images))
end_time = time.time()
print(f'Time taken by DenseNet: {end_time - start_time:.2f} seconds')

# Benchmark CNN model
start_time = time.time()
features = new_model.predict(new_model_preprocess(images))
end_time = time.time()
print(f'Time taken by CNN model: {end_time - start_time:.2f} seconds')
