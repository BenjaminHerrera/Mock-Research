# Imports
from tensorflow.python.keras.regularizers import l2
from tensorflow.keras.callbacks import TensorBoard
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import os

# Variables
NAME = "model-002"

# Prevents CUBLAS_STATUS_ALLOC_FAILED errors
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
tf.compat.v1.keras.backend.set_session(session)

# Temporarily adds Graphviz to PATH
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'

# Initializes checkpoint and tensorboard callbacks
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint("checkpoints/" + NAME + "/cp-{epoch:01d}", save_best_only=True)
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

# Loads metadata dataset
metadata = pd.read_csv("data/HAM10000_metadata.csv")

# Removes Lesion ID column and dx_type
metadata = metadata.drop(["lesion_id"], axis=1)
metadata = metadata.drop(["dx_type"], axis=1)

# Loads translation key into memory
translation = json.loads(open("data/translation.json").read())

# Translates all non-numerical values into float64 values
metadata = metadata.replace(translation["localization"])
metadata = metadata.replace(translation["sex"])
metadata = metadata.replace(translation["dx"])

# "Shaves" off some data to make the model generalize better and to optimize memory usage
# metadata = metadata.drop(metadata[metadata.dx == 5.0].iloc[:5300].index)

# Separates Image IDs and Labels to different variables
image_id = metadata.pop("image_id")
label = metadata.pop("dx")

# Initializes MinMax scaler for metadata
metadata_scaling = MinMaxScaler(feature_range=(0, 1))

# Normalizes metadata
metadata = metadata_scaling.fit_transform(metadata)

# Converts metadata and changes dtype of label
metadata = np.array(metadata)

# Categorical Formatting
new_label = tf.keras.utils.to_categorical(label, num_classes=7)

# Initializes image features set
image = np.zeros((len(image_id), 150, 200, 3), dtype=np.float32)

# Iterates through extracted image IDs list, resizes images (originally 600 x 450), and places RGB values into image features set
for i in tqdm(image_id):
    image[int(np.where(image_id == i)[0])] = np.array(Image.open("data/HAM10000_images/{}.jpg".format(i)).resize((200, 150)))

# Normalizes images
image = image / 255

# Convolutional Branch
image_input = tf.keras.layers.Input(shape=(150, 200, 3))

conv2D_1 = tf.keras.layers.Conv2D(16, kernel_size=(2, 2), activation='relu', kernel_regularizer=l2(0.01))(image_input)
batch_norm_1 = tf.keras.layers.BatchNormalization()(conv2D_1)
conv2D_2 = tf.keras.layers.Conv2D(16, kernel_size=(2, 2), activation='relu', kernel_regularizer=l2(0.01))(batch_norm_1)
batch_norm_2 = tf.keras.layers.BatchNormalization()(conv2D_2)
max_pool_1 = tf.keras.layers.MaxPool2D((2, 2))(batch_norm_2)
batch_norm_3 = tf.keras.layers.BatchNormalization()(max_pool_1)
dropout_1 = tf.keras.layers.Dropout(0.25)(batch_norm_3)
batch_norm_4 = tf.keras.layers.BatchNormalization()(dropout_1)

conv2D_3 = tf.keras.layers.Conv2D(32, kernel_size=(2, 2), activation='relu', kernel_regularizer=l2(0.01))(batch_norm_4)
batch_norm_5 = tf.keras.layers.BatchNormalization()(conv2D_3)
conv2D_4 = tf.keras.layers.Conv2D(32, kernel_size=(2, 2), activation='relu', kernel_regularizer=l2(0.01))(batch_norm_5)
batch_norm_6 = tf.keras.layers.BatchNormalization()(conv2D_4)
max_pool_2 = tf.keras.layers.MaxPool2D((2, 2))(batch_norm_6)
batch_norm_7 = tf.keras.layers.BatchNormalization()(max_pool_2)
dropout_2 = tf.keras.layers.Dropout(0.25)(batch_norm_7)
batch_norm_8 = tf.keras.layers.BatchNormalization()(dropout_2)

conv2D_5 = tf.keras.layers.Conv2D(64, kernel_size=(2, 2), activation='relu', kernel_regularizer=l2(0.01))(batch_norm_8)
batch_norm_9 = tf.keras.layers.BatchNormalization()(conv2D_5)
conv2D_6 = tf.keras.layers.Conv2D(64, kernel_size=(2, 2), activation='relu', kernel_regularizer=l2(0.01))(batch_norm_9)
batch_norm_10 = tf.keras.layers.BatchNormalization()(conv2D_6)
max_pool_3 = tf.keras.layers.MaxPool2D((2, 2))(batch_norm_10)
batch_norm_11 = tf.keras.layers.BatchNormalization()(max_pool_3)
dropout_3 = tf.keras.layers.Dropout(0.25)(batch_norm_11)
batch_norm_12 = tf.keras.layers.BatchNormalization()(dropout_3)

conv2D_7 = tf.keras.layers.Conv2D(128, kernel_size=(2, 2), activation='relu', kernel_regularizer=l2(0.01))(batch_norm_12)
batch_norm_13 = tf.keras.layers.BatchNormalization()(conv2D_7)
conv2D_8 = tf.keras.layers.Conv2D(128, kernel_size=(2, 2), activation='relu', kernel_regularizer=l2(0.01))(batch_norm_13)
batch_norm_14 = tf.keras.layers.BatchNormalization()(conv2D_8)
max_pool_4 = tf.keras.layers.MaxPool2D((2, 2))(batch_norm_14)
batch_norm_15 = tf.keras.layers.BatchNormalization()(max_pool_4)
dropout_4 = tf.keras.layers.Dropout(0.25)(batch_norm_15)
batch_norm_16 = tf.keras.layers.BatchNormalization()(dropout_4)

flatten = tf.keras.layers.Flatten()(batch_norm_16)

# Metadata Branch
metadata_input = tf.keras.layers.Input(shape=(3,))

# Concatenated Branch
concat = tf.keras.layers.Concatenate()([flatten, metadata_input])
hidden_1 = tf.keras.layers.Dense(4096, activation='relu', kernel_regularizer=l2(0.01))(concat)
batch_norm_17 = tf.keras.layers.BatchNormalization()(hidden_1)
dropout_5 = tf.keras.layers.Dropout(0.25)(batch_norm_17)
batch_norm_18 = tf.keras.layers.BatchNormalization()(dropout_5)
hidden_2 = tf.keras.layers.Dense(4096, activation='relu', kernel_regularizer=l2(0.01))(batch_norm_18)
batch_norm_19 = tf.keras.layers.BatchNormalization()(hidden_2)
dropout_6 = tf.keras.layers.Dropout(0.25)(batch_norm_19)
batch_norm_20 = tf.keras.layers.BatchNormalization()(dropout_6)

# Model Creation
output = tf.keras.layers.Dense(7, activation='softmax')(batch_norm_20)
model = tf.keras.Model(inputs=[image_input, metadata_input], outputs=[output])

# # Prints structure of model
plot_model(model, to_file='./notes/architecture/{}_plot.png'.format(NAME), show_shapes=True, show_layer_names=True)

# Model Compilation
model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=['accuracy'])

# Model Training
model.fit(x=[image, metadata], y=new_label, batch_size=16, epochs=200, validation_split=0.2, callbacks=[tensorboard, checkpoint_callback])