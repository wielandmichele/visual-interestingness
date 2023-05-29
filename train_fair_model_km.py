import pandas as pd
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay

from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

from utils.gradient_reversal import *
from utils.process_functions import *

import wandb
from wandb.keras import WandbCallback

config_defaults = { 
    "data": "",
    "img_size": 150,
    "batch_size": 128,
    
    "layer_1_feature_extraction": 512,
    "layer_2_feature_extraction": 256,
    "layer_3_feature_extraction": 128,
    "activation_feature_extraction": "relu",
    
    "layer_target": 2,
    "activation_target": "softmax",
    
    "layer_1_protected": 128,
    "layer_2_protected": 128,
    "layer_3_protected": 64,
    "layer_4_protected": 64,
    "layer_5_protected": 32,
    "layer_6_protected": 32,
    "layer_protected_output": 10,
    "activation_protected": "relu",
    "activation_protected_output": "softmax",
    
    "optimizer": "adam",
    "loss": {"target_branch": "categorical_crossentropy", "protected_branch": "categorical_crossentropy"},
    "metric": ["CategoricalAccuracy","AUC"],
    "epochs": 20
}

wandb.init(
    # set the wandb project where this run will be logged
    project="Flickr_fair_model_km",
    name = 'fair_flickr_km',
    config = config_defaults
)

config = wandb.config

# Set paths
path_train_filenames_array = '/nasdata/abdu/Learning2Rank/Flickr/data/FlickrFinal/top_train_filenames.npy'
path_train_labels_array = '/nasdata/abdu/Learning2Rank/Flickr/data/FlickrFinal/top_train_labels.npy'
path_train_cl_labels_array = '/nasdata/abdu/Learning2Rank/Flickr/data/FlickrFinal/top_train_cl_labels.npy'

path_val_filenames_array = '/nasdata/abdu/Learning2Rank/Flickr/data/FlickrFinal/top_val_filenames.npy'
path_val_labels_array = '/nasdata/abdu/Learning2Rank/Flickr/data/FlickrFinal/top_val_labels.npy'
path_val_cl_labels_array = '/nasdata/abdu/Learning2Rank/Flickr/data/FlickrFinal/top_val_cl_labels.npy'

path_test_filenames_array = '/nasdata/abdu/Learning2Rank/Flickr/data/FlickrFinal/top_test_filenames.npy'
path_test_labels_array = '/nasdata/abdu/Learning2Rank/Flickr/data/FlickrFinal/top_test_labels.npy'
path_test_cl_labels_array = '/nasdata/abdu/Learning2Rank/Flickr/data/FlickrFinal/top_test_cl_labels.npy'

# Load Data
train_filenames_array = np.load(path_train_filenames_array)
train_labels_array = np.load(path_train_labels_array)
train_cl_labels_array = np.load(path_train_cl_labels_array)

val_filenames_array = np.load(path_val_filenames_array)
val_labels_array = np.load(path_val_labels_array)
val_cl_labels_array = np.load(path_val_cl_labels_array)

test_filenames_array = np.load(path_test_filenames_array)
test_labels_array = np.load(path_test_labels_array)
test_cl_labels_array = np.load(path_test_cl_labels_array)

train_filenames = train_filenames_array.tolist()
train_labels_target = tf.keras.utils.to_categorical(train_labels_array.tolist(), num_classes=2)
train_labels_protected = tf.keras.utils.to_categorical(train_cl_labels_array.tolist(), num_classes=10)

val_filenames = val_filenames_array.tolist()
val_labels_target = tf.keras.utils.to_categorical(val_labels_array.tolist(), num_classes=2)
val_labels_protected = tf.keras.utils.to_categorical(val_cl_labels_array.tolist(), num_classes=10)

test_filenames = test_filenames_array.tolist()
test_labels_target = tf.keras.utils.to_categorical(test_labels_array.tolist(), num_classes=2,dtype=int)
test_labels_protected = tf.keras.utils.to_categorical(test_cl_labels_array.tolist(), num_classes=10,dtype=int)


train_data_gen = tf.data.Dataset.from_generator(
    lambda: data_generator(train_filenames, train_labels_target, train_labels_protected),
    output_types=(tf.string, {'target_branch': tf.int32, 'protected_branch': tf.int32}),
    output_shapes=((), {'target_branch': (2,), 'protected_branch': (10,)}))

train_data_map = train_data_gen.map(lambda x, y: (process_img(x), y))
train_data_batch = train_data_map.batch(config.batch_size)
X_train = train_data_batch.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

val_data_gen = tf.data.Dataset.from_generator(
    lambda: data_generator(val_filenames, val_labels_target, val_labels_protected),
    output_types=(tf.string, {'target_branch': tf.int32, 'protected_branch': tf.int32}),
    output_shapes=((), {'target_branch': (2,), 'protected_branch': (10,)}))

val_data_map = val_data_gen.map(lambda x, y: (process_img(x), y))
val_data_batch = val_data_map.batch(config.batch_size)
X_val = val_data_batch.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

test_data_gen = tf.data.Dataset.from_generator(
    lambda: data_generator(test_filenames, test_labels_target, test_labels_protected),
    output_types=(tf.string, {'target_branch': tf.int32, 'protected_branch': tf.int32}),
    output_shapes=((), {'target_branch': (2,), 'protected_branch': (10,)}))

test_data_map = test_data_gen.map(lambda x, y: (process_img(x), y))
test_data_batch = test_data_map.batch(config.batch_size)
X_test = test_data_batch.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

def build_fair_model() -> tf.keras.models.Sequential:
    #initialise base model
    IMG_SHAPE = (config.img_size, config.img_size, 3)
    base_model = tf.keras.applications.Xception(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    base_model.trainable = False

    #define feature extractor
    feature_extractor = tf.keras.Sequential(name = "feature_extractor")
    feature_extractor.add(tf.keras.layers.Lambda(tf.keras.applications.xception.preprocess_input, input_shape=IMG_SHAPE))
    feature_extractor.add(base_model)
    feature_extractor.add(tf.keras.layers.Dense(config.layer_1_feature_extraction, activation = config.activation_feature_extraction))
    feature_extractor.add(tf.keras.layers.Dense(config.layer_2_feature_extraction, activation = config.activation_feature_extraction))
    feature_extractor.add(tf.keras.layers.Dense(config.layer_3_feature_extraction, activation = config.activation_feature_extraction))
    feature_extractor.add(tf.keras.layers.GlobalAveragePooling2D())

    #define target branch
    target_branch = tf.keras.Sequential(name = "target_branch")
    target_branch.add(tf.keras.layers.Dense(config.layer_target, activation = config.activation_target))

    #define protected branch
    protected_branch = tf.keras.Sequential(name = "protected_branch")
    protected_branch.add(GradientReversal())
    protected_branch.add(tf.keras.layers.Dense(config.layer_1_protected, activation = config.activation_protected))
    protected_branch.add(tf.keras.layers.Dense(config.layer_2_protected, activation = config.activation_protected))
    protected_branch.add(tf.keras.layers.Dense(config.layer_3_protected, activation = config.activation_protected))
    protected_branch.add(tf.keras.layers.Dense(config.layer_4_protected, activation = config.activation_protected))
    protected_branch.add(tf.keras.layers.Dense(config.layer_5_protected, activation = config.activation_protected))
    protected_branch.add(tf.keras.layers.Dense(config.layer_6_protected, activation = config.activation_protected))
    protected_branch.add(tf.keras.layers.Dense(config.layer_protected_output, activation = config.activation_protected_output))
    
    #combine everything
    input_layer = Input(shape=IMG_SHAPE)
    extracted_features = feature_extractor(input_layer)
    target_output = target_branch(extracted_features)
    protected_output = protected_branch(extracted_features)

    fair_model = tf.keras.models.Model(inputs = input_layer, outputs=[target_output, protected_output])
    fair_model.summary()
    return fair_model

fair_model = build_fair_model()

wandb.init(
    # set the wandb project where this run will be logged
    project="Flickr_fair_model_km",
    name = 'fair_flickr_km',
    config = config_defaults
)

config = wandb.config

fair_model.compile(optimizer=config.optimizer, 
      loss = config.loss, 
      loss_weights = [1, 1],
      metrics=config['metric']
)

#fit the model
hist = fair_model.fit(
    X_train, 
    epochs = config.epochs,
    validation_data = X_val, 
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=20),
        WandbCallback()
    ]
)
wandb.finish()

fair_model.save('./models/fair_model_km.h5')

y_pred = fair_model.predict(X_test)
y_pred_target = y_pred[0]
y_pred_protected = y_pred[1]
np.save("./pred/y_pred_target_km.npy",y_pred_target)
np.save("./pred/y_pred_protected_km.npy",y_pred_protected)