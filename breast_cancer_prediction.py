# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 11:52:55 2022

@author: ANEH
"""

import pandas as pd
import numpy as np
import sklearn
import tensorflow as tf

#1. Read data
bcw_data = pd.read_csv(r"C:\Users\ANEH\Documents\Deep Learning Class\TensorFlow Deep Learning\Datasets\bcwisconsin\bcw_data.csv")

#%%
#2. Drop unnecessary column
bcw_data = bcw_data.drop(['id','Unnamed: 32'], axis=1)

#%%
#3. Split data into features and label
bcw_features = bcw_data.copy()
bcw_labels = bcw_features.pop('diagnosis')

#%%
#4. One hot label
bcw_label_OH = pd.get_dummies(bcw_labels)

features_np = np.array(bcw_features)
labels_np = np.array(bcw_label_OH)

#%%
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing

SEED= 12345
x_train, x_test, y_train, y_test = train_test_split(features_np, labels_np, 
                                                    test_size=0.2, random_state=SEED)
x_train = np.array(x_train)
x_test = np.array(x_test)

#%%
standardizer = preprocessing.StandardScaler()
standardizer.fit(x_train)
x_train = standardizer.transform(x_train)
x_test = standardizer.transform(x_test)

#%%

nClass = len(np.unique(y_test))
inputs = tf.keras.Input(shape=(x_train.shape[-1],))
dense = tf.keras.layers.Dense(64,activation='relu')
x = dense(inputs)
dense = tf.keras.layers.Dense(32,activation='relu')
x = dense(x)
dense = tf.keras.layers.Dense(16,activation='relu')
x = dense(x)
outputs = tf.keras.layers.Dense(nClass,activation='softmax')(x)
model = tf.keras.Model(inputs=inputs,outputs=outputs,name='breast_cancer_model')
model.summary()

#%%
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=32,epochs=100)