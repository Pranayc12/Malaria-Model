#!/usr/bin/env python
# coding: utf-8

# In[2]:


train_path = r'C:\Users\asua\Documents\SKILL DEVP\DATASETS\Malaria\train'
valid_path = r'C:\Users\asua\Documents\SKILL DEVP\DATASETS\Malaria\validation'


# In[3]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[4]:


#image data generataor
train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True)


# In[5]:


#validation_datagen
valid_datagen = ImageDataGenerator(rescale=1./255)


# In[6]:


train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')
validation_generator = valid_datagen.flow_from_directory(
        valid_path,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')


# In[12]:


import tensorflow as tf

model = tf.keras.models.Sequential([
    #first_convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    #second_convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    #third_convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    #fourth_convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid') 
])


# In[13]:


model.compile(loss='binary_crossentropy',
              optimizer=tf.optimizers.Adam(),
              metrics=['accuracy'])


# In[14]:


model.summary()


# In[17]:


history = model.fit(
      train_generator,
      steps_per_epoch=25,
      epochs=50,
      validation_data=validation_generator,
      validation_steps=5,
      verbose=2)


# In[18]:


model.save("malaria_cell.h5") 


# In[19]:


import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# In[47]:


test_path  = r'C:\Users\asua\Documents\SKILL DEVP\DATASETS\Malaria\test'
#test_datagen = valid_datagen.flow_from_directory(test_path,target_size=(150, 150))
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_path,
    image_size=(150,150))#test set
pred = model.predict(test_ds)


# In[48]:


print(pred)


# In[ ]:





# In[50]:


model.evaluate(test_ds)


# In[ ]:




