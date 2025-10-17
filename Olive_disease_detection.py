#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set_style('whitegrid')

import tensorflow as tf 
from tensorflow.keras import layers 
from tensorflow.keras.models import Sequential,Model
import os 


# In[2]:


data_dir=os.path.join("C:\\Users\\ASUS TUF F15\\OneDrive\\Desktop\\dataset")


# In[3]:


data_dir


# In[4]:


os.listdir(data_dir)


# In[5]:


os.listdir("C:\\Users\\ASUS TUF F15\\OneDrive\\Desktop\\dataset\\train")


# In[6]:


len(os.listdir("C:\\Users\\ASUS TUF F15\\OneDrive\\Desktop\\dataset\\train\\olive_peacock_spot"))


# In[7]:


import PIL 


# In[8]:


PIL.Image.open("C:\\Users\\ASUS TUF F15\\OneDrive\\Desktop\\dataset\\train\\olive_peacock_spot\\A440.jpg")


# In[9]:


train_dir=os.path.join(data_dir,'train')


# In[10]:


train_dir


# In[11]:


test_dir=os.path.join(data_dir,'test')


# In[12]:


test_dir


# In[13]:


#Dividing this directories into datasets 


# In[14]:


train_ds=tf.keras.utils.image_dataset_from_directory(train_dir,image_size=(224,224),batch_size=60)


# In[15]:


test_ds=tf.keras.utils.image_dataset_from_directory(test_dir,image_size=(224,224),batch_size=60)


# In[16]:


class_n=train_ds.class_names
class_n


# In[17]:


plt.figure(figsize=(10,10))
for image,label in train_ds.take(1): 
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(image[i].numpy().astype('uint8'))
        plt.title(class_n[label[i]])
        plt.axis('off')


# In[18]:


image,label=next(iter(train_ds))
print(image.numpy())
print(label)


# In[19]:


base_model=tf.keras.applications.MobileNetV3Large(include_top=False,weights='imagenet',pooling='avg')


# In[20]:


base_model.summary()


# In[21]:


base_model.trainable=False


# In[22]:


base_model.summary()


# In[23]:


X=layers.Dense(32,'relu')(base_model.output)
X=layers.Dropout(0.3)(X)
X=layers.Dense(64,'relu')(X)

#Output layer 
out=layers.Dense(3,'softmax')(X)


# In[24]:


model=Model(inputs=base_model.input,outputs=out)


# In[25]:


model.summary()


# In[26]:


model.compile(loss='SparseCategoricalCrossentropy',optimizer='adam',metrics=['accuracy'])


# In[27]:


history=model.fit(train_ds,validation_data=test_ds,epochs=10)


# In[28]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(10)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# In[37]:


from keras.utils import load_img,img_to_array


# In[79]:


#olive_peacock_spot
img1=load_img('A-13.JPG',target_size=(150,150))


# In[61]:


img1


# In[62]:


img1_ar=img_to_array(img1).reshape(1,150,150,3)


# In[63]:


img1_ar


# In[64]:


pred1=class_n[np.argmax(model.predict(img1_ar))]


# In[65]:


pred1


# In[78]:


#aculus_olearius
img2=load_img('128.jpg',target_size=(150,150))


# In[68]:


img2


# In[71]:


img2_ar=img_to_array(img2).reshape(1,150,150,3)


# In[72]:


img2_ar


# In[75]:


pred2=class_n[np.argmax(model.predict(img2_ar))]


# In[76]:


pred2


# In[77]:


model.save('olive_model.h5')


# In[ ]:




