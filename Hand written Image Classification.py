#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf 
from tensorflow import keras
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[5]:


(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()


# In[6]:


len(X_train)


# In[7]:


len(X_test)


# In[8]:


X_train[0].shape


# In[9]:


X_train[0]


# In[10]:


plt.matshow(X_train[2])


# In[11]:


y_train[2]


# In[12]:


y_train[:5]


# In[13]:


X_train.shape


# In[14]:


X_train= X_train/255
X_test= X_test/255


# In[15]:


X_train_flattened = X_train.reshape(len(X_train),28*28)
X_test_flattened = X_test.reshape(len(X_test),28*28)


# In[16]:


X_train_flattened.shape


# In[17]:


X_test_flattened.shape


# In[18]:


X_train_flattened[0]


# In[19]:


model = keras.Sequential([
    keras.layers.Dense(10,input_shape=(784,),activation ='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train_flattened,y_train,epochs=5)


# In[20]:


model.evaluate(X_test_flattened,y_test)


# In[21]:


plt.matshow(X_test[0])


# In[22]:


y_predicted = model.predict(X_test_flattened)
y_predicted[0]


# In[23]:


np.argmax(y_predicted[0])


# In[28]:


y_predicted_labels=[np.argmax(i) for i in y_predicted]
y_predicted_labels[:5]


# In[25]:


y_test[:5]


# In[33]:


cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)
cm


# In[35]:


import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True,fmt='d')
plt.xlabel('predicted')
plt.ylabel('Truth')


# In[37]:


model = keras.Sequential([
    keras.layers.Dense(100,input_shape=(784,),activation ='relu'),
    keras.layers.Dense(10,activation ='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train_flattened,y_train,epochs=5)


# In[38]:


model.evaluate(X_test_flattened,y_test)


# In[39]:


import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True,fmt='d')
plt.xlabel('predicted')
plt.ylabel('Truth')

