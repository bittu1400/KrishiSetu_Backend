#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries
# 

# In[2]:


import tensorflow as tf 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# # Data Preprocessing

# ## Training Image Preprocessing

# In[5]:


training_set = tf.keras.utils.image_dataset_from_directory(
    'Detector/train',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)


# ## Validation Image Preprocessing

# In[4]:


validation_set = tf.keras.utils.image_dataset_from_directory(
    'Detector/valid',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)


# In[84]:


training_set


# In[85]:


for x,y in training_set:
    print(x,x.shape)
    print(y,y.shape)
    break


# ### To avoid Overshooting
# 1. Choose small learning rate default 0.001 we are taking 0.0001
# 2. There may be chance of Underfitting, so increase number of neuron
# 3. Add more Convolution layer to extract more feature from images there may be possibility that model unable to capture relevant feature or model is confusing due to lack of feature so feed with more feature

# ## Building Model

# In[86]:


from tf_keras.models import Sequential
from tf_keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout

# In[87]:


model = Sequential()


# ## Building Convolution Layer

# In[88]:


model.add(Conv2D(filters=32,kernel_size=3,padding='same',activation='relu',input_shape=[128,128,3]))
model.add(Conv2D(filters=32,kernel_size=3,activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))


# In[89]:


model.add(Conv2D(filters=64,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=64,kernel_size=3,activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))


# In[90]:


model.add(Conv2D(filters=128,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=128,kernel_size=3,activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))


# In[91]:


model.add(Conv2D(filters=256,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=256,kernel_size=3,activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))


# In[92]:


model.add(Conv2D(filters=512,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=512,kernel_size=3,activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))


# In[93]:


model.add(Dropout(0.25)) # To avoid overfitting


# In[94]:


model.add(Flatten())


# In[95]:


model.add(Dense(units=1500,activation='relu'))


# In[96]:


model.add(Dropout(0.4))


# In[97]:


#Output Layer
model.add(Dense(units=38,activation='softmax'))


# ## Compiling Model

# In[98]:

# CORRECT - use tf_keras optimizer:
from tf_keras.optimizers import Adam

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[99]:


model.summary()


# ### Model Training

# In[101]:


training_history = model.fit(x=training_set,validation_data=validation_set,epochs=10)


# In[ ]:





# ## Model Evaluation
# 

# In[102]:


#Model Evaluation on Training set
train_loss,train_acc = model.evaluate(training_set)


# In[103]:


print(train_loss,train_acc)


# In[104]:


#Model on Validation set
val_loss,val_acc = model.evaluate(validation_set)


# In[105]:


print(val_loss,val_acc)


# In[ ]:





# ## Saving Model

# In[110]:


model.save("trained_model.h5")


# In[111]:


model.save("Trained_model.keras")


# In[112]:


training_history.history


# In[114]:


#Recording Historyin json
import json
with open("training_hist.json","w") as f:
    json.dump(training_history.history,f)


# In[119]:


training_history.history['accuracy']


# ### Accuracy Visualization

# In[123]:


epochs = [i for i in range(1,11)]
plt.plot(epochs,training_history.history['accuracy'],color='red',label='Training Accuracy')
plt.plot(epochs,training_history.history['val_accuracy'],color='green',label='Validation Accuracy')
plt.xlabel('No. of Epochs')
plt.ylabel("Accuracy Result")
plt.title('Visualization of Accuracy Result')
plt.legend()
plt.show()


# In[ ]:





# ### Some other metrics for model evaluation
# 

# In[124]:


class_name = validation_set.class_names
class_name


# In[ ]:





# In[125]:


test_set = tf.keras.utils.image_dataset_from_directory(
    'valid',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=False,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)


# In[127]:


y_pred = model.predict(test_set)
y_pred,y_pred.shape


# In[131]:


predicted_categories = tf.argmax(y_pred,axis=1)
predicted_categories


# In[134]:


true_categories = tf.concat([y for x,y in test_set], axis=0)
true_categories


# In[140]:


Y_true = tf.argmax(true_categories,axis=1)
Y_true


# In[ ]:





# In[148]:


from sklearn.metrics import classification_report,confusion_matrix


# In[147]:


print(classification_report(Y_true,predicted_categories,target_names=class_name))


# In[ ]:





# In[153]:


cm = confusion_matrix(Y_true,predicted_categories)
cm


# ### Confusion Matrix Visualization
# 
# 

# In[170]:


plt.figure(figsize=(40,40))
sns.heatmap(cm,annot=True,annot_kws={'size':10})
plt.xlabel("Predicted Class",fontsize=20)
plt.ylabel("Actual Class",fontsize=20 )
plt.title("Plant Disease Prediction Confusion Matrix", fontsize=25)
plt.show()


# In[ ]:





# In[ ]:




