import tensorflow as tf 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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

training_set

for x,y in training_set:
    print(x,x.shape)
    print(y,y.shape)
    break


from tf_keras.models import Sequential
from tf_keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout

model = Sequential()

model.add(Conv2D(filters=32,kernel_size=3,padding='same',activation='relu',input_shape=[128,128,3]))
model.add(Conv2D(filters=32,kernel_size=3,activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))

model.add(Conv2D(filters=64,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=64,kernel_size=3,activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))

model.add(Conv2D(filters=128,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=128,kernel_size=3,activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))

model.add(Conv2D(filters=256,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=256,kernel_size=3,activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))

model.add(Conv2D(filters=512,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=512,kernel_size=3,activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))


model.add(Dropout(0.25)) # To avoid overfitting

model.add(Flatten())

model.add(Dense(units=1500,activation='relu'))

model.add(Dropout(0.4))

#Output Layer
model.add(Dense(units=38,activation='softmax'))

# CORRECT - use tf_keras optimizer:
from tf_keras.optimizers import Adam

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.summary()

training_history = model.fit(x=training_set,validation_data=validation_set,epochs=10)


#Model Evaluation on Training set
train_loss,train_acc = model.evaluate(training_set)


print(train_loss,train_acc)

#Model on Validation set
val_loss,val_acc = model.evaluate(validation_set)


print(val_loss,val_acc)



model.save("trained_model.h5")


model.save("Trained_model.keras")


# In[112]:


training_history.history


# In[114]:


#Recording Historyin json
import json
with open("training_hist.json","w") as f:
    json.dump(training_history.history,f)



training_history.history['accuracy']


epochs = [i for i in range(1,11)]
plt.plot(epochs,training_history.history['accuracy'],color='red',label='Training Accuracy')
plt.plot(epochs,training_history.history['val_accuracy'],color='green',label='Validation Accuracy')
plt.xlabel('No. of Epochs')
plt.ylabel("Accuracy Result")
plt.title('Visualization of Accuracy Result')
plt.legend()
plt.show()


class_name = validation_set.class_names
class_name


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

y_pred = model.predict(test_set)
y_pred,y_pred.shape


predicted_categories = tf.argmax(y_pred,axis=1)
predicted_categories


true_categories = tf.concat([y for x,y in test_set], axis=0)
true_categories


Y_true = tf.argmax(true_categories,axis=1)
Y_true


from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(Y_true,predicted_categories,target_names=class_name))

cm = confusion_matrix(Y_true,predicted_categories)
cm


plt.figure(figsize=(40,40))
sns.heatmap(cm,annot=True,annot_kws={'size':10})
plt.xlabel("Predicted Class",fontsize=20)
plt.ylabel("Actual Class",fontsize=20 )
plt.title("Plant Disease Prediction Confusion Matrix", fontsize=25)
plt.show()


# Help me make this code vs code ready.

# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# from sklearn.metrics import classification_report, confusion_matrix
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import shutil
# from pathlib import Path
# # Define constants
# IMG_WIDTH, IMG_HEIGHT = 224, 224
# BATCH_SIZE = 16 # Reduced for better generalization
# NUM_CLASSES = 9
# VALIDATION_SPLIT = 0.2 # 20% of training data for validation
# # Define directories
# train_dir = 'images/train'
# test_dir = 'images/test'
# validation_dir = 'images/validation'
# import os
# import random
# # Path to your training directory
# train_dir = 'images/train'
# MAX_IMAGES_PER_CLASS = 500
# # Loop over each class folder
# for class_name in os.listdir(train_dir):
# class_path = os.path.join(train_dir, class_name)
# if not os.path.isdir(class_path):
# continue
# # List all image files
# images = [f for f in os.listdir(class_path)
# if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
# # Only trim if more than MAX_IMAGES_PER_CLASS
# if len(images) > MAX_IMAGES_PER_CLASS:
# print(f"⚠️ {class_name} has {len(images)} images. Trimming to {MAX_IMAGES_PER_CLASS}...")
# random.seed(42) # for reproducibility
# images_to_remove = random.sample(images, len(images) - MAX_IMAGES_PER_CLASS)
# for img in images_to_remove:
# os.remove(os.path.join(class_path, img))
# print(f" ✓ {len(images_to_remove)} images deleted. Remaining: {MAX_IMAGES_PER_CLASS}")
# else:
# print(f" ✓ {class_name} has {len(images)} images. No deletion needed.")
# # ============================================
# # STEP 1: Create validation split from training data
# # ============================================
# def create_validation_split(train_dir, validation_dir, split_ratio=0.2):
# print("📂 Creating validation split...")
# os.makedirs(validation_dir, exist_ok=True)
# classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
# for class_name in classes:
# class_train_path = os.path.join(train_dir, class_name)
# class_val_path = os.path.join(validation_dir, class_name)
# os.makedirs(class_val_path, exist_ok=True)
# # Skip if validation folder already has images
# if len(os.listdir(class_val_path)) > 0:
# print(f"⚠️ Skipping {class_name} (already has validation data).")
# continue
# images = [f for f in os.listdir(class_train_path)
# if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
# num_val = int(len(images) * split_ratio)
# if num_val == 0:
# print(f"⚠️ {class_name} too few images to split, skipping.")
# continue
# np.random.seed(42)
# np.random.shuffle(images)
# val_images = images[:num_val]
# for img in val_images:
# src = os.path.join(class_train_path, img)
# dst = os.path.join(class_val_path, img)
# shutil.move(src, dst)
# print(f" ✓ {class_name}: {len(images)} total → {len(images)-num_val} train, {num_val} validation")
# print("✅ Validation split created successfully!\n")
# # Create validation split
# create_validation_split(train_dir, validation_dir, VALIDATION_SPLIT)
# # ============================================
# # STEP 2: Calculate class weights for imbalanced data
# # ============================================
# def calculate_class_weights(directory):
# """
#     Calculate class weights to handle imbalanced datasets
#     """
# classes = os.listdir(directory)
# class_counts = {}
# for class_name in classes:
# class_path = os.path.join(directory, class_name)
# if os.path.isdir(class_path):
# count = len([f for f in os.listdir(class_path)
# if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
# class_counts[class_name] = count
# total = sum(class_counts.values())
# class_weights = {i: total / (len(class_counts) * count)
# for i, (name, count) in enumerate(class_counts.items())}
# print("📊 Class distribution:")
# for name, count in class_counts.items():
# print(f" {name}: {count} images")
# print(f"\n⚖️ Class weights: {class_weights}\n")
# return class_weights
# class_weights = calculate_class_weights(train_dir)
# # ============================================
# # STEP 3: Enhanced Data Augmentation
# # ============================================
# train_datagen = ImageDataGenerator(
# rescale=1./255,
# rotation_range=40,
# width_shift_range=0.25,
# height_shift_range=0.25,
# shear_range=0.2,
# zoom_range=0.25,
# horizontal_flip=True,
# vertical_flip=True, # Plants can appear at various angles
# brightness_range=[0.8, 1.2], # Lighting variations
# fill_mode='nearest'
# )
# validation_datagen = ImageDataGenerator(rescale=1./255)
# test_datagen = ImageDataGenerator(rescale=1./255)
# # Generators
# train_generator = train_datagen.flow_from_directory(
# train_dir,
# target_size=(IMG_WIDTH, IMG_HEIGHT),
# batch_size=BATCH_SIZE,
# class_mode='categorical',
# shuffle=True
# )
# validation_generator = validation_datagen.flow_from_directory(
# validation_dir,
# target_size=(IMG_WIDTH, IMG_HEIGHT),
# batch_size=BATCH_SIZE,
# class_mode='categorical',
# shuffle=False
# )
# test_generator = test_datagen.flow_from_directory(
# test_dir,
# target_size=(IMG_WIDTH, IMG_HEIGHT),
# batch_size=BATCH_SIZE,
# class_mode='categorical',
# shuffle=False
# )
# # Save class indices for later use
# class_indices = train_generator.class_indices
# print(f"📋 Class mapping: {class_indices}\n")
# # ============================================
# # STEP 4: Build Enhanced Model
# # ============================================
# base_model = MobileNetV2(
# input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
# include_top=False,
# weights='imagenet'
# )
# base_model.trainable = False
# model = Sequential([
# base_model,
#     GlobalAveragePooling2D(),
#     BatchNormalization(), # Added for better training stability
#     Dropout(0.5),
#     Dense(256, activation='relu'),
#     BatchNormalization(),
#     Dropout(0.4),
#     Dense(128, activation='relu'), # Additional layer for better feature learning
#     Dropout(0.3),
#     Dense(NUM_CLASSES, activation='softmax')
# ])
# # Compile
# model.compile(
# optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
# loss='categorical_crossentropy',
# metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
# )
# print("🏗️ Model Architecture:")
# model.summary()
# print()
# # ============================================
# # STEP 5: Enhanced Callbacks
# # ============================================
# callbacks = [
#     EarlyStopping(
# monitor='val_loss',
# patience=10,
# restore_best_weights=True,
# verbose=1
#     ),
#     ReduceLROnPlateau(
# monitor='val_loss',
# factor=0.2,
# patience=5,
# min_lr=1e-7,
# verbose=1
#     ),
#     ModelCheckpoint(
# 'best_model_initial.keras',
# monitor='val_accuracy',
# save_best_only=True,
# verbose=1
#     )
# ]
# # ============================================
# # STEP 6: Initial Training
# # ============================================
# print("🚀 Starting initial training phase...")
# history = model.fit(
# train_generator,
# validation_data=validation_generator,
# epochs=25,
# callbacks=callbacks,
# class_weight=class_weights,
# verbose=1
# )
# # ============================================
# # STEP 7: Fine-tuning
# # ============================================
# print("\n🔧 Starting fine-tuning phase...")
# base_model.trainable = True
# # Freeze early layers, unfreeze last 20 layers
# for layer in base_model.layers[:-20]:
# layer.trainable = False
# model.compile(
# optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
# loss='categorical_crossentropy',
# metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
# )
# fine_tune_callbacks = [
#     EarlyStopping(
# monitor='val_loss',
# patience=8,
# restore_best_weights=True,
# verbose=1
#     ),
#     ReduceLROnPlateau(
# monitor='val_loss',
# factor=0.3,
# patience=4,
# min_lr=1e-8,
# verbose=1
#     ),
#     ModelCheckpoint(
# 'best_model_finetuned.keras',
# monitor='val_accuracy',
# save_best_only=True,
# verbose=1
#     )
# ]
# fine_tune_history = model.fit(
# train_generator,
# validation_data=validation_generator,
# epochs=25,
# callbacks=fine_tune_callbacks,
# class_weight=class_weights,
# verbose=1
# )
# # ============================================
# # STEP 8: Evaluation and Visualization
# # ============================================
# print("\n📊 Evaluating model on test set...")
# # Get predictions
# test_generator.reset()
# predictions = model.predict(test_generator, verbose=1)
# predicted_classes = np.argmax(predictions, axis=1)
# true_classes = test_generator.classes
# # Class names
# class_names = list(class_indices.keys())
# # Classification Report
# print("\n📈 Classification Report:")
# print(classification_report(true_classes, predicted_classes, target_names=class_names))
# # Confusion Matrix
# cm = confusion_matrix(true_classes, predicted_classes)
# plt.figure(figsize=(10, 8))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
# xticklabels=class_names, yticklabels=class_names)
# plt.title('Confusion Matrix')
# plt.ylabel('True Label')
# plt.xlabel('Predicted Label')
# plt.tight_layout()
# plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
# print("✅ Confusion matrix saved as 'confusion_matrix.png'")
# # Plot Training History
# def plot_history(history, fine_tune_history):
# # Combine histories
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# if fine_tune_history:
# acc += fine_tune_history.history['accuracy']
# val_acc += fine_tune_history.history['val_accuracy']
# loss += fine_tune_history.history['loss']
# val_loss += fine_tune_history.history['val_loss']
# epochs_range = range(len(acc))
# plt.figure(figsize=(14, 5))
# # Accuracy plot
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.axvline(x=len(history.history['accuracy']), color='r', linestyle='--', label='Fine-tuning Start')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.grid(True)
# # Loss plot
# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.axvline(x=len(history.history['loss']), color='r', linestyle='--', label='Fine-tuning Start')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
# print("✅ Training history saved as 'training_history.png'")
# plot_history(history, fine_tune_history)
# # ============================================
# # STEP 9: Save Final Model
# # ============================================
# model.save('plant_disease_detection_final.keras')
# print("\n✅ Final model saved as 'plant_disease_detection_final.keras'")
# # Save class indices
# import json
# with open('class_indices.json', 'w') as f:
# json.dump(class_indices, f, indent=4)
# print("✅ Class indices saved as 'class_indices.json'")
# print("\n🎉 Training complete! Model is ready for deployment.")
# print(f"📁 Files created:")
# print(f" - plant_disease_detection_final.keras (final model)")
# print(f" - best_model_initial.keras (best initial training)")
# print(f" - best_model_finetuned.keras (best fine-tuned)")
# print(f" - confusion_matrix.png")
# print(f" - training_history.png")
# print(f" - class_indices.json")