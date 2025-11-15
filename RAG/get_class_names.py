# get_class_names.py
import tensorflow as tf
import json

# Load training data to get class names
training_set = tf.keras.utils.image_dataset_from_directory(
    'Detector/train',
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
    image_size=(128, 128),
)

# Get class names
class_names = training_set.class_names

print(f"Found {len(class_names)} classes:")
for i, name in enumerate(class_names):
    print(f"  {i}: {name}")

# Save to JSON file
with open('class_names.json', 'w') as f:
    json.dump(class_names, f, indent=2)

print(f"\n✅ Saved class names to class_names.json")