import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import math
import random
from tensorflow.keras.applications import DenseNet121

# Set random seed for reproducibility
random.seed(42)

# Constants
image_size = (224, 224)
batch_size = 32
epochs = 5
num_classes = 2  # Adjust according to your dataset
class_names = ['healthy', 'scab']

# Data paths
train_dir = 'apple_ds/train'
val_dir = 'apple_ds/val'
test_dir = 'apple_ds/test'

# Data Augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Data Generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Model
base_model = DenseNet121(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Training
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator
)

# Testing
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc}")

# Predict test labels
y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)

# Save Results to Excel
results_df = pd.DataFrame({
    'Test Image': test_generator.filenames,
    'True Class': [class_names[i] for i in test_generator.classes],
    'Predicted Class': [class_names[i] for i in y_pred]
})

results_df.to_excel('results/densenet121_test_results.xlsx', index=False)

# Confusion Matrix
cm = confusion_matrix(test_generator.classes, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm')
plt.xlabel('Predicted', fontsize=16, weight='bold')
plt.ylabel('True', fontsize=16, weight='bold')
plt.title('Confusion Matrix', fontsize=18, weight='bold')

plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
plt.yticks(np.arange(len(class_names)), class_names, rotation=45)
plt.savefig('results/densenet121_confusion_matrix.png')
plt.show()

# Classification Report
classification_rep = classification_report(test_generator.classes, y_pred, target_names=class_names)
print(classification_rep)

# ROC Plot for multi-class classification
plt.figure(figsize=(8, 6))
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(test_generator.classes == i, y_pred_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC Curve ({class_names[i]}, area = {roc_auc:.2f})', linewidth=3)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate', fontsize=14, weight='bold')
plt.ylabel('True Positive Rate', fontsize=14, weight='bold')
plt.title('ROC Curve for Multi-Class Classification', fontsize=16, weight='bold')
plt.legend(loc='lower right')
plt.savefig('results/densenet121_roc_curve.png')
plt.show()

# Function to plot images with labels
def plot_images(images, labels, predicted_labels, class_names, rows=1, figsize=(15, 10)):
    fig, axes = plt.subplots(rows, math.ceil(len(images)/rows), figsize=figsize)
    axes = axes.flatten()
    for i, (image, label, predicted_label) in enumerate(zip(images, labels, predicted_labels)):
        axes[i].imshow(image)
        axes[i].axis('off')
        title = f'True Label: {label}\nPredicted Label: {predicted_label}'
        axes[i].set_title(title, fontsize=12, color='white')
    plt.tight_layout()
    plt.savefig('results/densenet121_selected_images.png')
    plt.show()

# Labeling selected test images
healthy_images = random.sample(os.listdir(os.path.join(test_dir, 'healthy')), 5)
scab_images = random.sample(os.listdir(os.path.join(test_dir, 'scab')), 5)

# Load and plot the selected images with labels
selected_images = healthy_images + scab_images
selected_labels = ['healthy'] * 5 + ['scab'] * 5

# Load images
images = []
for img_path in selected_images:
    img = plt.imread(os.path.join(test_dir, 'healthy' if img_path in healthy_images else 'scab', img_path))
    images.append(img)

# Get predictions for selected images
images_resized = [tf.image.resize(img, (224, 224)) for img in images]
predictions = model.predict(np.array(images_resized))
predicted_labels = np.argmax(predictions, axis=1)
predicted_labels = [class_names[i] for i in predicted_labels]

# Plot images with labels
plot_images(images, selected_labels, predicted_labels, class_names, rows=2)
