import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU is available and being used")
else:
    print("No GPU available, using CPU")

data_dir = '/kaggle/input/plantvillageupdated/Crop-Res_split'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

class NestedDirectoryGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, labels, class_names, target_size=(224, 224), batch_size=32, shuffle=True, augmentation=None):
        self.image_paths = image_paths
        self.labels = labels
        self.class_names = class_names
        self.target_size = target_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.num_classes = len(class_names)
        self.indexes = np.arange(len(image_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_x = []
        batch_y = []
        
        for i in batch_indexes:
            img_path = self.image_paths[i]
            label = self.labels[i]
            
            img = tf.keras.preprocessing.image.load_img(
                img_path, target_size=self.target_size)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            
            if self.augmentation:
                img_array = self.augmentation.random_transform(img_array)
            
            batch_x.append(img_array)
            batch_y.append(label)
        
        batch_x = np.array(batch_x) / 255.0
        batch_y = tf.keras.utils.to_categorical(batch_y, num_classes=self.num_classes)
        
        return batch_x, batch_y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

def collect_data(directory):
    image_paths = []
    labels = []
    class_names = []
    class_indices = {}
    
    class_dirs = set()
    for crop_dir in os.listdir(directory):
        crop_path = os.path.join(directory, crop_dir)
        if os.path.isdir(crop_path):
            for disease_dir in os.listdir(crop_path):
                disease_path = os.path.join(crop_path, disease_dir)
                if os.path.isdir(disease_path):
                    class_dirs.add(disease_dir)
    
    class_names = sorted(list(class_dirs))
    class_indices = {cls_name: i for i, cls_name in enumerate(class_names)}
    
    for crop_dir in os.listdir(directory):
        crop_path = os.path.join(directory, crop_dir)
        if os.path.isdir(crop_path):
            for disease_dir in os.listdir(crop_path):
                disease_path = os.path.join(crop_path, disease_dir)
                if os.path.isdir(disease_path) and disease_dir in class_indices:
                    label = class_indices[disease_dir]
                    for img_file in os.listdir(disease_path):
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            image_paths.append(os.path.join(disease_path, img_file))
                            labels.append(label)
    
    return image_paths, labels, class_names

train_image_paths, train_labels, class_names = collect_data(train_dir)
test_image_paths, test_labels, _ = collect_data(test_dir)

train_paths, val_paths, train_labels_split, val_labels = train_test_split(
    train_image_paths, train_labels, test_size=0.2, random_state=42, stratify=train_labels
)

print(f"\nFound {len(train_paths)} training images, {len(val_paths)} validation images, and {len(test_image_paths)} test images in {len(class_names)} classes")
print(f"Class names: {class_names}")

train_augmentation = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = NestedDirectoryGenerator(
    train_paths,
    train_labels_split,
    class_names,
    target_size=(224, 224),
    batch_size=32,
    shuffle=True,
    augmentation=train_augmentation
)

val_generator = NestedDirectoryGenerator(
    val_paths,
    val_labels,
    class_names,
    target_size=(224, 224),
    batch_size=32,
    shuffle=False,
    augmentation=None
)

test_generator = NestedDirectoryGenerator(
    test_image_paths,
    test_labels,
    class_names,
    target_size=(224, 224),
    batch_size=32,
    shuffle=False,
    augmentation=None
)

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('best_disease_model.keras', save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=3)
]

print("\nTraining model...")
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

def plot_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_history(history)

print("\nEvaluating on test set...")
test_results = model.evaluate(test_generator)
print(f"Test Loss: {test_results[0]:.4f}")
print(f"Test Accuracy: {test_results[1]:.4f}")

print("\nComputing predictions and metrics...")
test_pred_classes = []
test_true_classes = []

for i in range(len(test_generator)):
    x, y = test_generator[i]
    preds = model.predict(x, verbose=0)
    test_pred_classes.extend(np.argmax(preds, axis=-1))
    test_true_classes.extend(np.argmax(y, axis=-1))
    
    if (i+1) % 10 == 0:
        print(f"Processed {i+1}/{len(test_generator)} batches")

test_pred_classes = np.array(test_pred_classes)
test_true_classes = np.array(test_true_classes)

print("\nTest Classification Report:")
print(classification_report(test_true_classes, test_pred_classes, target_names=class_names))

plt.figure(figsize=(12, 10))
if len(class_names) > 20:
    print("Showing confusion matrix for first 20 classes due to large number of classes")
    mask = test_true_classes < 20
    cm = confusion_matrix(test_true_classes[mask], test_pred_classes[mask])
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names[:20],
        yticklabels=class_names[:20]
    )
else:
    cm = confusion_matrix(test_true_classes, test_pred_classes)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(test_true_classes, test_pred_classes, average='weighted')
recall = recall_score(test_true_classes, test_pred_classes, average='weighted')
f1 = f1_score(test_true_classes, test_pred_classes, average='weighted')

print(f"\nWeighted Precision: {precision:.4f}")
print(f"Weighted Recall: {recall:.4f}")
print(f"Weighted F1 Score: {f1:.4f}")