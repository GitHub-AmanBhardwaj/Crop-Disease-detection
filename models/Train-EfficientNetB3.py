import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from PIL import Image

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
    def __init__(self, root_dir, target_size=(300, 300), batch_size=32, 
                 shuffle=True, augmentation=None, validation_split=0.2, 
                 subset=None, seed=42):
        super().__init__()
        self.root_dir = root_dir
        self.target_size = target_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.validation_split = validation_split
        self.subset = subset
        self.seed = seed
        
        self.image_paths = []
        self.labels = []
        self.class_names = []
        self.class_indices = {}
        
        class_dirs = set()
        for crop_dir in os.listdir(root_dir):
            crop_path = os.path.join(root_dir, crop_dir)
            if os.path.isdir(crop_path):
                for disease_dir in os.listdir(crop_path):
                    disease_path = os.path.join(crop_path, disease_dir)
                    if os.path.isdir(disease_path):
                        class_dirs.add(disease_dir)
        
        self.class_names = sorted(list(class_dirs))
        self.class_indices = {cls_name: i for i, cls_name in enumerate(self.class_names)}
        
        for crop_dir in os.listdir(root_dir):
            crop_path = os.path.join(root_dir, crop_dir)
            if os.path.isdir(crop_path):
                for disease_dir in os.listdir(crop_path):
                    disease_path = os.path.join(crop_path, disease_dir)
                    if os.path.isdir(disease_path) and disease_dir in self.class_indices:
                        label = self.class_indices[disease_dir]
                        for img_file in os.listdir(disease_path):
                            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                self.image_paths.append(os.path.join(disease_path, img_file))
                                self.labels.append(label)
        
        self.num_classes = len(self.class_names)
        
        if self.validation_split > 0:
            train_idx, val_idx = train_test_split(
                np.arange(len(self.image_paths)),
                test_size=self.validation_split,
                random_state=self.seed,
                stratify=self.labels
            )
            
            if subset == 'training':
                self.image_paths = [self.image_paths[i] for i in train_idx]
                self.labels = [self.labels[i] for i in train_idx]
            elif subset == 'validation':
                self.image_paths = [self.image_paths[i] for i in val_idx]
                self.labels = [self.labels[i] for i in val_idx]
        
        self.indexes = np.arange(len(self.image_paths))
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
            
            img = Image.open(img_path)
            img = img.resize(self.target_size, Image.LANCZOS)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            
            if self.augmentation:
                img_array = self.augmentation.random_transform(img_array)
            
            img_array = preprocess_input(img_array)
            
            batch_x.append(img_array)
            batch_y.append(label)
        
        batch_x = np.array(batch_x)
        batch_y = tf.keras.utils.to_categorical(batch_y, num_classes=self.num_classes)
        
        return batch_x, batch_y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def get_labels(self):
        return np.array(self.labels)

train_augmentation = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = NestedDirectoryGenerator(
    train_dir,
    target_size=(300, 300),
    batch_size=32,
    shuffle=True,
    augmentation=train_augmentation,
    validation_split=0.2,
    subset='training'
)

val_generator = NestedDirectoryGenerator(
    train_dir,
    target_size=(300, 300),
    batch_size=32,
    shuffle=False,
    augmentation=None,
    validation_split=0.2,
    subset='validation'
)

test_generator = NestedDirectoryGenerator(
    test_dir,
    target_size=(300, 300),
    batch_size=32,
    shuffle=False,
    augmentation=None
)

class_names = train_generator.class_names
num_classes = train_generator.num_classes
print(f"\nFound {len(train_generator.image_paths)} training images in {num_classes} classes")
print(f"Found {len(val_generator.image_paths)} validation images")
print(f"Found {len(test_generator.image_paths)} test images")
print(f"Class names: {class_names}")

labels = train_generator.get_labels()
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

base_model = EfficientNetB3(
    input_shape=(300, 300, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.Dense(1024, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('best_disease_model_b3.keras', save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=3)
]

print("\nTraining model...")
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

print("\nFine-tuning model...")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

def plot_history(history, history_fine=None):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    if history_fine:
        plt.plot(range(len(history.history['accuracy']), 
                       len(history.history['accuracy']) + len(history_fine.history['accuracy'])),
                 history_fine.history['accuracy'], label='Fine-Tune Training Accuracy')
        plt.plot(range(len(history.history['val_accuracy']), 
                       len(history.history['val_accuracy']) + len(history_fine.history['val_accuracy'])),
                 history_fine.history['val_accuracy'], label='Fine-Tune Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    if history_fine:
        plt.plot(range(len(history.history['loss']), 
                       len(history.history['loss']) + len(history_fine.history['loss'])),
                 history_fine.history['loss'], label='Fine-Tune Training Loss')
        plt.plot(range(len(history.history['val_loss']), 
                       len(history.history['val_loss']) + len(history_fine.history['val_loss'])),
                 history_fine.history['val_loss'], label='Fine-Tune Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_history(history, history_fine)

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
print(classification_report(test_true_classes, test_pred_classes, target_names=class_names, zero_division=0))

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
plt.title('Confusion Matrix (Test Set)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(test_true_classes, test_pred_classes, average='weighted', zero_division=0)
recall = recall_score(test_true_classes, test_pred_classes, average='weighted', zero_division=0)
f1 = f1_score(test_true_classes, test_pred_classes, average='weighted', zero_division=0)

print(f"\nWeighted Precision: {precision:.4f}")
print(f"Weighted Recall: {recall:.4f}")
print(f"Weighted F1 Score: {f1:.4f}")