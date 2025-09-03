import tensorflow as tf
import os
import numpy as np
from sklearn.utils import class_weight
import pathlib

# --- Configuration ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 15
FINE_TUNE_EPOCHS = 20
LEARNING_RATE = 0.001
FINE_TUNE_LEARNING_RATE = 0.00005
UNFREEZE_LAYERS = 20

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'models', 'breed_recognition_model_v4.h5')

# Ensure model save directory exists
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

def get_available_breeds(image_dir):
    """Scans the image directory and returns a list of breeds (subdirectories) that are not empty."""
    available_breeds = []
    for breed in os.listdir(image_dir):
        breed_path = os.path.join(image_dir, breed)
        if os.path.isdir(breed_path) and len(os.listdir(breed_path)) > 0:
            available_breeds.append(breed)
    print(f"Found {len(available_breeds)} breeds with images.")
    return sorted(available_breeds)

def load_and_preprocess_data(image_dir, class_names, validation_split=0.2, seed=123):
    """Manually loads image paths, creates datasets, and preprocesses them."""
    print("Manually loading image paths and creating datasets...")
    num_classes = len(class_names)
    data_dir = pathlib.Path(image_dir)

    all_image_paths = [str(path) for path in data_dir.glob('*/*') if path.parent.name in class_names]
    class_to_index = dict((name, i) for i, name in enumerate(class_names))
    all_image_labels = [class_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]

    path_ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))
    path_ds = path_ds.shuffle(buffer_size=len(all_image_paths), seed=seed)

    val_size = int(len(all_image_paths) * validation_split)
    train_ds = path_ds.skip(val_size)
    val_ds = path_ds.take(val_size)

    def process_path(file_path, label):
        img = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
        label = tf.one_hot(label, num_classes)
        return img, label

    train_ds = train_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.2)
    ])
    preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input

    def prepare(ds, augment=False):
        if augment:
            ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

    train_ds = prepare(train_ds, augment=True)
    val_ds = prepare(val_ds)

    return train_ds, val_ds, num_classes

def build_and_train_model(train_ds, val_ds, num_classes, class_names, class_weight_dict):
    print("\nBuilding and training model...")

    base_model = tf.keras.applications.MobileNetV3Large(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                                  include_top=False,
                                  weights='imagenet')

    base_model.trainable = False

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])

    print("--- Model Summary (Feature Extraction Phase) ---")
    model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001, verbose=1)

    print("\n--- Training (Feature Extraction Phase) ---")
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=[early_stopping, reduce_lr],
        class_weight=class_weight_dict
    )

    print("\n--- Fine-tuning Phase ---")
    base_model.trainable = True
    for layer in base_model.layers[:-UNFREEZE_LAYERS]:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])
    
    print("--- Model Summary (Fine-Tuning Phase) ---")
    model.summary()

    fine_tune_history = model.fit(
        train_ds,
        epochs=EPOCHS + FINE_TUNE_EPOCHS,
        initial_epoch=history.epoch[-1],
        validation_data=val_ds,
        callbacks=[early_stopping, reduce_lr],
        class_weight=class_weight_dict
    )

    print("\nSaving model...")
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    class_names_path = os.path.join(os.path.dirname(MODEL_SAVE_PATH), 'class_names.txt')
    with open(class_names_path, 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    print(f"Class names saved to {class_names_path}")

    return model, history, fine_tune_history

if __name__ == '__main__':
    available_breeds = get_available_breeds(IMAGE_DIR)
    train_ds, val_ds, num_classes = load_and_preprocess_data(IMAGE_DIR, available_breeds)

    print("\n--- Calculating Class Weights ---")
    y_labels_one_hot = np.concatenate([y for x, y in train_ds], axis=0)
    y_integers = np.argmax(y_labels_one_hot, axis=1)
    
    class_weights_array = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_integers),
        y=y_integers
    )
    
    class_weight_dict = dict(enumerate(class_weights_array))
    print("Class weights calculated to counteract data imbalance.")

    model, history, fine_tune_history = build_and_train_model(train_ds, val_ds, num_classes, available_breeds, class_weight_dict)
