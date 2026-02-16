import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.applications import EfficientNetB0, VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Parameters ---
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
SEED = 123
DATA_DIR = "/kaggle/input/fruit-ripeness-unripe-ripe-and-rotten/fruit_ripeness_dataset/archive (1)/dataset/dataset/train"  # Update this path

# --- 1. Load and prepare dataset info with pandas ---

def get_image_paths_and_labels(data_dir):
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    image_paths = []
    labels = []
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(cls_dir, fname))
                labels.append(cls)
    return image_paths, labels, classes

image_paths, labels, classes = get_image_paths_and_labels(DATA_DIR)

df = pd.DataFrame({'filepath': image_paths, 'label': labels})
print("Class distribution:\n", df['label'].value_counts())

# --- 2. Split data into train, val, test (60/20/20) stratified ---

trainval_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=SEED)
train_df, val_df = train_test_split(trainval_df, test_size=0.25, stratify=trainval_df['label'], random_state=SEED)  # 0.25 x 0.8 = 0.2

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# --- 3. Prepare directories for flow_from_directory ---

def prepare_dir(df, base_dir):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    for cls in classes:
        cls_dir = os.path.join(base_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
    for _, row in df.iterrows():
        dst = os.path.join(base_dir, row['label'], os.path.basename(row['filepath']))
        if not os.path.exists(dst):
            tf.io.gfile.copy(row['filepath'], dst, overwrite=True)

base_train_dir = "/kaggle/working/train"
base_val_dir = "/kaggle/working/val"
base_test_dir = "/kaggle/working/test"

prepare_dir(train_df, base_train_dir)
prepare_dir(val_df, base_val_dir)
prepare_dir(test_df, base_test_dir)

# --- 4. Data generators with augmentation for train only ---

train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    base_train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    shuffle=True,
    seed=SEED
)

val_generator = val_test_datagen.flow_from_directory(
    base_val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    base_test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    shuffle=False
)

# --- 5. Model builder function with hyperparameter tuning for dropout ---

def build_model(model_name='efficientnet', dropout_rate=0.3):
    if model_name == 'efficientnet':
        base_model = EfficientNetB0(include_top=False, input_shape=IMG_SIZE + (3,), weights='/kaggle/input/imagenet/keras/default/1/efficientnetb0_notop.h5')
    elif model_name == 'vgg16':
        base_model = VGG16(include_top=False, input_shape=IMG_SIZE + (3,), weights='/kaggle/input/vgg16/keras/default/1/vgg16_weights_tf_dim_ordering_tf_kernels_notop (1).h5')
    else:
        raise ValueError("Unsupported model_name")

    base_model.trainable = False  # Freeze base initially

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(len(classes), activation='softmax')(x)
    model = models.Model(inputs, outputs)

    return model, base_model

# --- 6. Training function with fine-tuning ---

def train_model(model_name, dropout_rate=0.3, initial_epochs=15, fine_tune_epochs=10, fine_tune_at=100):
    print(f"\nTraining {model_name} model with dropout={dropout_rate}")

    model, base_model = build_model(model_name, dropout_rate)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    checkpoint = callbacks.ModelCheckpoint(f'best_{model_name}.h5', monitor='val_accuracy', save_best_only=True)

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=initial_epochs,
        callbacks=[early_stopping, reduce_lr, checkpoint]
    )

    # Fine-tuning
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history_fine = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=initial_epochs + fine_tune_epochs,
        initial_epoch=history.epoch[-1],
        callbacks=[early_stopping, reduce_lr, checkpoint]
    )

    # Load best weights
    model.load_weights(f'best_{model_name}.h5')

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"{model_name} Test accuracy: {test_acc:.4f}")

    return model, history, history_fine, test_acc

# --- 7. Train both models ---

efficientnet_model, eff_hist, eff_hist_fine, eff_test_acc = train_model('efficientnet', dropout_rate=0.3)
vgg16_model, vgg_hist, vgg_hist_fine, vgg_test_acc = train_model('vgg16', dropout_rate=0.5, fine_tune_at=15)

# --- 8. Ensemble predictions on test set ---

def ensemble_predictions(models, generator):
    preds = [model.predict(generator, verbose=0) for model in models]
    avg_preds = np.mean(preds, axis=0)
    y_pred = np.argmax(avg_preds, axis=1)
    y_true = generator.classes
    acc = accuracy_score(y_true, y_pred)
    print(f"Ensemble Test accuracy: {acc:.4f}")
    return y_pred, acc

ensemble_pred, ensemble_acc = ensemble_predictions([efficientnet_model, vgg16_model], test_generator)

# --- 9. Plot training history ---

def plot_history(hist1, hist2, model_name):
    acc = hist1.history['accuracy'] + hist2.history['accuracy']
    val_acc = hist1.history['val_accuracy'] + hist2.history['val_accuracy']
    loss = hist1.history['loss'] + hist2.history['loss']
    val_loss = hist1.history['val_loss'] + hist2.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Train Accuracy')
    plt.plot(epochs_range, val_acc, label='Val Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Val Loss')
    plt.title(f'{model_name} Loss')
    plt.legend()

    plt.show()

plot_history(eff_hist, eff_hist_fine, 'EfficientNetB0')
plot_history(vgg_hist, vgg_hist_fine, 'VGG16')

# --- 10. Save best model ---

best_model = efficientnet_model if eff_test_acc >= vgg_test_acc else vgg16_model
best_model_name = 'efficientnet' if eff_test_acc >= vgg_test_acc else 'vgg16'
best_model.save(f'/kaggle/working/best_{best_model_name}_fruit_ripeness_model.h5')
print(f"Best model saved: {best_model_name} with test accuracy {max(eff_test_acc, vgg_test_acc):.4f}")
