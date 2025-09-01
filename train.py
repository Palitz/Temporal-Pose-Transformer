# filename: train.py

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
from sklearn.model_selection import train_test_split
import os

# --- 1. TPT Model Definition (No changes here) ---
@tf.keras.utils.register_keras_serializable()
class PositionalEncoding(layers.Layer):
    # ... (content of this class is unchanged) ...
    def __init__(self, position, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.position = position
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(position, d_model)
    def get_angles(self, pos, i, d_model):
        return pos * (1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model)))
    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)
    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({"position": self.position, "d_model": self.d_model})
        return config

@tf.keras.utils.register_keras_serializable()
class TransformerEncoderLayer(layers.Layer):
    # ... (content of this class is unchanged) ...
    def __init__(self, d_model, num_heads, dff, rate=0.1, **kwargs):
        super(TransformerEncoderLayer, self).__init__(**kwargs)
        self.d_model, self.num_heads, self.dff, self.rate = d_model, num_heads, dff, rate
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([layers.Dense(dff, activation='relu'), layers.Dense(d_model)])
        self.layernorm1, self.layernorm2 = layers.LayerNormalization(epsilon=1e-6), layers.LayerNormalization(epsilon=1e-6)
        self.dropout1, self.dropout2 = layers.Dropout(rate), layers.Dropout(rate)
    def call(self, x, training=False):
        attn_output = self.mha(x, x, x)
        out1 = self.layernorm1(x + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + self.dropout2(ffn_output, training=training))
        return out2
    def get_config(self):
        config = super(TransformerEncoderLayer, self).get_config()
        config.update({"d_model": self.d_model, "num_heads": self.num_heads, "dff": self.dff, "rate": self.rate})
        return config

@tf.keras.utils.register_keras_serializable()
class ClassToken(layers.Layer):
    # ... (content of this class is unchanged) ...
    def __init__(self, d_model, **kwargs):
        super(ClassToken, self).__init__(**kwargs)
        self.d_model = d_model
        self.class_token = self.add_weight(shape=(1, 1, d_model), initializer='random_normal', trainable=True, name='class_token')
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        cls_broadcasted = tf.tile(self.class_token, [batch_size, 1, 1])
        return tf.concat([cls_broadcasted, inputs], axis=1)
    def get_config(self):
        config = super(ClassToken, self).get_config()
        config.update({"d_model": self.d_model})
        return config

def TPT_model_with_class_token(sequence_length, num_keypoints, num_classes, d_model, num_heads, dff, num_encoder_layers, dropout_rate=0.1):
    # ... (content of this function is unchanged) ...
    inputs = layers.Input(shape=(sequence_length, num_keypoints))
    x = layers.Dense(d_model)(inputs)
    x = ClassToken(d_model)(x)
    x = PositionalEncoding(sequence_length + 1, d_model)(x)
    x = layers.Dropout(dropout_rate)(x)
    for _ in range(num_encoder_layers):
        x = TransformerEncoderLayer(d_model, num_heads, dff, dropout_rate)(x)
    cls_token_output = layers.Lambda(lambda t: t[:, 0, :], name='get_cls_token')(x)
    x = layers.LayerNormalization()(cls_token_output)
    x = layers.Dense(d_model // 2, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# --- 2. Training Configuration ---
SEQUENCE_LENGTH = 30
NUM_KEYPOINTS = 33 * 3
NUM_CLASSES = 101
D_MODEL = 64
NUM_HEADS = 4
DFF = 128
NUM_ENCODER_LAYERS = 3
DROPOUT_RATE = 0.1
EPOCHS = 150
BATCH_SIZE = 64
LEARNING_RATE = 0.0005

# <<< MODIFIED SECTION START >>>
# --- 3. Data Augmentation Function (Now using pure TensorFlow) ---
@tf.function
def augment_sequence_tf(sequence, label):
    """
    Applies jittering and scaling to a single keypoint sequence using TensorFlow operations.
    """
    # Reshape from (30, 99) to (30, 33, 3)
    sequence = tf.reshape(sequence, (SEQUENCE_LENGTH, NUM_KEYPOINTS // 3, 3))
    
    # Separate coordinates and visibility
    coords = sequence[:, :, :2]
    visibility = sequence[:, :, 2:]
    
    # 1. Jittering: Add small random noise to x and y coordinates
    jitter = tf.random.uniform(shape=tf.shape(coords), minval=-0.02, maxval=0.02, dtype=tf.float32)
    coords = coords + jitter
    
    # 2. Scaling: Randomly scale the entire skeleton
    scale = tf.random.uniform(shape=[], minval=0.85, maxval=1.15, dtype=tf.float32)
    coords = coords * scale
    
    # Recombine coordinates and visibility
    sequence = tf.concat([coords, visibility], axis=-1)
    
    # Flatten back to the original shape (30, 99)
    sequence = tf.reshape(sequence, (SEQUENCE_LENGTH, NUM_KEYPOINTS))
    
    return sequence, label
# <<< MODIFIED SECTION END >>>

# --- 4. Main Training Script ---
def main():
    print("Loading processed data from .npy files...")
    X_DATA_PATH = 'X_data_ucf101.npy'
    Y_LABELS_PATH = 'y_labels_ucf101.npy'

    if not os.path.exists(X_DATA_PATH) or not os.path.exists(Y_LABELS_PATH):
        print(f"\nERROR: Processed data files not found. Please run `preprocess_data.py` first.")
        return

    X = np.load(X_DATA_PATH)
    y = np.load(Y_LABELS_PATH)
    print(f"Data loaded. X shape: {X.shape}, y shape: {y.shape}")

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print(f"\nTraining samples: {len(X_train)} | Validation samples: {len(X_val)} | Test samples: {len(X_test)}")

    print("\nCreating TensorFlow datasets with on-the-fly augmentation...")
    
    # <<< MODIFIED SECTION START >>>
    # Training dataset now uses the pure TensorFlow augmentation function
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=len(X_train))
    train_dataset = train_dataset.map(augment_sequence_tf, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Validation and Test datasets (no changes needed)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    # <<< MODIFIED SECTION END >>>
    
    print("\nBuilding TPT model...")
    model = TPT_model_with_class_token(
        sequence_length=SEQUENCE_LENGTH, num_keypoints=NUM_KEYPOINTS, num_classes=NUM_CLASSES,
        d_model=D_MODEL, num_heads=NUM_HEADS, dff=DFF, num_encoder_layers=NUM_ENCODER_LAYERS,
        dropout_rate=DROPOUT_RATE
    )
    model.summary()

    print("\nUsing AdamW optimizer for this training run.")
    optimizer = tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=0.01)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    checkpoint_cb = ModelCheckpoint('tpt_ucf101_augmented.h5', save_best_only=True, monitor='val_accuracy', mode='max')
    early_stopping_cb = EarlyStopping(patience=20, monitor='val_accuracy', mode='max', restore_best_weights=True)
    reduce_lr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6)

    print("\nStarting training with data augmentation...")
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb]
    )

    print("\nTraining complete. Evaluating model on the hold-out test set...")
    best_model = tf.keras.models.load_model(
        'tpt_ucf101_augmented.h5',
        custom_objects={
            "ClassToken": ClassToken, "PositionalEncoding": PositionalEncoding, "TransformerEncoderLayer": TransformerEncoderLayer
        },
        safe_mode=False
    )
    
    test_loss, test_accuracy = best_model.evaluate(test_dataset)

    print("\n" + "="*50)
    print("       FINAL MODEL PERFORMANCE (WITH AUGMENTATION)")
    print("="*50)
    print(f"Test Set Loss: {test_loss:.4f}")
    print(f"Test Set Accuracy: {test_accuracy*100:.2f}%")
    print("="*50)

if __name__ == '__main__':
    main()