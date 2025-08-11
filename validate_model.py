import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os

# A simple, direct validation script that bypasses the unittest framework.
# It will either print a success message or crash with an error.

# Prerequisite: Make sure h5py is installed (`pip install h5py`)

# --- 1. Custom Layer Definitions ---
@tf.keras.utils.register_keras_serializable()
class PositionalEncoding(layers.Layer):
    def __init__(self, position, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.position = position
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(position, d_model)
    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates
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

# --- 2. Model Builder Function ---
def TPT_model_with_class_token(sequence_length, num_keypoints, num_classes, d_model, num_heads, dff, num_encoder_layers, dropout_rate=0.1):
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

# --- 3. Main Validation Logic ---
def run_validation():
    """Performs all checks and prints a success or failure message."""
    MODEL_FILE = "tpt_model.h5"
    print("--- Starting TPT Model Validation ---")
    try:
        # Define Hyperparameters
        SEQUENCE_LENGTH, NUM_KEYPOINTS, NUM_CLASSES = 30, 99, 5
        D_MODEL, NUM_HEADS, DFF, NUM_ENCODER_LAYERS = 48, 2, 96, 2
        BATCH_SIZE = 16

        # 1. Model Creation
        print("[1/5] Creating model...")
        model = TPT_model_with_class_token(
            sequence_length=SEQUENCE_LENGTH, num_keypoints=NUM_KEYPOINTS, num_classes=NUM_CLASSES,
            d_model=D_MODEL, num_heads=NUM_HEADS, dff=DFF, num_encoder_layers=NUM_ENCODER_LAYERS
        )
        model.summary()
        print("Model creation successful.")

        # 2. Compile and Train for one step
        print("\n[2/5] Compiling and training one step...")
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        dummy_input = np.random.rand(BATCH_SIZE, SEQUENCE_LENGTH, NUM_KEYPOINTS).astype(np.float32)
        dummy_labels = np.random.randint(0, NUM_CLASSES, size=(BATCH_SIZE,))
        model.fit(dummy_input, dummy_labels, epochs=1, verbose=0)
        print("Compile and train step successful.")

        # 3. Save the model
        print(f"\n[3/5] Saving model to {MODEL_FILE}...")
        model.save(MODEL_FILE)
        assert os.path.exists(MODEL_FILE), "Model file was not created!"
        print("Model saving successful.")

        # 4. Reload the model
        print(f"\n[4/5] Loading model from {MODEL_FILE}...")
        reloaded_model = tf.keras.models.load_model(MODEL_FILE)
        assert isinstance(reloaded_model, tf.keras.Model), "Failed to reload model correctly."
        print("Model loading successful.")

        # 5. Verify reloaded model can predict
        print("\n[5/5] Verifying prediction with reloaded model...")
        predictions = reloaded_model.predict(dummy_input, verbose=0)
        assert predictions.shape == (BATCH_SIZE, NUM_CLASSES), "Prediction output shape is incorrect."
        print("Prediction verification successful.")

        print("\n" + "="*50)
        print("  ✅ ALL CHECKS PASSED! The model code is valid. ✅")
        print("="*50 + "\n")

    except Exception as e:
        print("\n" + "!"*50)
        print(f"  ❌ VALIDATION FAILED: An error occurred.")
        print(f"  ERROR: {e}")
        print("!"*50 + "\n")
    
    finally:
        # Clean up the created model file
        if os.path.exists(MODEL_FILE):
            os.remove(MODEL_FILE)
            print(f"Cleaned up {MODEL_FILE}.")

# --- Run the validation function ---
if __name__ == "__main__":
    run_validation()