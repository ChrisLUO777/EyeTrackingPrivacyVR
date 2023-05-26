import tensorflow as tf
from keras.layers import Input, Dense, BatchNormalization, Activation, Subtract, Lambda, Conv1D, MaxPooling1D, \
    Flatten, GlobalAveragePooling1D, LayerNormalization
from keras.models import Model


# Comparison subnetwork
def create_comparison_subnetwork(input_size):
    model = tf.keras.Sequential([
        # Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(input_size, 1)),
        # Conv1D(filters=64, kernel_size=3, activation='relu'),
        # Conv1D(filters=128, kernel_size=3, activation='relu'),
        # GlobalAveragePooling1D(),
        # Dense(128),
        # Activation('relu'),
        # Dense(128),
        # Activation('relu')

        Dense(128, input_shape=(input_size,)),
        BatchNormalization(momentum=0.9),
        # LayerNormalization(),
        Activation('relu'),
        Dense(128),
        BatchNormalization(momentum=0.9),
        # LayerNormalization(),
        Activation('relu'),
    ])
    return model


def create_jfg_comparator(jfg_output_size):
    # Input placeholders
    input_a = Input(shape=(jfg_output_size,))
    input_b = Input(shape=(jfg_output_size,))

    # Create shared comparison subnetwork
    comparison_subnetwork = create_comparison_subnetwork(jfg_output_size)
    # Process the input feature vectors
    processed_a = comparison_subnetwork(input_a)
    processed_b = comparison_subnetwork(input_b)
    # Compute absolute difference between the processed feature vectors
    abs_diff = Lambda(lambda x: tf.abs(x[0] - x[1]))([processed_a, processed_b])
    # Final similarity score
    similarity_score = Dense(1, activation='sigmoid')(abs_diff)

    # # Element-wise difference and absolute value
    # diff = tf.keras.layers.Subtract()([input_a, input_b])
    # abs_diff = tf.keras.layers.Lambda(tf.abs)(diff)
    #
    # # Dense layers to capture non-linear relationships
    # dense1 = tf.keras.layers.Dense(64, activation='relu')(abs_diff)
    #
    # # Dense layer with sigmoid activation for similarity score
    # similarity_score = tf.keras.layers.Dense(1, activation='sigmoid')(dense1)

    # Create the Siamese network model
    siamese_model = Model(inputs=[input_a, input_b], outputs=similarity_score)
    return siamese_model
