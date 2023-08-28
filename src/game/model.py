import tensorflow as tf
from tensorflow.keras.optimizers import Adam

def build_deep_learning_model(input_dims):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(256, input_dim=input_dims, activation='relu'))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model
