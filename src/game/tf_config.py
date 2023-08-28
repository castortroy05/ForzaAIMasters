import tensorflow as tf

def configure_tensorflow():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.run_functions_eagerly(True)
    tf.data.experimental.enable_debug_mode()

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
        except RuntimeError as e:
            print(e)
