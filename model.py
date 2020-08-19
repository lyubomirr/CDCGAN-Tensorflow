import tensorflow as tf
from tensorflow.keras import layers

LEAKY_RELU_ALPHA = 0.2
IMAGE_SIZE = 28
IMAGE_CHANNELS = 1
NOISE_DIMS = 100
NUM_CLASSES = 10

def make_generator_model():
    weight_initializer = tf.random_normal_initializer(stddev=0.02)

    noise = layers.Input(shape=(NOISE_DIMS,))    
    dense_n = layers.Dense(7*7*256, use_bias=False, kernel_initializer=weight_initializer)(noise)
    batch_n = layers.BatchNormalization()(dense_n)
    relu_n = layers.LeakyReLU(LEAKY_RELU_ALPHA)(batch_n)
    reshaped_n = layers.Reshape((7, 7, 256))(relu_n)
    
    label = layers.Input(shape=(NUM_CLASSES,))
    dense_l = layers.Dense(7*7*256, use_bias=False, kernel_initializer=weight_initializer)(label)
    batch_l = layers.BatchNormalization()(dense_l)
    relu_l = layers.LeakyReLU(LEAKY_RELU_ALPHA)(batch_l)
    reshaped_l = layers.Reshape((7, 7, 256))(relu_l)
    
    input = layers.Concatenate()([reshaped_n, reshaped_l])
    
    conv = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False, 
                                     kernel_initializer=weight_initializer)(input)
    batch = layers.BatchNormalization()(conv)                         
    drop = layers.Dropout(0.5)(batch)
    relu = layers.LeakyReLU(LEAKY_RELU_ALPHA)(drop)
                              
    conv = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False, 
                                  kernel_initializer=weight_initializer)(relu)
    batch = layers.BatchNormalization()(conv)                         
    drop = layers.Dropout(0.5)(batch)
    relu = layers.LeakyReLU(LEAKY_RELU_ALPHA)(drop)

    out = layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, 
                                  activation='tanh')(relu)
                              
    model = tf.keras.Model(inputs=[noise, label], outputs=out)
    #tf.keras.utils.plot_model(model, to_file="generator.png", show_shapes=True, show_layer_names=False)
    model.summary()
    return model

def make_discriminator_model():
    weight_initializer = tf.random_normal_initializer(stddev=0.02)
    
    image = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS))
    noise = layers.GaussianNoise(0.2)(image)
    
    conv = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', 
                         kernel_initializer= weight_initializer)(noise)
    batch = layers.BatchNormalization()(conv)
    relu = layers.LeakyReLU(LEAKY_RELU_ALPHA)(batch)
    drop = layers.Dropout(0.3)(relu)
    
    conv = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', 
                         kernel_initializer= weight_initializer)(drop)
    batch = layers.BatchNormalization()(conv)
    relu = layers.LeakyReLU(LEAKY_RELU_ALPHA)(batch)
    img_out = layers.Dropout(0.3)(relu)
    
    label = layers.Input(shape=(NUM_CLASSES,))
    dense_l = layers.Dense(7*7*128, use_bias=False, kernel_initializer=weight_initializer)(label)
    batch_l = layers.BatchNormalization()(dense_l)
    relu_l = layers.LeakyReLU(LEAKY_RELU_ALPHA)(batch_l)
    label_out = layers.Reshape((7, 7, 128))(relu_l)
            
    merged = layers.Concatenate()([img_out, label_out])
    out = layers.Dense(1, activation='sigmoid')(merged)
    
    model = tf.keras.Model(inputs=[image, label], outputs=out)
    #tf.keras.utils.plot_model(model, to_file="discriminator.png", show_shapes=True, show_layer_names=False)
    model.summary()
    return model