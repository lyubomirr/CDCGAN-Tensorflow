import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'

import tensorflow as tf
import numpy as np
import time
import datetime
import glob

from IPython import display
from utils import generate_and_save_images
from checkpoint import *

BUFFER_SIZE = 60000
BATCH_SIZE = 128
EPOCHS = 50

def get_dataset():
  (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

  train_labels = tf.one_hot(train_labels, NUM_CLASSES)
  train_images = train_images.reshape(train_images.shape[0], IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS).astype('float32')
  train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]

  # Batch and shuffle the data
  return tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.random.uniform(real_output.shape.as_list(), minval=0.7, maxval=1.2), real_output)
    fake_loss = cross_entropy(tf.random.uniform(fake_output.shape.as_list(), minval=0.0, maxval=0.3), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

                            

@tf.function
def train_step(images, labels):
    noise = tf.random.normal([images.shape[0], NOISE_DIMS])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator([noise, labels], training=True)

      real_output = discriminator([images, labels], training=True)
      fake_output = discriminator([generated_images, labels], training=True)

      disc_loss = discriminator_loss(real_output, fake_output)
      gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
      discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        
      fake_output2 = discriminator([generated_images, labels], training=True)

      gen_loss = generator_loss(fake_output2)
      gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
      generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
      
      return disc_loss, gen_loss
      
def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()
     
    step = 1
    for image_batch, label_batch in dataset:
      d_loss, g_loss = train_step(image_batch, label_batch)
      step += 1
      if(step % 100 == 0):
        print("Epoch: {}, Step: {}".format(epoch, step))
        tf.print("Discriminator loss: ", d_loss)
        tf.print("Generator loss: ", g_loss)
 
    # Produce images for the GIF as we go
    display.clear_output(wait=True)
    generate_and_save_images(generator, epoch + 1, seed, labels_seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator, epochs, seed, labels_seed)


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
num_examples_to_generate = 50
seed = tf.random.normal([num_examples_to_generate, NOISE_DIMS])
labels_seed = tf.one_hot(np.repeat(np.arange(10), 5), NUM_CLASSES)

def main():
    train_dataset = get_dataset()
    train(train_dataset, EPOCHS)
	
if __name__ == "__main__":
    main()