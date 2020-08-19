import tensorflow as tf
import os
from model import *

ADAM_BETA1 = 0.5
LR = 0.0002

generator = make_generator_model()
discriminator = make_discriminator_model()
generator_optimizer = tf.keras.optimizers.Adam(LR, beta_1=ADAM_BETA1)
discriminator_optimizer = tf.keras.optimizers.Adam(LR, beta_1=ADAM_BETA1)

checkpoint_dir = './checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")		
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
								 