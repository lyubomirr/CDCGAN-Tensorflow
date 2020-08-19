import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from checkpoint import *
from utils import generate_and_save_images

def main():
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    generate_and_save_images(generator, 0, tf.random.normal([100, NOISE_DIMS]), 
                            	tf.one_hot(np.repeat(np.arange(10), 10), NUM_CLASSES), (10, 10), 10, 10)
	
if __name__ == "__main__":
    main()