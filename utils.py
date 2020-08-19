
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def generate_and_save_images(model, epoch, test_input, test_label, figsize=(5,10), nrows=10, ncols=5):
  predictions = model([test_input, test_label], training=False)

  fig = plt.figure(figsize=figsize)

  for i in range(predictions.shape[0]):
      sub = plt.subplot(nrows, ncols, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')
        
  plt.savefig('result_{:04d}.png'.format(epoch))
  #plt.show()    