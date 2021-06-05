import numpy as np
import matplotlib.pyplot as plt
losses = np.load("losses.npy")

plt.plot(losses, '-b', label='Train loss')
#plt.plot(validation_losses, '-r', label='Validation loss')
plt.legend(loc=0)
plt.title('Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()
#print('Iteration: %d, train loss: %.4f' % (i, loss_))
