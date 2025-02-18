from input_data import get_labels,get_images
from sklearn import svm
import pickle
import numpy as np

train_data = get_images('Input_Data/train-images-idx3-ubyte/train-images.idx3-ubyte', length=50000)
train_labels = get_labels('Input_Data/train-labels-idx1-ubyte/train-labels.idx1-ubyte')

# Setup the kernel functions here
clf = svm.SVC(kernel="rbf")
print(f"INFO: Using {clf.kernel} kernels")
print(f"INFO: Using hyperparameters of C={clf.C}, gamma={clf.gamma}")
train_data = np.asarray(train_data[:(50000*784)]).reshape(50000, 784)

clf.fit(train_data, train_labels[:50000])

# save the model to disk
filename = 'finalized_model_50000_f.sav'
pickle.dump(clf, open(filename, 'wb'))
print("Succeed!")