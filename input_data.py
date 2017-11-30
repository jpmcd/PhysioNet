from __future__ import division

import numpy as np
import scipy.io as sio
import scipy.signal as signal
import matplotlib.pyplot as plt

import os.path
import pickle
from random import shuffle


def open_data(mat_path, save_path):
  '''
    For each file open the signal as a numpy array and append to
    list, append label to separate list, and pickle.

    mat_path = path to .mat files
    save_path = pickled lists of numpy arrays
  '''

  with open('files.txt') as F:
    filenames = [f.strip() for f in F.readlines()]

  signals = []
  for f in filenames:
    mat = sio.loadmat(mat_path+f)
    val = np.squeeze(mat['val'])
    signals.append(val)

  labels = []
  with open(mat_path+'REFERENCE.csv') as L:
    for line in L.readlines():
      key, lab = line.strip().split(',')

      if lab == 'N':
        labels.append(0)
      elif lab == 'A':
        labels.append(1)
      elif lab == 'O':
        labels.append(2)
      elif lab == '~':
        labels.append(3)
      else:
        errorstr = 'Unexpected label: %s, %s '%(key, lab)
        raise ValueError(errorstr)

  with open(save_path, 'wb') as datafile:
    pickle.dump(signals, datafile)
    pickle.dump(labels, datafile)

  return


def split_datasets(data_path, n_folds):
  '''
    Load the signals and labels from pickle file, partition training set into n_folds partitions
    Ensure that the same ratio of different labels is in each training set
  '''

  with open(data_path, 'rb') as datafile:
    signals = pickle.load(datafile)
    labels = pickle.load(datafile)

  label_ct = [0, 0, 0, 0]
  category_str = ['Normal', 'Arrhythmia', 'Other', 'Noisy']
  category_inds = [[] for cat in category_str]
  for i, l in enumerate(labels):
    label_ct[l] += 1
    category_inds[l].append(i)
  total = sum(label_ct)

  train_inds = [[] for i in range(n_folds)]

  for i in range(len(label_ct)):
    print 'Percent %s = %d / %d, %f' % (category_str[i], label_ct[i], total, label_ct[i]/total)
    
    catlen = label_ct[i]
    shuffle(category_inds[i])
    for j in range(n_folds):
      start = int(np.round(j/n_folds*catlen))
      end = int(np.round((j+1)/n_folds*catlen))
      train_inds[j].extend(category_inds[i][start:end])

  for i in range(n_folds):
    train_inds[i].sort()

  split_x = [[] for i in range(n_folds)]
  split_y = [[] for i in range(n_folds)]

  for fold, fold_inds in enumerate(train_inds):
    for i in fold_inds:
      split_x[fold].append(signals[i])
      split_y[fold].append(labels[i])

  split = [split_x, split_y]

  return split


def prepare_dataset(data_x, data_y, maxlen=None):
  '''
    Make numpy-friendly dataset from signals and labels
  '''

  data_y = list(data_y)
  nsamples = len(data_y)
  lengths = [len(sig) for sig in data_x]

  if not maxlen:
    maxlen = min(lengths)

  x = np.zeros((nsamples, maxlen), dtype=float)
  y = np.array(data_y, dtype=int)
  for i, sig in enumerate(data_x):
    end = min(maxlen, lengths[i])
    x[i,:end] = sig[:end]

##WHY AM I EXPANDING DIMENSION HERE?############################
  #x = np.expand_dims(x, axis=2) #removing this to see if it causes issues

  return x, y


def get_minibatch_inds(n_samples, batch_size):
  #Batch input pipeline to give randomized minibatches for epochs

  shuffled = np.random.permutation(n_samples)
  n_batches = n_samples//batch_size
  batch_inds = [shuffled[i*batch_size:(i+1)*batch_size] for i in range(n_batches)]

  return batch_inds


if __name__ == '__main__':

  if False:
    key = 'A00004'
    d = sio.loadmat(path+key+'.mat')
    val = np.squeeze(d['val'])
    t = np.arange(val.size)/300.
    plt.plot(t,val)
    plt.show()

    f, t, Zxx = signal.stft(val, fs=300, nperseg=1500, noverlap=1350, padded=False)
    print Zxx.shape
    #print f[0:50]

    plt.pcolormesh(t, f[0:100], np.abs(Zxx[0:100,:]), )
    plt.show()

  pickle_path = '/scratch/PhysioNet/data.pkl'
  path = '/home/mcdonald/Desktop/training2017/'

  #Test to see if variance(x[i]-x[i-1]) is different for noisy signals vs other signals
  if not os.path.isfile(pickle_path):
    input_data.open_data(path, pickle_path)
  
  with open(pickle_path, 'rb') as datafile:
    signals = pickle.load(datafile)
    labels = pickle.load(datafile)

  label_ct = [0, 0, 0, 0]
  n_categories = len(label_ct)

  category_var = [[] for i in range(n_categories)]
  for i, sig in enumerate(signals):
    x_i = np.array(sig[1:])
    x_im1 = np.array(sig[0:-1])
    diff = x_i - x_im1
    var = np.var(diff)
    category_var[labels[i]].append(var)
    label_ct[labels[i]] += 1

  mean_var = [np.mean(variances) for variances in category_var]

  print mean_var
  print label_ct
  for i in range(4):
    print np.random.choice(category_var[i], 10, replace=False)




