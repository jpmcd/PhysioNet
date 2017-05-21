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
    For each file convert the signal to an STFT and append to
    list, append label to separate list, and pickle.

    mat_path = path to .mat files
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
        errorstr = 'Unexpected label: '+key+','+lab
        raise ValueError(errorstr)

  with open(save_path,'wb') as datafile:
    pickle.dump(signals, datafile)
    pickle.dump(labels,datafile)

  return

def split_datasets(data_path, valid_pct):
  #Load the STFTs and labels from pickle file, make a training and validation set

  with open(data_path,'rb') as datafile:
    dataset = pickle.load(datafile)
    labels = pickle.load(datafile)

  label_ct = [0, 0, 0, 0]
  category_str = ['Normal', 'Arrhythmia', 'Other', 'Noisy']
  category_inds = [[]]*4
  for i,l in enumerate(labels):
    label_ct[l] += 1
    category_inds[l].append(i)
  total = sum(label_ct)

  train_inds = []
  valid_inds = []
  for i in range(4):
    print 'Percent %s = %d / %d, %f' % (category_str[i], label_ct[i], total, label_ct[i]/total)

    catlen = len(category_inds[i])
    ntrain = int(np.round(catlen*(1-valid_pct)))
    shuffle(category_inds[i])
    train_inds.extend(category_inds[i][0:ntrain])
    valid_inds.extend(category_inds[i][ntrain:])

  train_inds.sort()
  valid_inds.sort()

  train_x = []
  train_y = []
  valid_x = []
  valid_y = []

  for i in train_inds:
    train_x.append(dataset[i])
    train_y.append(labels[i])

  for i in valid_inds:
    valid_x.append(dataset[i])
    valid_y.append(labels[i])

  train = (train_x, train_y)
  valid = (valid_x, valid_y)

  return train, valid


def prepare_dataset(data_x, data_y, maxlen=None):
  #Make numpy-friendly dataset from signals and labels

  nsamples = len(data_y)
  lengths = [len(sig) for sig in data_x]

  if not maxlen:
    maxlen = min(lengths)

  x = np.zeros((nsamples, maxlen)).astype('float32')
  y = np.array(data_y).astype('int32')
  for i, sig in enumerate(data_x):
    end = min(maxlen, lengths[i])
    x[i,:end] = sig[:end]

  x = np.expand_dims(x, axis=2)

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


