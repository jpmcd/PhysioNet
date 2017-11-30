import sys

JOEY_PATH = '/home/mcdonald/Desktop/training2017/'
JOEY_PICKLE_PATH = '/scratch/PhysioNet/data.pkl'

STEVEN_PATH = '/mnt/datadrive/projects/physionet/data/training2017/'
STEVEN_PICKLE_PATH = '/mnt/datadrive/projects/physionet/data/data.pkl'

STEVEN_LAPTOP_PATH = '/Users/steven/Documents/projects/ekg/data/training2017/'
STEVEN_LAPTOP_PICKLE_PATH = '/Users/steven/Documents/projects/ekg/data/training2017/data.pkl'

if sys.platform == 'linux2':
  PATH = JOEY_PATH
  PICKLE_PATH = JOEY_PICKLE_PATH
else:
  PATH = STEVEN_PATH
  PICKLE_PATH = STEVEN_PICKLE_PATH
