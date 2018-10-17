import sys
import h5py
import time
import argparse
import cPickle as pickle
import numpy as np
np.random.seed(1234)
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser()
parser.add_argument("mfcc_db", type=str, help="the database to store extracted frames, HDF5 format")
parser.add_argument("-k", "--cluster", type=int, default=128, help="the number of clusters for kmeans")
parser.add_argument("-n", "--nsample", type=int, default=1000000, help="the number of samples")
args = parser.parse_args()

db = h5py.File(args.mfcc_db, 'r')

K = args.cluster

mfccs = db

words = []
for vid in mfccs:
    words.append(np.asarray(mfccs[vid]))

words = np.concatenate(words, axis=0)

kmeans = KMeans(K, n_jobs=20)

sample_idx = np.linspace(0, words.shape[0], args.nsample, endpoint=False, dtype=np.int32)

print 'Sampling %d from %d samples' % (args.nsample, words.shape[0])

words_t = words[sample_idx]
N = words_t.shape[0]

print 'Start running kmeans ...'
t0 = time.time()
kmeans.fit(words_t)
print 'Time for running kmeans:', time.time() - t0

pickle.dump(kmeans, file('kmeans_model-%d-%d.pkl' % (N, K), 'wb'), 2)
