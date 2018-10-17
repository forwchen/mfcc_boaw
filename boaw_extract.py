import sys
import h5py
import argparse
import random
random.seed(1234)
import numpy as np
np.random.rand(1234)
import cPickle as pickle
from tqdm import tqdm
from sklearn.cluster import KMeans


parser = argparse.ArgumentParser()
parser.add_argument("mfcc_db", type=str, help="the database to store extracted frames, HDF5 format")
parser.add_argument("boaw_db", type=str, help="the database to store bag-of-audio-words, HDF5 format")
parser.add_argument("kmeans_model", type=str, help="the trained kmeans model")
args = parser.parse_args()


mfcc_db = h5py.File(args.mfcc_db, 'r')
boaw_db = h5py.File(args.boaw_db, 'w')
kmeans = pickle.load(file(args.kmeans_model, 'rb'))

K = kmeans.cluster_centers_.shape[0]


for vid in tqdm(mfcc_db.keys(), ncols=64):
    mfcc = mfcc_db[vid]
    t = kmeans.predict(mfcc)
    u, c = np.unique(t, return_counts=True)
    h = np.zeros((K,), dtype=np.float32)
    for u_, c_ in zip(u,c):
        h[u_] = c_
    h /= np.linalg.norm(h, 2)

    boaw_db[vid] = h


