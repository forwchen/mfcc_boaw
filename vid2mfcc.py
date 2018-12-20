import os
import sys
import lmdb
import h5py
import numpy as np
import argparse
from tqdm import tqdm
from subprocess import call
import cPickle as pickle
from python_speech_features import mfcc
import scipy.io.wavfile as wav


def read_img(path):
    with open(path, 'rb') as f:
        return f.read()


parser = argparse.ArgumentParser()
parser.add_argument("split_file", type=str, help="the pickled split file")
parser.add_argument("split", type=str, help="the split to use, e.g. split-0")
parser.add_argument("mfcc_db", type=str, help="the database to store extracted frames, HDF5 format")
# MFCC options

parser.add_argument("-n", "--nfft", type=int, default=2048, help="the FFT size")
parser.add_argument("-w", "--winlen", type=float, default=0.025, help="the length of the analysis window in seconds")
parser.add_argument("-s", "--winstep", type=float, default=0.01, help="the step between successive windows in seconds")
args = parser.parse_args()



split = pickle.load(open(args.split_file,'rb'))
print split.keys(), 'using %s' %(args.split)
all_videos = split[args.split]

db = h5py.File(args.mfcc_db, 'a') # append mode

tmp_dir = '/tmp'

done_videos = set()

for vid in tqdm(all_videos, ncols=64):
    #vvid = vid.split('/')[-1].split('.')[0]
    vvid, _ = os.path.splitext(vid) # discard extension
    _, vvid = os.path.split(vvid)   # get filename without path
    if vvid in done_videos:
        print 'video %s seen before, ignored.' % vvid

    v_dir = os.path.join(tmp_dir, vvid)
    call(["rm", "-rf", v_dir])
    os.mkdir(v_dir)    # caching directory to store ffmpeg extracted frames

    wav_out = "%s/%s.wav" % (v_dir, vvid)
    # Step 1. extract audio
    call(["ffmpeg", "-loglevel", "panic", "-i", vid, "-q:a", "0", "-c:a", "pcm_f32le", "-ac", "1", wav_out])
    if not os.path.exists(wav_out):
        print 'this video %s does not have audio channel' % vvid
        call(["rm", "-rf", v_dir])
        continue

    # Step 2. extract MFCC
    rate, sig = wav.read(wav_out)
    try:
        mfcc_feat = mfcc(sig, rate, nfft=args.nfft, winlen=args.winlen, winstep=args.winstep)
    except:
        print 'failed extracting mfcc for %s' % vvid
        continue
    db[vvid] = mfcc_feat

    call(["rm", "-rf", v_dir])
    #call(["rm", "-f", mono_wav])
    done_videos.add(vvid)



