# mfcc_boaw
Extract MFCC from videos and make bag-of-audio-words (BoAW) representations.
MFCCs are extracted from mono-channel audio of given video, then a K-means model is trained on the MFCCs.
BoAW representations are computed by method of this paper: [Softening quantization in bag-of-audio-words](https://ieeexplore.ieee.org/document/6853821).

## Usage

### 0.Split the video dataset with `split_video_dataset.py`
```
usage: split_video_dataset.py [-h] vid_dir num_splits split_file

positional arguments:
  vid_dir     the video directory
  num_splits  the number of splits
  split_file  the split stored as pickle file

optional arguments:
  -h, --help  show this help message and exit
```

Sample usage: `python split_video_dataset.py 1 split-sample.pkl`

**Note**

There is no need for more than 1 split unless your video dataset is really large (e.g. more than 100k videos), since extracting audio and MFCC is fast.

### 1.Extract audio and then MFCCs with `vid2mfcc.py`
```
usage: vid2mfcc.py [-h] [-n NFFT] [-w WINLEN] [-s WINSTEP]
                   split_file split mfcc_db

positional arguments:
  split_file            the pickled split file
  split                 the split to use, e.g. split-0
  mfcc_db               the database to store extracted frames, HDF5 format

optional arguments:
  -h, --help            show this help message and exit
  -n NFFT, --nfft NFFT  the FFT size
  -w WINLEN, --winlen WINLEN
                        the length of the analysis window in seconds
  -s WINSTEP, --winstep WINSTEP
                        the step between successive windows in seconds
```

Sample usage: `python vid2mfcc.py split-sample.pkl split-0 mfcc_db.hdf5 -n 2048 -w 0.04 -s 0.02`

**Note**

* There are 3 tunable parameters for MFCC, `nfft`, `winlen` and `winstep`.
* The recommended values are 2048, 0.04 and 0.02, respectively. Adjust them if you need to.

### 2.Train the K-means model with `kmeans.py`
```
usage: kmeans.py [-h] [-k CLUSTER] [-n NSAMPLE] mfcc_db

positional arguments:
  mfcc_db               the database to store extracted frames, HDF5 format

optional arguments:
  -h, --help            show this help message and exit
  -k CLUSTER, --cluster CLUSTER
                        the number of clusters for kmeans
  -n NSAMPLE, --nsample NSAMPLE
                        the number of samples
```

Sample usage: `python kmeans.py mfcc_db.hdf5 -k 128 -n 250000`

**Note**

* The recommended value for `cluster` is 128, which is also the dimension for BoAW representations.
* The number of samples depends on the size of the total MFCC data in `mfcc_db`, usually 1/10 of `mfcc_db` should suffice.

### 3.Compute BoAW representations with `boaw_extract.py`
```
usage: boaw_extract.py [-h] mfcc_db boaw_db kmeans_model

positional arguments:
  mfcc_db       the database to store extracted frames, HDF5 format
  boaw_db       the database to store bag-of-audio-words, HDF5 format
  kmeans_model  the trained kmeans model

optional arguments:
  -h, --help    show this help message and exit
```
Sample usage: `python boaw_extract.py vtt_mfcc.hdf5 vtt_boaw.hdf5 kmeans_model.pkl`


## Dependencies
* Python 2.7
* FFmpeg: Install on Ubuntu. Other platforms.
* Python libraries: pip install -r requirements.txt
