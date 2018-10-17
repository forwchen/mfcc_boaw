# mfcc_boaw
Extract MFCC from videos and make bag-of-audio-words (BoAW) representations.
MFCCs are extracted from mono-channel audio of given video, then a K-means model is trained on the MFCCs.
BoAW representations are computed by method of this paper: [Softening quantization in bag-of-audio-words](https://ieeexplore.ieee.org/document/6853821).

## Usage

### 0.Split the video dataset


### 1.Extract audio and then MFCCs


### 2.Train the K-means model


### 3.Compute BoAW representations
