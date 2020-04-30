# Music Genre Classifier 
This project used transfer learning to train a CNN to classify music genres using melspectrogram images of audio files. The model was trained in Google Colab for GPU access. 

## Data
The data came from the [GITZAN dataset](http://opihi.cs.uvic.ca/sound/genres.tar.gz) used in the paper “ [Musical genre classification of audio signals](https://ieeexplore.ieee.org/document/1021072) “ by G. Tzanetakis and P. Cook in IEEE Transactions on Audio and Speech Processing 2002.

The dataset contains 1000 total 30 second audio clips split equally among 10 genres: blues, classical, country, disco, hiphop, jazz, reggae, rock, metal and pop. 

## Packages
* Python 3.6.5
* Numpy, Pandas, Matplotlib
* Librosa - 0.6.2
* fastai v1
* PyTorch 1.2.0

## Results
The final model was able to achieve 75.88% accuracy compared to 61% accuracy for the model proposed in the paper. Based on an experiment conducted in the paper, human accuracy is roughly 70%.