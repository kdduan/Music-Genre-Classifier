import os
import pathlib
import librosa
import librosa.display
from fastai.vision import *
import numpy as np
from tqdm import tqdm

'''
Def:
    create melspectrogram from audio file and save to folder label
Params:
    filename = audio file name
    title = title of melspectrogram image
    folder = folder to store image
'''
def create_melspec(filename, title, folder):
    y, sr = librosa.load(filename)
    # trim silent edges
    audio, _ = librosa.effects.trim(y)
    n_fft = 2048
    hop_length = 512
    n_mels = 128
    mel = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    S = librosa.feature.melspectrogram(audio, sr=sr, n_fft=n_fft, 
                                       hop_length=hop_length, n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length)
    plt.savefig(f'{folder}/{title[:-4]}.png', bbox_inches='tight', pad_inches=0)

def main():
    orig_path = pathlib.Path().absolute()
    data_path = orig_path/'genres'
    os.mkdir('training_data')
    train_path = orig_path/'training_data'
    
    for path in tqdm(data_path.ls()):
        genre_folder = str(path).split('/')[-1]
        folder = train_path/genre_folder
        os.mkdir(folder)
        for filename in tqdm(os.listdir(path)):
            full_name = path/filename
            create_melspec(full_name, filename, folder)

if __name__ == '__main__':
    main()
