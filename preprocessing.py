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

def create_train_val(data_path, train_folder, val_folder, train_count):
    for path in tqdm(data_path.ls()):
        genre_folder = str(path).split('/')[-1]
        os.mkdir(train_folder/genre_folder)
        os.mkdir(val_folder/genre_folder)
        train_counter = 0
        for filename in tqdm(os.listdir(path)):
            train_counter += 1
            full_name = path/filename
            if train_counter <= train_count:
                create_melspec(full_name, filename, train_folder/genre_folder)
            else:
                create_melspec(full_name, filename, val_folder/genre_folder)
    
def main():
    orig_path = pathlib.Path().absolute()
    data_path = orig_path/'genres'
    os.mkdir('train')
    os.mkdir('valid')
    train_path = orig_path/'train'
    val_path = orig_path/'valid'
    create_train_val(data_path, train_path, val_path, 20)


if __name__ == '__main__':
    main()
