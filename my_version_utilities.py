import keras
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import vgg19
from keras import backend as K

#converts audio files to Mel spectogram format by using librosa
def conv_to_spectogram_mel(y, sr):
    melSpec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128) 
    # Convert to log scale (dB). We'll use the peak power (max) as reference.
    spectogram = librosa.power_to_db(melSpec, ref=np.max)
    return spectogram

#separates harmonic and percussive elements of the file
def isolate_harm_perc(y, sr):
    harmonic_y, percussive_y = librosa.effects.hpss(y)
    harmonic = conv_to_spectogram_mel(harmonic_y, sr)
    percussive = conv_to_spectogram_mel(percussive_y, sr)
    return harmonic, percussive

#creates a spectogram and saves it to the digitrectory (modified from librosa tutorial)
def display_spectogram(spectogram, sr, title):
    plt.figure(dpi=1200)
    # Display the spectrogram on a mel scale
    librosa.display.specshow(spectogram, sr=sr, x_axis='time', y_axis='mel')
    # Make the figure layout compact
    #plt.tight_layout()
    #save to directory
    plt.imsave(title, spectogram[:1000, :571]) 

#preprocesses the spectogram to make it compatible with vgg19
def preprocess_image(image_path, img_height, img_width):
    img = load_img(image_path, target_size=(img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

#deprocesses the spectogram
def deprocess_image(x):
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

#define the content loss
def content_loss(base, combination):
    return K.sum(K.square(combination - base))

#define the style loss
def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

def style_loss(style, combination, img_height, img_width):
        S = gram_matrix(style)
        C = gram_matrix(combination)
        channels = 3
        size = img_height * img_width
        return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

#define the total variation loss
def total_variation_loss(x, img_height, img_width):
        a = K.square(
            x[:, :img_height - 1, :img_width - 1, :] -
            x[:, 1:, :img_width - 1, :])
        b = K.square(
            x[:, :img_height - 1, :img_width - 1, :] -
            x[:, :img_height - 1, 1:, :])
        return K.sum(K.pow(a + b, 1.25))


#preprocesses the input signal by performing a short time fourier transform on it 
#(modified from Ulyanov's implementation)
def preprocess(y):
    processed_signal = librosa.stft(y)
    phases = np.angle(processed_signal)
    processed_signal = np.log1p(np.abs(processed_signal[:,:430]))
    return processed_signal

#reconstruct the signal
#(modified from Ulyanov's implementation)
def deprocess(y, sr, outfile):
    p = 2 * np.pi * np.random.random_sample(y.shape) - np.pi
    for i in range(500):
        S = y * np.exp(1j * p)
        deprocessed_signal = librosa.istft(S)
        p = np.angle(librosa.stft(deprocessed_signal))
    librosa.output.write_wav(outfile, deprocessed_signal, sr)

