import keras
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio, display #for testing purposes
from keras import backend as K
from my_version_utilities import *
from keras.applications import vgg19
from keras.preprocessing.image import load_img, img_to_array
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
import time

#content and style sounds
#resamples the sound to 22050 Hz
content_path, sr_content = librosa.load("inputs/imperial.wav") 
style_path, sr_style = librosa.load("inputs/usa.wav") 

#creates spectograms and saves them in the directory with appropriate names
display_spectogram((conv_to_spectogram_mel(content_path, sr_content)), sr_content, "content_spec.png")
display_spectogram((conv_to_spectogram_mel(style_path, sr_style)), sr_style, "style_spec.png")

#update the new content and style paths
content_path = "content_spec.png"
style_path ="style_spec.png"

img_height = 1000
img_width = 571

#store content and style spectograms as constants
content = K.constant(preprocess_image(content_path, img_height, img_width)) 
style = K.constant(preprocess_image(style_path, img_height, img_width)) 
combination = K.placeholder((1, img_height, img_width, 3))

input_tensor = K.concatenate([content, style, combination], axis=0)

model = vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False) 
print('Model loaded.')

#define the final loss which we will minimize
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers]) 
content_layer = 'block5_conv2'
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']
total_variation_weight = 1e-4
style_weight = 1.
content_weight = 0.025
loss = K.variable(0.)
layer_features = outputs_dict[content_layer]
target_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(target_image_features, combination_features)

for layer_name in style_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features, img_height, img_width)
    loss += (style_weight / len(style_layers)) * sl

loss += total_variation_weight * total_variation_loss(combination, img_height, img_width)

#set up the gradient-descent process
grads = K.gradients(loss, combination)[0]
fetch_loss_and_grads = K.function([combination], [loss, grads])

class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, img_height, img_width, 3))
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value
    
    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()

#run the gradient-ascent process
result_prefix = 'my_result'
iterations = 2

x = preprocess_image(content_path, img_height, img_width) 
x = x.flatten()

for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x, fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    img = x.copy().reshape((img_height, img_width, 3))
    img = deprocess_image(img)
    fname = result_prefix + '_at_iteration_%d.png' % i
    imsave(fname, img)
    print('Image saved as', fname)
    end_time = time.time()
    print('Iteration %d completed in %ds' % (i, end_time - start_time))

