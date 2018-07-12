import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import librosa 
import math

print("Tensorflow version " + tf.__version__)
tf.set_random_seed(0)

RESCALING_FACTOR = 1
BATCH_LENGTH = int(64*100)

def get_batches(x, batch_size, batch_length=BATCH_LENGTH):
    indexes = np.random.randint(0, high=x.size - batch_length-1, size=batch_size)
    Xbatch = np.zeros([batch_size, batch_length])
    for ii in range(batch_size):
        Xbatch[ii, :] = x[indexes[ii]:indexes[ii]+batch_length]
    
    return Xbatch.reshape([batch_size, batch_length, 1, 1])

def audio2batches(input_audio):
    #cut the audio so that it is exactly 10 seconds
    seconds = int( np.floor( input_audio.size / BATCH_LENGTH ) )
    audio_10s = input_audio[:seconds*BATCH_LENGTH]
    batches = np.reshape(audio_10s, [seconds, BATCH_LENGTH, 1, 1])
    return batches

def batches2audio(input_batches):
    audio_sequence = np.reshape(input_batches, [-1, 1])
    return audio_sequence

def savemusic(input_sequence, sr=BATCH_LENGTH, filename='outputs/output.wav'):
    librosa.output.write_wav(filename, input_sequence, sr=22050)

def down_upsample_vec(x, factor=64):
    out  = x.copy()
    seconds = int( np.floor( out.size / factor ) )
    out = out[:seconds*factor]
    for ii in range(1, factor):
        out[ii::factor] = 0
    return out

def down_upsample(x, factor=64):
    for ii in range(len(x)):
        x[ii, :] = down_upsample_vec(x[ii,:], factor=factor)
    return x

filename = "inputs/bach.wav"
raw_audio, sr = librosa.load(filename, sr=22050) #loads with 22050 samples per second

# savemusic(down_upsample_vec(raw_audio, factor=64))


#savemusic(batches2audio(audio2batches(raw_audio)), sr)

print("audio loaded")

height = math.trunc(BATCH_LENGTH) #every 1 second is taken
# input X
X = tf.placeholder(tf.float32, [None, height, 1, 1])
Xdownup = tf.placeholder(tf.float32, [None, height, 1, 1])

# step for variable learning rate
step = tf.placeholder(tf.int32)

# three convolutional layers with their channel counts, and a
# fully connected layer (tha last layer has 10 softmax neurons)
K = 8  # first convolutional layer output depth
L = 16  # second convolutional layer output depth
M = 24  # third convolutional layer
N = 200  # fully connected layer

W1 = tf.Variable(tf.truncated_normal([500, 1, 1, K], stddev=0.01))  # 5x5 patch, 1 input channel, K output channels
B1 = tf.Variable(tf.ones([K])/1000)
W2 = tf.Variable(tf.truncated_normal([500, 1, K, L], stddev=0.01))
B2 = tf.Variable(tf.ones([L])/1000)
W3 = tf.Variable(tf.truncated_normal([400, 1, L, M], stddev=0.01))
B3 = tf.Variable(tf.ones([M])/1000)

W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.01))
B4 = tf.Variable(tf.ones([N])/1000)
W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.01))
B5 = tf.Variable(tf.ones([10])/1000)


# ENCODER
stride1 = 2  # output is 5513x1x8
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride1, stride1, 1], padding='SAME') + B1)
stride2 = 2  # output is 1379x1x16
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride2, stride2, 1], padding='SAME') + B2)
stride3 = 2  # output is 345x1x24
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride3, stride3, 1], padding='SAME') + B3)

# reshape the output from the third convolution for the fully connected layer
# Y3_flat = tf.reshape(Y3, shape=[-1, 345 * 1 * M])
Y3_flat = tf.layers.flatten(Y3)
dense_size = int(BATCH_LENGTH/stride1/stride2/stride3)
# dense_size = 345

print('Y flat')
print(Y3_flat)
print('dense size')
print(dense_size)

# fully connected layer
Z = tf.layers.dense(Y3_flat, int(dense_size*M), name='z_mean_dense')

print("encoded")

# DECODER
W1d = tf.Variable(tf.truncated_normal([500, 1, 1, K], stddev=0.01))  # 5x5 patch, 1 input channel, K output channels
B1d = tf.Variable(tf.ones([1])/1000)
W2d = tf.Variable(tf.truncated_normal([500, 1, K, L], stddev=0.01))
B2d = tf.Variable(tf.ones([K])/1000)
W3d = tf.Variable(tf.truncated_normal([400, 1, L, M], stddev=0.01))
B3d = tf.Variable(tf.ones([L])/1000)
# fully connected layer
z_fc = tf.layers.dense(Z, int(dense_size*M))

# print()
# print('Z')
# print(Z)
# print('z_fc')
# print(z_fc)
# print()

# first deconvolutional layer 345x1x24 -> 1378x1x16
z_matrix = tf.nn.relu(tf.reshape(z_fc, [-1, dense_size, 1, M]))

batch_size_tensor = tf.shape(X)[0]
h1 = tf.nn.relu(tf.nn.conv2d_transpose(z_matrix, W3d, output_shape=[batch_size_tensor, int(dense_size*stride3), 1, L], strides=[1, stride3, stride3, 1], name='deconv1')+ B3d)
# second deconvolutional layer  1378x1x16 -> 5512x1x8
h2 = tf.nn.relu(tf.nn.conv2d_transpose(h1, W2d, output_shape=[batch_size_tensor, int(dense_size*stride3*stride2), 1, K], strides=[1, stride2, stride2, 1], name='deconv2')+ B2d)
# third deconvolutional layer  5512x1x8 -> 22050x1x1
h3 = tf.nn.conv2d_transpose(h2, W1d, output_shape=[batch_size_tensor, int(dense_size*stride1*stride2*stride3), 1, 1], strides=[1, stride1, stride1, 1], name='deconv3')+ B1d
output_audio = tf.nn.tanh(h3) + Xdownup

print("decoded")

#mean squared loss
loss = tf.multiply(tf.losses.mean_squared_error(output_audio, X), 1e8)

# the learning rate is: # 0.0001 + 0.003 * (1/e)^(step/2000)), i.e. exponential decay from 0.003->0.0001
# lr = 0.0001 +  tf.train.exponential_decay(0.003, step, 2000, 1/math.e)
lr = 0.0001

# Get cost and define the optimizer
cost = tf.reduce_mean(loss)
opt = tf.train.AdamOptimizer(lr).minimize(cost)

# TRAINING
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    Nits = 1000
    # batch_size = 10

    input_rescaled = raw_audio
    Xbatch = audio2batches(raw_audio)
    for iteration in range(Nits):
        # Xbatch = get_batches(input_rescaled, batch_size, batch_length=BATCH_LENGTH)
        feed_dict = {X:Xbatch, Xdownup:down_upsample(Xbatch, factor=8) , step : iteration}
        sess.run(opt, feed_dict)

        if iteration % 10 == 0:
            loss_iter = sess.run(loss, feed_dict)
            print('iteration {}: loss = {}'.format(iteration, loss_iter))

    # TEST ON AUDIO 1 (TRAIN)

    # print(raw_audio)
    Xbatches = audio2batches(raw_audio)
    print(Xbatches.shape)
    out = sess.run(output_audio,  {X: Xbatches, Xdownup:down_upsample(Xbatches, factor=8)})
    out_audio = batches2audio(out)
    # print(out_audio)
    savemusic(out_audio, filename='outputs/output.wav')

    # TEST ON AUDIO 2
    filename = "inputs/midnight.wav"
    audio2, sr = librosa.load(filename, sr=22050)
    Xbatches = audio2batches(audio2)
    print(Xbatches.shape)
    out = sess.run(output_audio,  {X: Xbatches, Xdownup:down_upsample(Xbatches, factor=8)})
    out_audio = batches2audio(out)
    savemusic(out_audio, filename='outputs/output2.wav')

    save_path = saver.save(sess, "/savedmodel/model.ckpt")
    print("Model saved in path: %s" % save_path)

    print('INPUT -- Average value = {}'.format(np.mean(np.abs(raw_audio))))
    print('OUPUT -- Average value = {}'.format(np.mean(np.abs(out_audio))))

    print('INPUT -- MAX value = {}, MIN value {}'.format(np.max(raw_audio), np.min(raw_audio)))
    print('OUPUT -- MAX value = {}, MIN value {}'.format(np.max(out_audio), np.min(out_audio)))
