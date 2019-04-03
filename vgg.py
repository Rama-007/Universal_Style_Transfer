import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, UpSampling2D, Activation, Lambda, MaxPooling2D, Dense
from keras.models import Sequential, Model, load_model
from keras.layers import InputLayer, Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding, TimeDistributed, Bidirectional
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.merge import Concatenate, Multiply, Add 
from keras.layers.recurrent import LSTM, SimpleRNN, GRU 
from keras.regularizers import l2
import torchfile


def pad_reflect(x, padding=1):
	return tf.pad(x, [[0, 0], [padding, padding], [padding, padding], [0, 0]],mode='REFLECT')


def vgg(filename,target_layer):
	modell=torchfile.load(filename,force_8bytes_long=True)
	input11=Input(shape=(None,None,3))
	input1=input11
	for i in range(0,len(modell.modules)):
		if modell.modules[i].name is not None:
			name=modell.modules[i].name.decode()
		else:
			name = None

		if modell.modules[i]._typename == b'nn.SpatialReflectionPadding':
			input1=Lambda(pad_reflect)(input1)
		elif modell.modules[i]._typename == b'nn.SpatialConvolution':
			filters = modell.modules[i].nOutputPlane
			kernel_size = modell.modules[i].kH
			weight = modell.modules[i].weight.transpose([2,3,1,0])
			bias = modell.modules[i].bias
			input1 = Conv2D(filters, kernel_size, padding='valid', activation=None, name=name,
						kernel_initializer=lambda shape: K.constant(weight, shape=shape),
						bias_initializer=lambda shape: K.constant(bias, shape=shape),
						trainable=False)(input1)
		elif modell.modules[i]._typename == b'nn.ReLU':
			input1 = Activation('relu', name=name)(input1)
		elif modell.modules[i]._typename == b'nn.SpatialMaxPooling':
			input1 = MaxPooling2D(padding='same', name=name)(input1)

		if name==target_layer:
			break
	model=Model(inputs=input11,outputs=input1)
	return model


