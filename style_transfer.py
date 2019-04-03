from __future__ import print_function, division

import argparse
import functools
import time
import tensorflow as tf, numpy as np, os, random
import threading
from vgg import vgg
from keras import backend as K
from keras.models import Model
from keras.layers import Input, UpSampling2D, Lambda
from collections import namedtuple
import tensorflow as tf
import numpy as np
from keras.layers import Conv2D, Lambda
from tensorflow.python.layers import utils
import cv2
import argparse
from coral import coral_numpy

def get_img_random_crop(src, resize=512, crop=256):
	img = cv2.imread(src)
	img = cv2.resize(img,(crop,crop))
	return img

def get_img(src):
	img = cv2.imread(src)
	if not (len(img.shape) == 3 and img.shape[2] == 3):
	   img = np.dstack((img,img,img))
	return img


def get_files(img_dir):
	files = os.listdir(img_dir)
	paths = []
	for x in files:
		paths.append(os.path.join(img_dir, x))
	return paths

def preserve_colors_np(style_rgb, content_rgb):
	coraled = coral_numpy(style_rgb/255., content_rgb/255.)
	coraled = np.uint8(np.clip(coraled, 0, 1) * 255.)
	return coraled

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoints', nargs='+', type=str, help='List of checkpoint directories', required=True)
parser.add_argument('--relu-targets', nargs='+', type=str, help='List of reluX_1 layers, corresponding to --checkpoints', required=True)
parser.add_argument('--content-path', type=str, dest='content_path', help='Content image or folder of images')
parser.add_argument('--style-path', type=str, dest='style_path', help='Style image or folder of images')
parser.add_argument('--out-path', type=str, dest='out_path', help='Output folder path')
parser.add_argument('--alpha', type=float, help="Alpha blend value", default=1)

args = parser.parse_args()



def wct_tf(content, style, alpha, eps=1e-8):
	content_t = tf.transpose(tf.squeeze(content), (2, 0, 1))
	style_t = tf.transpose(tf.squeeze(style), (2, 0, 1))

	Cc, Hc, Wc = tf.unstack(tf.shape(content_t))
	Cs, Hs, Ws = tf.unstack(tf.shape(style_t))
	content_flat = tf.reshape(content_t, (Cc, Hc*Wc))
	style_flat = tf.reshape(style_t, (Cs, Hs*Ws))

	mc = tf.reduce_mean(content_flat, axis=1, keep_dims=True)
	fc = content_flat - mc
	fcfc = tf.matmul(fc, fc, transpose_b=True) / (tf.cast(Hc*Wc, tf.float32) - 1.) + tf.eye(Cc)*eps
	ms = tf.reduce_mean(style_flat, axis=1, keep_dims=True)
	fs = style_flat - ms
	fsfs = tf.matmul(fs, fs, transpose_b=True) / (tf.cast(Hs*Ws, tf.float32) - 1.) + tf.eye(Cs)*eps
	with tf.device('/cpu:0'):  
		Sc, Uc, _ = tf.svd(fcfc)
		Ss, Us, _ = tf.svd(fsfs)
	k_c = tf.reduce_sum(tf.cast(tf.greater(Sc, 1e-5), tf.int32))
	k_s = tf.reduce_sum(tf.cast(tf.greater(Ss, 1e-5), tf.int32))
	Dc = tf.diag(tf.pow(Sc[:k_c], -0.5))
	fc_hat = tf.matmul(tf.matmul(tf.matmul(Uc[:,:k_c], Dc), Uc[:,:k_c], transpose_b=True), fc)
	Ds = tf.diag(tf.pow(Ss[:k_s], 0.5))
	fcs_hat = tf.matmul(tf.matmul(tf.matmul(Us[:,:k_s], Ds), Us[:,:k_s], transpose_b=True), fc_hat)
	fcs_hat = fcs_hat + ms
	blended = alpha * fcs_hat + (1 - alpha) * (fc + mc)
	blended = tf.reshape(blended, (Cc,Hc,Wc))
	blended = tf.expand_dims(tf.transpose(blended, (1,2,0)), 0)
	return blended




class UST(object):

	def __init__(self, relu_targets=['relu5_1','relu4_1','relu3_1','relu2_1','relu1_1'], *args, **kwargs):

		self.style_input=tf.placeholder_with_default(tf.constant([[[[0.,0.,0.]]]]), shape=(None, None, None, 3))
		self.alpha=tf.placeholder_with_default(1.,shape=[])
		self.encoder_decoders=[]

		with tf.name_scope("vgg_encoder"):
			self.vgg_model=vgg("models/vgg_normalised.t7",sorted(relu_targets)[-1])

		with tf.name_scope("style_encoder"):
			style_layers=[self.vgg_model.get_layer(relu).output for relu in relu_targets]
			style_model=Model(inputs=self.vgg_model.input, outputs=style_layers)
			style_encodings=style_model(self.style_input)

		for i, relu in enumerate(relu_targets):
			if i==0:
				input_tensor=None
			else:
				input_tensor=clip(self.encoder_decoders[-1].decoded)
			enc_dec = self.build_model(relu, input_tensor=input_tensor,**kwargs)
			self.encoder_decoders.append(enc_dec)
		self.content_input  = self.encoder_decoders[0].content_input
		self.decoded_output = self.encoder_decoders[-1].decoded

	def build_model(self,relu_target, input_tensor,batch_size=8, feature_weight=1, pixel_weight=1,learning_rate=1e-4, lr_decay=5e-5):
		
		with tf.name_scope("encoder_decoder_"+relu_target):
			with tf.name_scope("content_encoder_"+relu_target):
				if input_tensor is None:
					content_imgs = tf.placeholder_with_default(tf.constant([[[[0.,0.,0.]]]]), shape=(None, None, None, 3), name='content_imgs')
				else:
					content_imgs = input_tensor
				content_encoding_layers = self.vgg_model.get_layer(relu_target).output
				content_encoder_model = Model(inputs=self.vgg_model.input, outputs=content_encoding_layers)
				content_encoded = content_encoder_model(content_imgs)

		with tf.name_scope("wct_"+relu_target):
			decoder_input=wct_tf(content_encoded, style_encoded_tensor, self.alpha)

		with tf.name_scope("decoder_"+relu_target):
			n_channels = content_encoded.get_shape()[-1].value
			decoder_model = self.build_decoder(input_shape=(None, None, n_channels), relu_target=relu_target)
			decoder_input_wrapped = tf.placeholder_with_default(decoder_input, shape=[None,None,None,n_channels])
			decoded = decoder_model(Lambda(lambda x: x)(decoder_input_wrapped))

		decoded_encoded = content_encoder_model(decoded)

		pixel_loss, feature_loss, tv_loss, total_loss, train_op, global_step, learning_rate, summary_op = [None]*8

		encoder_decoder = EncoderDecoder(content_input=content_imgs, 
										 content_encoder_model=content_encoder_model,
										 content_encoded=content_encoded,
										 decoder_input=decoder_input,
										 decoder_model=decoder_model,
										 decoded=decoded,
										 decoded_encoded=decoded_encoded,
										 pixel_loss=pixel_loss,
										 feature_loss=feature_loss,
										 total_loss=total_loss,
										 train_op=train_op,
										 global_step=global_step,
										 learning_rate=learning_rate,
										 summary_op=summary_op)
		return encoder_decoder
	def build_decoder(self, input_shape, relu_target):

		decoder_num = dict(zip(['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'], range(1,6)))[relu_target]

		decoder_archs = {
			5: [                                    
				(Conv2DReflect, 512), 
				(UpSampling2D,),      
				(Conv2DReflect, 512), 
				(Conv2DReflect, 512), 
				(Conv2DReflect, 512)],
			4: [
				(Conv2DReflect, 256), 
				(UpSampling2D,),      
				(Conv2DReflect, 256), 
				(Conv2DReflect, 256), 
				(Conv2DReflect, 256)],
			3: [
				(Conv2DReflect, 128), 
				(UpSampling2D,),      
				(Conv2DReflect, 128)], 
			2: [
				(Conv2DReflect, 64),   
				(UpSampling2D,)],      
			1: [
				(Conv2DReflect, 64)]   
		}
		code = Input(shape=input_shape, name='decoder_input_'+relu_target)
		x = code
		decoders = reversed(range(1, decoder_num+1))
		count = 0        
		for d in decoders:
			for layer_tup in decoder_archs[d]:

				layer_name = '{}_{}'.format(relu_target, count)

				if layer_tup[0] == Conv2DReflect:
					x = Conv2DReflect(layer_name, filters=layer_tup[1], kernel_size=3, padding='valid', activation='relu', name=layer_name)(x)
				elif layer_tup[0] == UpSampling2D:
					x = UpSampling2D(name=layer_name)(x)
				
				count += 1

		layer_name = '{}_{}'.format(relu_target, count) 
		output = Conv2DReflect(layer_name, filters=3, kernel_size=3, padding='valid', activation=None, name=layer_name)(x)  
		
		decoder_model = Model(code, output, name='decoder_model_'+relu_target)
				
		return decoder_model


def add_dim(image):
	if(len(image.shape)==3):
		image = np.expand_dims(image, 0)
	return image/255.0



class WCT(object):
	def __init__(self,checkpoints,relu_targets,device='/gpu:0'):
		graph = tf.get_default_graph()
		with graph.device(device):
			self.model=UST(relu_targets=relu_targets)
			self.content_input = self.model.content_input
			self.decoded_output = self.model.decoded_output

			config = tf.ConfigProto(allow_soft_placement=True)
			config.gpu_options.allow_growth = True
			self.sess = tf.Session(config=config)
			self.sess.run(tf.global_variables_initializer())

			for relu_target, checkpoint_dir in zip(relu_targets, checkpoints):
				decoder_prefix = 'decoder_{}'.format(relu_target)
				relu_vars = [v for v in tf.trainable_variables() if decoder_prefix in v.name]

				saver = tf.train.Saver(var_list=relu_vars)
				
				ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
				if ckpt and ckpt.model_checkpoint_path:
					print('Restoring vars for {} from checkpoint {}'.format(relu_target, ckpt.model_checkpoint_path))
					saver.restore(self.sess, ckpt.model_checkpoint_path)
				else:
					raise Exception('No checkpoint found for target {} in dir {}'.format(relu_target, checkpoint_dir))
						
	def predict(self,content,style,alpha=1):
		content = self.preprocess(content)
		style   = self.preprocess(style)
		s = time.time()
		stylized = self.sess.run(self.decoded_output, feed_dict={
														  self.content_input: content,
														  self.model.style_input: style,
														  self.model.alpha: alpha})
		print("Stylized in:",time.time() - s)

		return np.uint8(np.clip(stylized[0],0,1)*255)


def main():
	start = time.time()
	wct_model=WCT(checkpoints=args.checkpoints,relu_targets=args.relu_targets)
	content_files = get_files(args.content_path)
	style_files = get_files(args.style_path)

	try:
		os.makedirs(args.out_path)
	except:
		print("out dir exists")
		pass

	for content_fullpath in content_files:
		content_prefix, content_ext = os.path.splitext(content_fullpath)
		content_prefix = os.path.basename(content_prefix)

		content_img=get_img(content_fullpath)

		for style_fullpath in style_files: 
			style_prefix, _ = os.path.splitext(style_fullpath)
			style_prefix = os.path.basename(style_prefix)
			style_img = get_img(style_fullpath)
			stylized_rgb = wct_model.predict(content_img, style_img, args.alpha)
			out_f = os.path.join(args.out_path, '{}_{}{}'.format(content_prefix, style_prefix, content_ext))
			cv2.imwrite(out_f,stylized_rgb)

			print("{}: Wrote stylized output image to {}".format(count, out_f))

	print("Finished stylizing {} outputs in {}s".format(count, time.time() - start))

		
if __name__ == '__main__':
	main()