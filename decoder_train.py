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


def clip(x):
	return tf.clip_by_value(x,0,1)

def pad_reflect(x):
	return tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]],mode='REFLECT')

def Conv2DReflect(lambda_name, *args, **kwargs):
	return Lambda(lambda x: Conv2D(*args, **kwargs)(pad_reflect(x)), name=lambda_name)



parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str,
					dest='checkpoint', help='Checkpoint save dir', 
					required=True)
parser.add_argument('--log-path', type=str,
					dest='log_path', help='Logging dir path')
parser.add_argument('--relu-target', type=str, required=True,
					help='Target VGG19 relu layer to decode from, e.g. relu4_1')
parser.add_argument('--content-path', type=str, required=True,
					dest='content_path', help='Content images folder')
parser.add_argument('--val-path', type=str, default=None,
					dest='val_path', help='Validation images folder')



parser.add_argument('--learning-rate', type=float,
					dest='learning_rate', help='Learning rate',
					default=1e-4)
parser.add_argument('--lr-decay', type=float,
					dest='lr_decay', help='Learning rate decay',
					default=0)
parser.add_argument('--max-iter', type=int,
					dest='max_iter', help='Max # of training iterations',
					default=16000)

parser.add_argument('--save-iter', type=int,
					dest='save_iter', help='Checkpoint save frequency',
					default=200)
parser.add_argument('--summary-iter', type=int,
					dest='summary_iter', help='Summary write frequency',
					default=20)
parser.add_argument('--max-to-keep', type=int,
					dest='max_to_keep', help='Max # of checkpoints to keep around',
					default=10)

args = parser.parse_args()



def torch_decay(learning_rate, global_step, decay_rate, name=None):

	if global_step is None:
		raise ValueError("global_step is required for exponential_decay.")
	with tf.name_scope(name, "ExponentialDecay", [learning_rate, global_step, decay_rate]) as name:
		learning_rate = tf.convert_to_tensor(learning_rate, name="learning_rate")
		dtype = learning_rate.dtype
		return learning_rate / (1 + tf.cast(global_step, dtype)*tf.cast(decay_rate, dtype))



EncoderDecoder = namedtuple('EncoderDecoder', 
							'content_input content_encoder_model content_encoded \
							 decoder_input, decoder_model decoded decoded_encoded \
							 pixel_loss feature_loss total_loss \
							 train_op learning_rate global_step \
							 summary_op')



class UST(object):

	def __init__(self, relu_targets=['relu5_1','relu4_1','relu3_1','relu2_1','relu1_1'], *args, **kwargs):

		self.style_input=tf.placeholder_with_default(tf.constant([[[[0.,0.,0.]]]]), shape=(None, None, None, 3))
		self.alpha=tf.placeholder_with_default(1.,shape=[])
		self.encoder_decoders=[]

		with tf.name_scope("vgg_encoder"):
			self.vgg_model=vgg("models/vgg_normalised.t7",sorted(relu_targets)[-1])

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

		decoder_input=content_encoded

		with tf.name_scope("decoder_"+relu_target):
			n_channels = content_encoded.get_shape()[-1].value
			decoder_model = self.build_decoder(input_shape=(None, None, n_channels), relu_target=relu_target)
			decoder_input_wrapped = tf.placeholder_with_default(decoder_input, shape=[None,None,None,n_channels])
			decoded = decoder_model(Lambda(lambda x: x)(decoder_input_wrapped))

		decoded_encoded = content_encoder_model(decoded)

		
		with tf.name_scope("losses_"+relu_target):
			feature_loss = tf.losses.mean_squared_error(decoded_encoded, content_encoded)
			pixel_loss = tf.losses.mean_squared_error(decoded,content_imgs)
			total_loss = feature_loss + pixel_loss

		with tf.name_scope("train_"+relu_target):
			global_step = tf.Variable(0, name='global_step_train', trainable=False)
			learning_rate = torch_decay(learning_rate, global_step, lr_decay)
			d_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999)
			d_vars = [var for var in tf.trainable_variables() if 'decoder_'+relu_target in var.name]
			train_op = d_optimizer.minimize(total_loss, var_list=d_vars, global_step=global_step)

		with tf.name_scope('summary_'+relu_target):
			feature_loss_summary = tf.summary.scalar('feature_loss', feature_loss)
			pixel_loss_summary = tf.summary.scalar('pixel_loss', pixel_loss)
			total_loss_summary = tf.summary.scalar('total_loss', total_loss)
			content_imgs_summary = tf.summary.image('content_imgs', content_imgs)
			decoded_images_summary = tf.summary.image('decoded_images', clip(decoded))
			for var in d_vars:
				tf.summary.histogram(var.op.name, var)

			summary_op = tf.summary.merge_all()


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


def get_img_random_crop(src, resize=512, crop=256):
	img = cv2.imread(src)
	img = cv2.resize(img,(crop,crop))
	return img


def get_files(img_dir):
	files = os.listdir(img_dir)
	paths = []
	for x in files:
		paths.append(os.path.join(img_dir, x))
	return paths

def batch_gen(folder, batch_shape):
	files = np.asarray(get_files(folder))
	while True:
		X_batch = np.zeros(batch_shape, dtype=np.float32)
		idx = 0
		while idx < batch_shape[0]:
			try:
				f = np.random.choice(files)

				X_batch[idx] = get_img_random_crop(f, resize=512, crop=256)
				X_batch[idx] /= 255.

			except Exception as e:
				print(e,"pora")
				continue
			idx += 1
		yield X_batch



def train():
	batch_shape = (8,256,256,3)

	with tf.Graph().as_default():
		tf.logging.set_verbosity(tf.logging.INFO)

		queue_input_content = tf.placeholder(tf.float32, shape=batch_shape)
		queue_input_val = tf.placeholder(tf.float32, shape=batch_shape)
		queue = tf.FIFOQueue(capacity=100, dtypes=[tf.float32, tf.float32], shapes=[[256,256,3], [256,256,3]])
		enqueue_op = queue.enqueue_many([queue_input_content, queue_input_val])
		dequeue_op = queue.dequeue()
		content_batch_op, val_batch_op = tf.train.batch(dequeue_op, batch_size=8, capacity=100)

		def enqueue(sess):
			content_images = batch_gen(args.content_path, batch_shape)
			
			val_path = args.val_path if args.val_path is not None else args.content_path
			val_images = batch_gen(val_path, batch_shape)

			while True:
				content_batch = next(content_images)
				val_batch     = next(val_images)

				sess.run(enqueue_op, feed_dict={queue_input_content: content_batch,
												queue_input_val:     val_batch})

		model = UST(relu_targets=[args.relu_target],
						 batch_size=8).encoder_decoders[0]

		saver = tf.train.Saver(max_to_keep=5)

		config = tf.ConfigProto()
		config.gpu_options.allow_growth=True
		with tf.Session(config=config) as sess:
			enqueue_thread = threading.Thread(target=enqueue, args=[sess])
			enqueue_thread.isDaemon()
			enqueue_thread.start()
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord, sess=sess)

			log_path = args.log_path if args.log_path is not None else os.path.join(args.checkpoint,'log')
			summary_writer = tf.summary.FileWriter(log_path, sess.graph)

			sess.run(tf.global_variables_initializer())
 
			def load_latest():
				if os.path.exists(os.path.join(args.checkpoint,'checkpoint')):
					print("Restoring checkpoint")
					saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint))
			load_latest()

			for iteration in range(args.max_iter):
				start = time.time()
				
				content_batch = sess.run(content_batch_op)

				fetches = {
					'train':        model.train_op,
					'global_step':  model.global_step,
					'lr':           model.learning_rate,
					'feature_loss': model.feature_loss,
					'pixel_loss':   model.pixel_loss,
				}

				feed_dict = { model.content_input: content_batch }

				try:
					results = sess.run(fetches, feed_dict=feed_dict)
				except Exception as e:
					print(e)
					print("Exception encountered, re-loading latest checkpoint")
					load_latest()
					continue

				if iteration % args.summary_iter == 0:
					val_batch = sess.run(val_batch_op)
					summary = sess.run(model.summary_op, feed_dict={ model.content_input: val_batch })
					summary_writer.add_summary(summary, results['global_step'])

				if iteration % args.save_iter == 0:
					save_path = saver.save(sess, os.path.join(args.checkpoint, 'model.ckpt'), results['global_step'])
					print("Model saved in file: %s" % save_path)

				print("Step: {}  LR: {:.7f}  Feature: {:.5f}  Pixel: {:.5f} Time: {:.5f}".format(results['global_step'], 
																											  results['lr'], 
																											  results['feature_loss'], 
																											  results['pixel_loss'], 
																											  time.time() - start))

			save_path = saver.save(sess, os.path.join(args.checkpoint, 'model.ckpt'), results['global_step'])
			print("Model saved in file: %s" % save_path)


if __name__ == '__main__':
	train()
