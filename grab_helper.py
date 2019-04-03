import numpy as np
import cv2
import os


def algo_grabcut(filename1,filename2):
	command="python2 stylize.py --content-path "+filename1+" --style-path "+filename2+" --out-path output"+" --checkpoints models/relu5_1/ models/relu4_1/ models/relu3_1/ models/relu2_1/ models/relu1_1/ --relu-targets relu5_1 relu4_1 relu3_1 relu2_1 relu1_1"
	os.system(command)
	content_prefix, content_ext = os.path.splitext(filename1)
	content_prefix = os.path.basename(content_prefix)
	style_prefix, _ = os.path.splitext(filename2)
	style_prefix = os.path.basename(style_prefix)
	out_f = os.path.join("output", '{}_{}{}'.format(content_prefix, style_prefix, content_ext))
	return out_f,True
