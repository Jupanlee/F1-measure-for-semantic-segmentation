# -*- coding: utf-8 -*-
import numpy as np
import glob
import argparse
import os
import re
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='F1 measure for salient object extraction')
parser.add_argument('--predict_path', type = str, default = None)
parser.add_argument('--predict_prefix', type = str, default = None)
parser.add_argument('--predict_format', type = str, default = None)
parser.add_argument('--target_path', type = str, default = None)
parser.add_argument('--target_prefix', type = str, default = None)
parser.add_argument('--target_format', type = str, default = None)
parser.add_argument('--img_path', type = str, default = None)
parser.add_argument('--img_prefix', type = str, default = None)
parser.add_argument('--img_format', type = str, default = None)
parser.add_argument('--out_path', type = str, default = None)
parser.add_argument('--out_prefix', type = str, default = None)
parser.add_argument('--out_format', type = str, default = None)
parser.add_argument('--metric_file', type = str, default = None)

args = parser.parse_args()

predict_path = args.predict_path
predict_prefix = args.predict_prefix
predict_format = args.predict_format
target_path = args.target_path
target_prefix = args.target_prefix
target_format = args.target_format
img_path = args.img_path
img_prefix = args.img_prefix
img_format = args.img_format
out_path = args.out_path
out_prefix = args.out_prefix
out_format = args.out_format
metric_file = args.metric_file

predict_pattern = predict_path + '/*' + predict_format
predict_paths = glob.glob(predict_pattern)

target_pattern = target_path + '/*' + target_format
target_paths = glob.glob(target_pattern)

predict_file_num = len(predict_paths)
target_file_num = len(target_paths)

assert predict_file_num == target_file_num

file_num = predict_file_num
print(file_num)

Precision_s = np.zeros(file_num) 
Recall_s = np.zeros(file_num) 
F1_measure_s = np.zeros(file_num) 

# 计数变量
i = 0

for file_path in predict_paths:
	file_name_with_ext = os.path.basename(file_path)
	file_name = os.path.splitext(file_name_with_ext)[0]
	file_name = re.sub("\D", "", file_name)

	predict_mask = np.array((Image.open(file_path)).convert('L'))
	height, width = predict_mask.shape

	target_mask = Image.open(os.path.join(target_path, target_prefix + file_name + target_format))
	target_mask = np.array(target_mask.convert('L'))

	height1, width1 = target_mask.shape
	assert (height == height1 and width == width1)

	img = Image.open(os.path.join(img_path, img_prefix + file_name + img_format))

	img_out = np.array(img)
	img_out = img_out[:,:,:3]
	img_out[predict_mask < 0.5, :] = (0,0,0)

	img_TP = predict_mask.copy()
	for y in range(1, height):
		for x in range(1, width):
			if (target_mask[y, x] > 0.5) & (predict_mask[y,x] > 0.5):
				img_TP[y, x] = 1
			else:
				img_TP[y, x] = 0

	num_img_TP = img_TP.sum()
	

	img_FP = np.zeros((height, width), dtype = np.int)
	for y in range(1, height):
		for x in range(1, width):
			if (target_mask[y, x] < 0.5) & (predict_mask[y, x] > 0.5):
				img_FP[y, x] = 1
				img_out[y, x, :] = (255,0,0)
			else:
				img_FP[y, x] = 0
	num_img_FP = img_FP.sum()

	img_FN = np.zeros((height, width), dtype = np.int)
	for y in range(1, height):
		for x in range(1, width):
			if (target_mask[y, x] > 0.5) & (predict_mask[y, x] < 0.5):
				img_FN[y, x] = 1
				img_out[y, x, :] = (0,0,255)
			else:
				img_FN[y, x] = 0

	num_img_FN = img_FN.sum()

	plt.imsave(os.path.join(out_path, out_prefix + file_name + out_format), img_out)

	Precision = num_img_TP / ( num_img_TP + num_img_FP )
	Recall = num_img_TP / ( num_img_TP + num_img_FN )
	F1_measure = 2 * Precision * Recall /(Precision+Recall)
	Precision_s[i] = Precision
	Recall_s[i] = Recall
	F1_measure_s[i] = F1_measure

	i = i + 1

try:
	metric_file_ = open(metric_file, 'w')
	metric_file_.write("min F1_measure: " + str(F1_measure_s.min()) + "\n")
	metric_file_.write("max F1_measure: " + str(F1_measure_s.max()) + "\n")
	metric_file_.write("mean F1_measure: " + str(F1_measure_s.mean()) + "\n")
	metric_file_.write(" ".join(str(v) for v in F1_measure_s))
finally:
	if metric_file_:
		metric_file_.close()
