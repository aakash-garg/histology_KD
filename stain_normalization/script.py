from __future__ import division

import stain_utils as utils
import stainNorm_Reinhard
# import stainNorm_Macenko
import glob
import cv2 as cv
import os
from natsort import natsorted
# import stainNorm_Vahadane

import numpy as np
import matplotlib.pyplot as plt

n = stainNorm_Reinhard.Normalizer()

LABELS = ['Benign', 'InSitu', 'Invasive', 'Normal']
input_path = '../dataset/'
out_path = '../dataset_norm/'
for i in LABELS:
  if not os.path.exists(out_path+i):
    os.makedirs(out_path+i)

output_path = []
path = [name for index in range(len(LABELS)) for name in natsorted(glob.glob(input_path + LABELS[index] + '/*.tif'))]
for i in path:
  output_path.append(out_path + i[int(len(input_path)):])
  
# path = glob.glob(input_path + "/*.tif")
n.fit(cv.imread(path[0]))

for idx, img in enumerate(path):
  print(idx, img)
  name = img.split('/')[-1]
  i = cv.imread(img)
  transformed = n.transform(i)
  cv.imwrite(output_path[idx], transformed)
