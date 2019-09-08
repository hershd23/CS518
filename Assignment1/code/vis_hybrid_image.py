import matplotlib.pyplot as plt
import os
import cv2
from os.path import join
import numpy as np
from PIL import Image
import matplotlib.image as mpimg
from skimage.transform import rescale, resize, downscale_local_mean

def  vis_hybrid_image(hybrid_image):

  """
  %visualize a hybrid image by progressively downsampling the image and
  %concatenating all of the images together.
  """
  scales = 5#how many downsampled versions to create
  padding = 5 #how many pixels to pad.

  original_height = hybrid_image.shape[0]
  num_colors = hybrid_image.shape[2] #counting how many color channels the input has
  output = hybrid_image
  cur_image = hybrid_image

  for i in range(2,scales+1):
      # add padding
      output = cat(2, output, np.ones(original_height, padding, num_colors)) #implement the correct command here
      # dowsample image
      cur_image = resize(cur_image,(cur_image.shape[0] // 2, cur_image.shape[1] // 2), \
                    anti_aliasing=True)
      # pad the top and append to the output
      tmp = cat(1,np.ones(original_height - cur_image.shape[0], cur_image.shape[1], num_colors), cur_image);
      output = cat(2, output, tmp);    
  
  
  return(output)

""" adopted from code by James Hays (GATech)"""