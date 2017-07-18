# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 21:02:25 2017

@author: dwipr
"""
# Import required library
from PIL import Image
import os

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Final intended spatial size
RESIZE_TO = 400


def resize_dataset(set, type):
  ## PARAM
  # set => 'train', 'valid', or 'test' 
  # type => 'sat' (satelite images) or 'map' (ground truth) 
  input_path = '../../~vmnih/data/'+set+'/'+type
  output_path = '../../~vmnih/data_resized/'+set+'/'+type
  
  input_files = os.listdir(input_path)
  i=0
  for file in input_files:
    current_image = Image.open(os.path.join(input_path, file), 'r')
  
    resized_img = current_image.resize((RESIZE_TO,RESIZE_TO), Image.ANTIALIAS)
    resized_img.save(os.path.join(output_path, file.split('.')[0])+'.png' , 'PNG')
    
    i=i+1
    print(str(i)+' of '+str(len(input_files)), end='\r')
  
if __name__ == "__main__":
    resize_dataset('valid', 'sat')
    