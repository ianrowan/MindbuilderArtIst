import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from skimage.io import imread
from skimage.transform import resize
import scipy

"""
TODO: resize 128 abstract again
"""

def center_crop(x, resize_w):
    min_dim = np.min(x.shape[:2])
    re_y = x.shape[0] - min_dim
    re_x = x.shape[1] - min_dim
    j = int(re_y/2)
    i = int(re_x/2)
    return scipy.misc.imresize(x[j:j+min_dim, i:i+min_dim],
                               [resize_w, resize_w])

def transform(image, is_crop=True, resize_w=64):
    if is_crop:
        cropped_image = center_crop(image,resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.



im_h = 256
im_w = 256
path = "/home/ian/GAN_CARs"
ids_ = next(os.walk(path))[2]
print(len(ids_))
print(ids_[0:10])

train = np.zeros(shape=[len(ids_), im_h, im_w, 3], dtype=np.float32)
for i, id_ in tqdm(enumerate(ids_),total=len(ids_)):
    try:
     train[i] = resize(imread(path +"/"+id_)[:, :, :3], output_shape=[im_h, im_w, 3], mode="constant") * 2 - 1
    except IndexError:
        train[i] = train[i-1]
        tqdm.write(str(i))
    except ValueError:
        train[i] = train[i - 1]
        tqdm.write(str(i))
    except OSError:
        train[i] = train[i - 1]
        tqdm.write(str(i))

np.save("/home/ian/imgs_256_cars", train)
