import numpy as np
import tensorflow as tf
from scipy.misc import imsave
import scipy
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

def generate_images(file):
    meta = file + ".meta"
    ckpt = file
    with tf.Session() as sess:
        z = np.random.uniform(-1, 1, [32,100])
        graph = tf.train.import_meta_graph(meta)
        graph.restore(sess, ckpt)
        images = sess.run("generator/generate:0", feed_dict={"z:0": z})
    return images + 1 /2

def save_images(save_dir, net_file, num):
    for i in tqdm(range(num)):
        images = generate_images(net_file)
        for j, img in enumerate(images):
            imsave(save_dir+"/painting_{}{}.jpeg".format(str(i), str(j)),img)

save = "/Users/IanRowan/Pictures/paintings"
ckpt = "/Users/IanRowan/Dropbox/GANs_good/50k/GAN_HD.ckpt"
save_images(save, ckpt,10)

def save_images_from_event(fn, tag, output_dir='./'):
    """
    retreives specific tensorboard image
    :param fn:
    :param tag:
    :param output_dir:
    :return:
    """
    assert(os.path.isdir(output_dir))

    image_str = tf.placeholder(tf.string)
    im_tf = tf.image.decode_image(image_str)

    sess = tf.InteractiveSession()
    with sess.as_default():
        count = 0
        i = 0
        for e in tf.train.summary_iterator(fn):
            if e.step == 29010:
                for v in e.summary.value:
                    if v.tag == tag:
                        im = im_tf.eval({image_str: v.image.encoded_image_string})
                        output_fn = os.path.realpath('{}/image_{:05d}.png'.format(output_dir, count))
                        print("Saving '{}'".format(output_fn))
                        scipy.misc.imsave(output_fn, im)
                        count += 1
            i += 1

tag = "painted/image/1"
outputd = "/Users/IanRowan/Pictures/paintings/find"
dir = "/Users/IanRowan/events.out.tfevents.1527988957.ian-ThinkStation-S20"
#save_images_from_event(dir,tag, outputd)0