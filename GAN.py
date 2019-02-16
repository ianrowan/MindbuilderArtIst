import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt

base_path = os.path.dirname(os.path.realpath(__file__))
#Preprocessing
im_h = 256
im_w = 256
end_size = int(im_w/32)

#print("updated")
train = np.load("{}/imgs_256_cars.npy".format(base_path))
print("12")


def next_batch(num, data):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    #labels_shuffle = [labels[i] for i in idx]
    return np.asarray(data_shuffle)


def weight_var(shape, nam):
    init = tf.truncated_normal(shape, stddev=0.02, name=nam+ "weight")
    return tf.get_variable(nam+"weight",shape)


def bias_var(shape, nam):
    init = tf.constant(0.01, shape=shape, name=nam+ "bias")
    return tf.get_variable(nam + "bias", shape)


def conv2d (x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def Conv_Layer(input_, shape, name, str=1,bn=True, pool=False):
    with tf.variable_scope(name):
        w = weight_var(shape, name)
        b = bias_var([shape[3]], name)
        if not bn:
            return tf.nn.leaky_relu(conv2d(input_, w, stride=str) + b)

        h_conv = tf.nn.leaky_relu(tf.layers.batch_normalization(conv2d(input_, w, stride=str) + b, reuse=tf.AUTO_REUSE))

        if pool:
            return max_pool_2x2(h_conv)
    return h_conv


def deconv_layer(inp, shape, dim, stride, b, bn=True):
    w = weight_var(shape, "d_conv{}".format(str(dim) + str(stride)))
    bi = bias_var([shape[2]],"d_conv{}".format(str(dim) + str(stride)))
    if not bn:
        return tf.nn.conv2d_transpose(inp, filter=w, output_shape=[b, dim, dim, shape[2]], strides=[1, stride, stride, 1])+bi

    return tf.layers.batch_normalization(
        tf.nn.conv2d_transpose(inp, filter=w, output_shape=[b, dim, dim, shape[2]], strides=[1, stride, stride, 1])+bi)


def guassian_noise(input_layer):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=.1, dtype=tf.float32)
    return input_layer + noise

#########################
# Discriminator Network #
#########################
images = tf.placeholder(tf.float32,shape=[None, im_h, im_w, 3], name="images")
#images_gen = tf.placeholder(tf.float32,shape=[None, im_h, im_w, 3], name="images_gen")
#labs = tf.placeholder(tf.float32, shape=[None,1], name="labels")
def discrim(x, reuse=False):
    with tf.variable_scope("discriminator", reuse=reuse):
        #x1 = guassian_noise(x)
        con_0 = Conv_Layer(x, [5,5 ,3, 64], "conv_0", str=2, bn=False)
        conv_1 = Conv_Layer(con_0,[5, 5, 64, 128],"conv_1", str = 2)
        conv_2 = Conv_Layer(conv_1, [5, 5, 128, 256], "conv_2", str = 2)
        conv_3 = Conv_Layer(conv_2, [5,5, 256, 512], "conv_3", str=2)
        conv_4 = Conv_Layer(conv_3, [5,5,512, 1024], "conv_4", str=2)

        w_1 = weight_var([end_size*end_size*1024, 1], "flat_1", )
        #b_1 = bias_var([1], "flat_1", )
        flat = tf.reshape(conv_4, [-1, end_size*end_size*1024])
        return tf.matmul(flat, w_1)

#####################
# Generator Network #
#####################
z = tf.placeholder(tf.float32, [None, 100], name="z")
def gen(z, batch, reuse=False):

    with tf.variable_scope('generator', reuse=reuse):
        w_1 = weight_var([100, end_size*end_size*1024], "flat_2",)
        z1 = tf.matmul(z, w_1)
        z2 = tf.reshape(z1, [-1, end_size, end_size, 1024])
        d_conv1 = tf.nn.relu(deconv_layer(z2, [5,5, 512, 1024],end_size*2, stride=2, b=batch))
        d_conv2 = tf.nn.relu(deconv_layer(d_conv1, [5, 5, 256, 512], end_size*4, stride=2, b=batch))
        d_conv3 = tf.nn.relu(deconv_layer(d_conv2, [5, 5, 128, 256], end_size*8, stride=2, b=batch))
        d_conv4 = tf.nn.relu(deconv_layer(d_conv3, [5, 5, 64, 128], end_size*16, stride=2, b=batch))
        d_conv5 = deconv_layer(d_conv4, [5,5,3,64], end_size*32, stride=2, b=batch, bn=False)
        return tf.tanh(d_conv5, name="generate")


generate = gen(z, batch=32)
#print(generate)
d_logits = discrim(images)
gen_logits = discrim(generate, reuse=True)
#genn = gen(z, batch=32, reuse=True)
img_test = tf.summary.image("painted", generate, max_outputs=10)
dis_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.concat((tf.ones(shape=[16, 1])*.95,
                                                                                    tf.zeros(shape=[16, 1])), axis=0), logits=d_logits))
gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones(shape=[32, 1]), logits=gen_logits))
scale_d = tf.summary.scalar("dis_loss", dis_loss)
scale_g = tf.summary.scalar("g_loss", gen_loss)
t_vars = tf.trainable_variables()

d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
g_vars = [var for var in t_vars if var.name.startswith('generator')]

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_dis = tf.train.AdamOptimizer(.0001, beta1=.6).minimize(dis_loss, var_list=d_vars)
    train_gen = tf.train.AdamOptimizer(.0002, beta1=.6).minimize(gen_loss, var_list=g_vars)
saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    write = tf.summary.FileWriter('{}/GANs256_abs'.format(base_path), sess.graph)
    batch_size = 16
    try:
        for i in tqdm(range(90000)):
            vector1 = np.random.uniform(-1, 1, size=[32, 100])
            dis_in = next_batch(16, train)
            dis_fake = generate.eval(feed_dict={z: vector1})
            dis_in = np.concatenate((dis_in, dis_fake[:16]), axis=0)
            if i%30 == 0:
                d_l = dis_loss.eval(feed_dict={images: dis_in})
                g_l = gen_loss.eval(feed_dict={z: vector1})
                tqdm.write("==============Step: {}==============".format(str(i)))
                tqdm.write("Discriminator Loss: {}". format(str(d_l)))
                tqdm.write("Generator Loss: {}".format(str(g_l)))
                sd, sg = sess.run([scale_d, scale_g], feed_dict={dis_loss: d_l, gen_loss: g_l})
                t_images = sess.run(img_test, feed_dict={z: vector1})
                write.add_summary(t_images, i)
                write.add_summary(sd, i)
                write.add_summary(sg, i)
            #Train Discriminator
            train_dis.run(feed_dict={images: dis_in})
            #Train Generator
            train_gen.run(feed_dict={z: vector1})
            if i % 5000 == 0:
                saver.save(sess, "{}/GAN_HD.ckpt".format(base_path))
                tqdm.write("Checkpoint Saved")
    except KeyboardInterrupt:
        pass

    saver.save(sess, "{}/GAN_HD.ckpt".format(base_path))
