# ArtistAI
Welcome to AI artist. This repo includes the code necessary to turn your computer into the next Picasso.

The GAN.py script is a custom implmenetation of a Genreative Adversarial Network based off of the DCGAN architechture. The network has been tested on multiple art forms and has proven to work sucessfully. A good training generally requires about 15-20k training examples in the category of your choice. Networks are trained on a NVIDIA GTX 970 and typically train for 24 hours.

To run on your system:
1. Change input directory
2. Change tensorboard directory
3. Change save directory
4. pip3 install (any needed requirements)

Util folder:
img_generator.py: Given a trained model directory, returns art from running generator op
resize.py: utility file to resize a set of images and save numpy array for faster load times
