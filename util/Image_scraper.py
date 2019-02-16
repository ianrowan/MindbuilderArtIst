from bs4 import BeautifulSoup
from urllib.request import urlopen, Request
import json
from numpy import genfromtxt
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.io import imread, imshow
from skimage.transform import resize
from time import time



def get_soup( url, header):
    return BeautifulSoup(urlopen(Request(url, headers=header)), 'html.parser')


def img_scrape(name, num, save):
    '''
    Scrapes google images for keyword for 'num' images. Outputs full set to folder
    :param name:
    :param num:
    :param save:
    :return:
    '''
    query = name  # raw_input(args.search)
    max_images = int(num)
    save_directory = save
    image_type = "Action"
    query = query.split()
    query = '+'.join(query)
    url = "https://www.google.co.in/search?q=" + query + "&source=lnms&tbm=isch"
    header = {
        'User-Agent': "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}
    soup = get_soup(url, header)
    ActualImages = []  # contains the link for Large original images, type of  image
    for a in soup.find_all("div", {"class": "rg_meta"}):
        link, Type = json.loads(a.text)["ou"], json.loads(a.text)["ity"]
        ActualImages.append((link, Type))
    for i, (img, Type) in enumerate(ActualImages[0:max_images]):
        try:
            req = Request(img, headers={
                'User-Agent': "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"})
            raw_ = urlopen(req)
            raw_img = raw_.read()
            if len(Type) == 0:
                f = open(os.path.join(save_directory, name.replace(" ", "") + "_" + str(i) + ".jpg"), 'wb')
            else:
                f = open(os.path.join(save_directory, name.replace(" ", "") + "_" + str(i) + "." + Type), 'wb')
            f.write(raw_img)
            f.close()
        except Exception as e:
            print("could not load : " + img)
            print(e)
            continue

def all_images(cat_path, num_imgs, save_path):
    '''
    Takes input of csv file, turns it into array of style len(keyword, label).
    Then iterates over each keyword to scrape images and out put to file named after keyword.
    :param cat_path:
    :param num_imgs:
    :param save_path:
    :return:
    '''
    kwds = cat_path
    num = num_imgs
    queries = genfromtxt(kwds, delimiter=',', dtype=str)

    for i in tqdm(queries):
        save_path1 = save_path
        if not os.path.exists(save_path1):
            os.makedirs(save_path1)
        img_scrape(i + " Car Design", num, save_path1)
all_images("/home/ian/car_design.csv", 100, "/home/ian/GAN_CARs")