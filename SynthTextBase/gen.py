# Author: Ankush Gupta
# Date: 2015

"""
Entry-point for generating synthetic text images, as described in:

@InProceedings{Gupta16,
      author       = "Gupta, A. and Vedaldi, A. and Zisserman, A.",
      title        = "Synthetic Data for Text Localisation in Natural Images",
      booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition",
      year         = "2016",
    }
"""

import numpy as np
import h5py
import os, sys, traceback
import os.path as osp
from synthgen import *
from common import *
import wget, tarfile
from PIL import Image
from matplotlib import pyplot as plt
import time
import pickle
import random

## Define some configuration variables:
NUM_IMG = -1  # no. of images to use for generation (-1 to use all available):
INSTANCE_PER_IMAGE = 200  # no. of times to use the same image
SECS_PER_IMG = 1  # max time per image in seconds

# path to the data-file, containing image, depth and segmentation:
DATA_PATH = 'data'
DB_FNAME = "data/dset.h5"
# url of the data (google-drive public file):
DATA_URL = 'http://www.robots.ox.ac.uk/~ankush/data.tar.gz'
RENDER_PERCENTAGE=0.09375

os.makedirs("results", exist_ok=True)
os.makedirs("strdata/images", exist_ok=True)
os.makedirs("generated_data/pickles",exist_ok=True)


count = 0
def get_data():
    """
  Download the image,depth and segmentation data:
  Returns, the h5 database.
  """
    if not osp.exists(DB_FNAME):
        try:
            colorprint(Color.BLUE, '\tdownloading data (56 M) from: ' + DATA_URL, bold=True)
            print()
            sys.stdout.flush()
            out_fname = 'data.tar.gz'
            wget.download(DATA_URL, out=out_fname)
            tar = tarfile.open(out_fname)
            tar.extractall()
            tar.close()
            os.remove(out_fname)
            colorprint(Color.BLUE, '\n\tdata saved at:' + DB_FNAME, bold=True)
            sys.stdout.flush()
        except:
            print(colorize(Color.RED, 'Data not found and have problems downloading.', bold=True))
            sys.stdout.flush()
            sys.exit(-1)
    # open the h5 file and return:
    return h5py.File(DB_FNAME, 'r')



def save_str(image_name, image_path, labels, boxes, delete):
    global count
    print("Total words generated so far: " + str(count) + " WORDS: ", str(labels))
    img = Image.open(image_path)
    with open("strdata/gts.txt", "a") as f:
        for idx, label in enumerate(labels):
            try:
                box = boxes[:, :, idx]
                left_top_x, left_top_y = int(box[0][0]), int(box[1][0])
                right_top_x, right_top_y = int(box[0][1]), int(box[1][1])
                left_bottom_x, left_bottom_y = int(box[0][2]), int(box[1][2])
                right_bottom_x, right_bottom_y = int(box[0][3]), int(box[1][3])
                box = [left_top_x, left_top_y, right_top_x, right_top_y, right_bottom_x, right_bottom_y, left_bottom_x,
                       left_bottom_y]
                bbox = (box[0], min(box[1], box[3]), box[6], max(box[5], box[7]))

                img_name = str(count) + ".jpg"
                img_save_path = "strdata/images/" + img_name
                img.crop(box=bbox).save(img_save_path)
                gt = img_name + "\t" + labels[idx]
                print(gt, file=f)
                count += 1
            except Exception as e:
                print(e)
    if delete:
        os.remove(image_path)


def add_res_to_db(imgname, res):
    """
  Add the synthetically generated text image instance
  and other metadata to the dataset.
  """
    ninstance = len(res)
    for i in range(ninstance):
        dname = "%s_%d" % (imgname, i)
        image_name = dname + '.jpg'
        image_path = 'generated_data/' + image_name
        plt.imsave(image_path, res[i]['img'])

        delete = 1
        if random.random() <= RENDER_PERCENTAGE:
            delete = 0
            with open("results/pickles/" + dname + ".pkl", "wb") as f:
                pickle.dump(res[i], f)

        labels = list(res[i]['txt'])
        boxes = res[i]['wordBB']
        save_str(image_name, image_path, labels, boxes, delete)


def main(viz=False):
    # open databases:
    print(colorize(Color.BLUE, 'getting data..', bold=True))
    db = get_data()
    print(colorize(Color.BLUE, '\t-> done', bold=True))

    # open the output h5 file:
    # out_db = h5py.File(OUT_FILE,'w')
    # out_db.create_group('/data')
    # print(colorize(Color.GREEN,'Storing the output in: '+OUT_FILE, bold=True))

    # get the names of the image files in the dataset:
    imnames = sorted(db['image'].keys())
    N = len(imnames)
    global NUM_IMG
    if NUM_IMG < 0:
        NUM_IMG = N
    start_idx, end_idx = 5125, min(NUM_IMG, N)

    RV3 = RendererV3(DATA_PATH, max_time=SECS_PER_IMG)
    for i in range(start_idx, end_idx):
        imname = imnames[i]
        try:
            # get the image:
            img = Image.fromarray(db['image'][imname][:])
            # get the pre-computed depth:
            #  there are 2 estimates of depth (represented as 2 "channels")
            #  here we are using the second one (in some cases it might be
            #  useful to use the other one):
            depth = db['depth'][imname][:].T
            depth = depth[:, :, 1]
            # get segmentation:
            seg = db['seg'][imname][:].astype('float32')
            area = db['seg'][imname].attrs['area']
            label = db['seg'][imname].attrs['label']

            # re-size uniformly:
            sz = depth.shape[:2][::-1]
            img = np.array(img.resize(sz, Image.ANTIALIAS))
            seg = np.array(Image.fromarray(seg).resize(sz, Image.NEAREST))

            print(colorize(Color.RED, '%d of %d' % (i, end_idx - 1), bold=True))
            t = time.time()
            res = RV3.render_text(img, depth, seg, area, label,
                                  ninstance=INSTANCE_PER_IMAGE, cnt=i, viz=viz)
            if len(res) > 0:
                # non-empty : successful in placing text:
                add_res_to_db(imname, res)
            print("Time taken: ", time.time() - t)
            # visualize the output:
            if viz:
                if 'q' in input(colorize(Color.RED, 'continue? (enter to continue, q to exit): ', True)):
                    break
        except:
            traceback.print_exc()
            print(colorize(Color.GREEN, '>>>> CONTINUING....', bold=True))
            continue
    db.close()
    # out_db.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Genereate Synthetic Scene-Text Images')
    parser.add_argument('--viz', action='store_true', dest='viz', default=False,
                        help='flag for turning on visualizations')
    args = parser.parse_args()
    main(args.viz)
