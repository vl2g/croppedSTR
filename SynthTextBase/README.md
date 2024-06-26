# SynthTextHindi
Code for generating synthetic text images for Hindi Language <!-- as described in ["Synthetic Data for Text Localisation in Natural Images", Ankush Gupta, Andrea Vedaldi, Andrew Zisserman, CVPR 2016](http://www.robots.ox.ac.uk/~vgg/data/scenetext/).
 -->

This code is the major modification of https://github.com/ankush-me/SynthText for Hindi Language.

**Prerequisites (better if you create a conda environment and check install.sh first ):**

- python3.5
- git clone https://github.com/ldo/harfpy.git
- cd harfpy 
- python3 setup.py install
- cd ..
- git clone https://github.com/ldo/qahirah.git
- git clone https://github.com/ldo/python_freetype.git
- opencv (3.x)  conda install -c menpo opencv3
- pygame conda install -c cogsci pygame 
- PIL (Image)
- numpy
- matplotlib
- h5py
- scipy


**To generate image**
```
	python gen.py
```
This will store images with text in folder 'do' folder in '.jpg' format and all other info about image(character and word bounding box info, text) will be stored in an h5 file in `results/SynthText.h5`).
The generated images will be in accordance to 'dset.h5' (these are background images with their segmentation and depth masks info)


**Synthetic Scene-Text Hindi Image Samples**
![Synthetic Scene-Text-Hindi Samples](hindi.jpg "Samples")

**Text below is same  as Ankush Gupta's Readme file**

<!-- **Synthetic Scene-Text Image Samples**
![Synthetic Scene-Text Samples](samples.png "Synthetic Samples")
 -->
The library is written in Python. The main dependencies are:

```
pygame, opencv (cv2), PIL (Image), numpy, matplotlib, h5py, scipy
```

### Generating samples

```
python gen.py --viz
```

This will download a data file (~56M) to the `data` directory. This data file includes:

  - **dset.h5**: This is a sample h5 file which contains a set of 5 images along with their depth and segmentation information. Note, this is just given as an example; you are encouraged to add more images (along with their depth and segmentation information) to this database for your own use.
  - **data/fonts**: three sample fonts (add more fonts to this folder and then update `fonts/fontlist.txt` with their paths).
  - **data/newsgroup**: Text-source (from the News Group dataset). This can be subsituted with any text file. Look inside `text_utils.py` to see how the text inside this file is used by the renderer.
  - **data/models/colors_new.cp**: Color-model (foreground/background text color model), learnt from the IIIT-5K word dataset.
  - **data/models**: Other cPickle files (**char\_freq.cp**: frequency of each character in the text dataset; **font\_px2pt.cp**: conversion from pt to px for various fonts: If you add a new font, make sure that the corresponding model is present in this file, if not you can add it by adapting `invert_font_size.py`).

This script will generate random scene-text image samples and store them in an h5 file in `results/SynthText.h5`. If the `--viz` option is specified, the generated output will be visualized as the script is being run; omit the `--viz` option to turn-off the visualizations. If you want to visualize the results stored in  `results/SynthText.h5` later, run:

```
python visualize_results.py
```
### Pre-generated Dataset
A dataset with approximately 800000 synthetic scene-text images generated with this code can be found [here](http://www.robots.ox.ac.uk/~vgg/data/scenetext/).

### Adding New Images
Segmentation and depth-maps are required to use new images as background. Sample scripts for obtaining these are available [here](https://github.com/ankush-me/SynthText/tree/master/prep_scripts).

* `predict_depth.m` MATLAB script to regress a depth mask for a given RGB image; uses the network of [Liu etal.](https://bitbucket.org/fayao/dcnf-fcsp/) However, more recent works (e.g., [this](https://github.com/iro-cp/FCRN-DepthPrediction)) might give better results.
* `run_ucm.m` and `floodFill.py` for getting segmentation masks using [gPb-UCM](https://github.com/jponttuset/mcg).

For an explanation of the fields in `dset.h5` (e.g.: `seg`,`area`,`label`), please check this [comment](https://github.com/ankush-me/SynthText/issues/5#issuecomment-274490044).

### Pre-processed Background Images
The 8,000 background images used in the paper(["Synthetic Data for Text Localisation in Natural Images", Ankush Gupta, Andrea Vedaldi, Andrew Zisserman, CVPR 2016](http://www.robots.ox.ac.uk/~vgg/data/scenetext/)), along with their segmentation and depth masks, have been uploaded here:
`http://www.robots.ox.ac.uk/~vgg/data/scenetext/preproc/<filename>`, where, `<filename>` can be:

- `imnames.cp` [180K]: names of filtered files, i.e., those files which do not contain text
- `bg_img.tar.gz` [8.9G]: compressed image files (more than 8000, so only use the filtered ones in imnames.cp)
- `depth.h5` [15G]: depth maps
- `seg.h5` [6.9G]: segmentation maps

### Files Created during synthetic image generation
- `strdata/pickles`: This folder contains a pickle file for all synthetic images. Stored pickle file will contain all numpy image, bounding boxes and words data.
- `generated_data`: This folder contains the rendered scene image for visual results.
- `strdata/gts.txt`: This file will contain the word image data in `image_name\tword_text` format.
- `strdata/images`: This folder will contain all cropped word images.


### Add a new language
- Run `SynthTextBase/prep_scripts/update_freq.py` by providing vocab file path for the new language in the file.
- Add fonts in `SynthTextBase/data/font` folder then
- Run `SynthTextBase/invert_font_size.py`

### Generate LMDB files
Run below command to after setting appropriate paths of word images files to generate lmdb file for training the recognition mode:
```
python prepare_lmdb_files.py
```

### Recogniser Training
```https://docs.google.com/document/d/1W_4Bp6cLdZxBU4opcMN7xLXw3I07aNtSonycgpxTnCU/edit```