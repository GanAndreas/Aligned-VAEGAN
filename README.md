# Aligned-VAEGAN
A Cross-modal Embedding Approach by Utilizing VAEGANs on Generalized Zero-Shot Learning

saved parameters can be downloaded here 
https://drive.google.com/file/d/1ziUTlGeRAAfYELFDSZYrg2fD3DkbNXbz/view?usp=sharing

put the folder in the directory with the other files

## Dataset

Download the following folder https://www.dropbox.com/sh/btoc495ytfbnbat/AAAaurkoKnnk0uV-swgF-gdSa?dl=0 and put it in this repository. Next to the folder "model", there should be a folder "data".
Raw data for:
1. AWA2 - https://cvml.ist.ac.at/AwA2/
2. CUB - http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
3. SUN - http://cs.brown.edu/~gmpatter/sunattributes.html (check the bottom of the site page)

## Package Requirement:

- python==3.6
- torch==0.4.1
- numpy==1.14.3
- scipy==1.1.0
- scikit_learn==0.20.3
- networkx==1.11
- tensorboardX==1.9
- tensorflow==2.0.0

## Testing Command

To try the model with pretrained parameters, download the saved parameters above, and type this command in the terminal

python single_experiment.py --dataset AWA2 --num_gen_iter 5 --num_dis_iter 4 --mod_dataset 5Gen4Dis --device cuda --pretrain True

another arguments and their explanation can be found in "single_experiment.py"
