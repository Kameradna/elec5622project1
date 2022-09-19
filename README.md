# ELEC5622 Signals and Health Project 1: Classifying AD patients and normal controls from brain images
ELEC5622 Project 1 90-dim SVM classification of Alzeimers brain AAL ROI volumes using bet, fsl, sklearn

Documentation for download order, setting up env

I have hosted some files on google drive to facilitate fewer canvas logins. 

https://drive.google.com/drive/folders/1fUGN2zkqs1PSAZGUVG5OLzHMMmE9350N?usp=sharing
```shell
git clone https://github.com/Kameradna/elec5622project1.git
cd elec5622project1
gdown https://drive.google.com/u/0/uc?id=1U8Bk_kZpOUZnXD34daumRdCuaQgxMhEo&export=download&confirm=t&uuid=4389a206-532a-4ff6-8d38-af73ddc5b284
gdown https://drive.google.com/uc?id=1-fnKe4yjIWFnodyPP8RiyeHBKxSfxXOv
unzip '*.zip'
mkdir -p Data/train Data/test Output/train Output/test
conda create -n elec5622project1 -f requirements.yml
conda activate elec5622project1
python data_split.py

```
And so we arrive at the beginning of the project
```shell
python svm.py --dummy --dummy_num 10000 --dummy_dim 40 --pca_components 2
```
```shell
We have 9000 training vectors with dim 40
Fitting SVM model...
The training accuracy of the trained SVM is 52.24%
The testing accuracy of the trained SVM is 51.50%
```
Basically what I have noticed is that if there is only 40 training examples it is nearly impossible to get an average looking training run.


And overall, we want to run the
```shell
split_data.py
SkullStripping
Registration
TissueSegmentation
Measurement
svm.py
```
