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
conda create -n elec5622project1 -f requirementsbasic.yml #or full
conda activate elec5622project1
python data_split.py

```
Additionally, we need fsleyes and nifty-reg to complete the skullstripping and registration, and measurements. Clone the niftyreg github and compile for your system following the install guide (ccmake, make, make install, add lines to .bashrc). Install fsleyes via conda. You may need to get conda-forge as a source for installation of fsleyes, guides are available.


And so we arrive at the beginning of the project. With 90-dim dummy random data, we get insane overfitting so radical 90 to 2 pca must be undertaken to allow for reasonable model parameters.
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
To summarise the whole process-

- split_data does some data allocation based on whether we have labels
- SkullStripping does some skull stripping and places the output in the output directory respective to the input (filename-stripped)
- Registration learns the deformable registration of the AAL atlas to native space for each image and applies it (aal_to_filename_transformed)
- TissueSegmentation segments via fast the greymatter regions from others for each image
- Measurement finds the volume of greymatter in each ROI via looping through and using fslstats, placing the values in an output csv.
- svm.py performs data splitting and trains an svm, perhaps some PCA or other techniques.


With the actual data, PCA makes no difference unless you use 1 single component. Any other training scheme will net you 100% training accuracy. The conclusions for the test set are unchanged for most training schema.

{'Data_40': 1, 'Data_41': -1, 'Data_42': 1, 'Data_43': -1, 'Data_44': 1, 'Data_45': -1, 'Data_46': 1, 'Data_47': 1, 'Data_48': 1, 'Data_49': 1}

Where 1 is Alzeimer's and -1 is normal control.

![image](https://user-images.githubusercontent.com/48018617/191641669-90be4f20-4020-4c14-83a1-e5bcb38c2460.png)
It was at this time that we realised that our use of -B had aggressively trimmed parts of the alzeimers examples. See Figure, the grey area is the original -B flag, the yellow is the extent of using -B -f 0.4 and the green is the extent of using the appropriate -B -f 0.3 flags.

This gives us much superior whole brain stripping.
