# elec5622project1
ELEC5622 Project 1 90-dim SVM classification of Alzeimers brain AAL ROI volumes using bet, fsl, sklearn

Documentation for download order, etc

To be tested once I get on a linux machine:
```shell
git clone https://github.com/Kameradna/elec5622project1.git
cd elec5622project1
gdown https://drive.google.com/u/0/uc?id=1U8Bk_kZpOUZnXD34daumRdCuaQgxMhEo&export=download&confirm=t&uuid=4389a206-532a-4ff6-8d38-af73ddc5b284
gdown https://drive.google.com/uc?id=1-fnKe4yjIWFnodyPP8RiyeHBKxSfxXOv
unzip '*.zip'
mkdir -p Data/train Data/test Output/train Output/test
conda create -n elec5622project1 python=3.7 pip scikit-learn pandas numpy --yes
conda activate elec5622project1
python data_split.py

```
And so we arrive at the beginning of the project
