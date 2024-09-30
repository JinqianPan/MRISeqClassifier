# MRISeqClassifier

<div align='center'><h2>MRISeqClassifier: A Deep Learning Toolkit for Precise MRI Sequence Classification</h2></div>
MRISeqClassifier is an open-source toolkit designed for classifying sequences (series) of MRI scans.  <br><br>

<div align="center">
  <img src="https://github.com/JinqianPan/MRISeqClassifier/blob/main/02_models/arch.jpg">
</div>

## News
[2024 0919] Paper submit to medRxiv  
[2024 0917] Paper submit to AMIA 2025 Informatics Summit  
[2024 0912] Github available  
[2024 0806] The project started

## Install
Create a new conda environment and install the required packages:

```
conda create -n nacc_img python=3.9
conda activate nacc_img

git clone git@github.com:JinqianPan/MRISeqClassifier.git
cd MRISeqClassifier
pip install -r requirements.txt
```

>[!IMPORTANT]
> In the `requirements.txt`, the PyTorch installation is not included. 
>
> Please use `nvidia-smi` to verify your CUDA version and then download the appropriate version of PyTorch from the [official website](https://pytorch.org/).


## Download Data And Best Models
If you want to use another data and train the model by yourself, you could skip this step.

### Data
This toolkit is training by NACC MRI dataset. Please request data from https://naccdata.org/.

>[!TIP]
> For how to `download the image data` from NACC AWS S3 bucket, please browse [my guide: NACC Image Data Download Experience](https://github.com/JinqianPan/NACC_img_download).
>
> For how to `preprocess NACC MRI data`, please browse [my guide: NACC MRI Data Processing](https://github.com/JinqianPan/NACC_image).

After preprocessing the data, please change the `IMAGE_PATH` in the yaml file `00_config.yml`.

<div align="center">
  <img src="https://github.com/JinqianPan/MRISeqClassifier/blob/main/01_data/ImageFolder/example.jpg">
</div>

### Best Models
Please download best models from [Google Drive](https://drive.google.com/drive/folders/1kY7bkytT5G3ihGefWhfwZOErwvvaufC5?usp=sharing), and put them into folder `02_models`.

>[!NOTE]
> If you are using your own dataset, please construct the training and testing sets as follows: (we use `torchvision.datasets.ImageFolder` to build data loader)
> ```
> 01_data
>     |- ImageFolder
>            |- train
>                 |- T1WI (label name)
>                     |- MRI.jpg
>                 |- T2WI
>                 |- ...
>            |- test
>                 |- ...
> ```

## Quick start
If you just want to use this tool to classify MRI images, please use the code below:
```
$PATH = 'your image path'
python 05_toolkit.py --path $PATH
```

>[!IMPORTANT]
> Just remember: Before training, this toolkit can only classify 6 types mentioned above. 
> 
> If you want to use your own data, please go through the step training.

## Training
For training data, you could use code below, or use slurm to run the code.
```
nohup python -u 03_training.py --proximal middle --model densenet121 --epoch 100 --fold 10 > ./output/10-Fold/denseNet121/output_mid.log 2>&1 &
```

After training all models, you could use code below to get the result of voting:
```
nohup python -u 04_training_vote.py --proximal middle --fold 10 > ./output/10-Fold/vote/output_mid.log 2>&1 &
```

## Citation
If you use MRISeqClassifier in your research, please cite our paper:

```
@article {Pan2024.09.19.24313976,
	author = {Pan, Jinqian and Chen, Qi and Sun, Chengkun and Liang, Renjie and Bian, Jiang and Xu, Jie},
	title = {MRISeqClassifier: A Deep Learning Toolkit for Precise MRI Sequence Classification},
	elocation-id = {2024.09.19.24313976},
	year = {2024},
	doi = {10.1101/2024.09.19.24313976},
	publisher = {Cold Spring Harbor Laboratory Press},
	URL = {https://www.medrxiv.org/content/early/2024/09/22/2024.09.19.24313976},
	journal = {medRxiv}
}
```
