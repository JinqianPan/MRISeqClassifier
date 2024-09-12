# MRISeqClassifier

<div align='center'><h2>MRISeqClassifier: A Deep Learning Toolkit for Precise MRI Sequence Classification</h2></div>
MRISeqClassifier is an open-source toolkit designed for classifying sequences (series) of MRI scans.

## Install
Create a new conda environment and install the required packages:

```
conda create -n nacc_img python=3.9
conda activate nacc_img

git clone git@github.com:JinqianPan/MRISeqClassifier.git
cd MRISeqClassifier
pip install -r requirements.txt
```

## Download Data And Pretraind-models
If you want to use another data and train the model by yourself, you could skip this step.

### Data
This toolkit use the NACC MRI dataset. Please request data from https://naccdata.org/.

>[!TIP]
> For how to `download the image data` from NACC AWS S3 bucket, please browse [my guide: NACC Image Data Download Experience](https://github.com/JinqianPan/NACC_img_download).
>
> For how to `preprocess NACC MRI data`, please browse [my guide: NACC MRI Data Processing](https://github.com/JinqianPan/NACC_image).

After preprocessing the data, please change the `IMAGE_PATH` in the yaml file `00_config.yml`.

### Pretrained-models
Please download pretrained-models from , and put them into folder `02_models`.

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
