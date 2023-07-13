# BDRC-Net

This repository is the official implementation of BDRC-Net.
**The code will be released after the paper has been accepted.**

## Requirements

To install requirements:
```setup
pip install -r requirements.txt
```

## Data Preparation

Download Thumos14 from [BaiDuYun](https://pan.baidu.com/s/12t2cLcP60rrkYKWtqEcdlg) (code: bdrc).

Please ensure the data structure is as below:

~~~~
├── data
   └── thumos
       ├── val
           ├── video_validation_0000051_02432.npz
           ├── video_validation_0000051_02560.npz
           └── ...
       └── test
           ├── video_test_0000004_00000.npz
           ├── video_test_0000004_00256.npz
           └── ...
       └── detclasslist.txt
       └── th14_groundtruth.json
~~~~

## Training

To train the BDRC-Net model on THUMOS'14 dataset, please first modify parameters in:
```parameters
./experiments/BDRCNet_thumos.yaml
```
Then run this command:
```train
python main.py --cfg experiments/BDRCNet_thumos.yaml
```

## Pre-trained Models

You can download pretrained models here:

- [THUMOS14](https://pan.baidu.com/s/1r2YIawVVRlekSO0G3IcaLg), (code: bdrc), trained on THUMOS'14 using parameters same as "./experiments/BDRCNet_thumos.yaml".

## Test

The command for test is
```test
python model_inference.py --cfg experiments/BDRCNet_thumos.yaml --resume output/thumos_cc/model_34.pth
```

## Acknowledgement

Our code is based on [A2Net](https://github.com/VividLe/A2Net?utm_source=catalyzex.com). We would like to express our gratitude for this outstanding work.
