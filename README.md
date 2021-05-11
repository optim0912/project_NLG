# project_NLG

This project is a simple translator project that converts Korean and English to each other. 
It supports not only the training of the translator, but also the use of the trained translator and the creation of the dataset required for training. 
In addition, we plan to support **TPU distributed training** for translator training. 
This project was created based on the [simple-nmt](https://github.com/kh-kim/simple-nmt) project.

## How to create a training dataset 

This section describes how to obtain a dataset that was used for training, validation, and evaluation of a trained translator. 
If you already have your own dataset or just want to use the translator, you can skip this section. 

As the dataset, we used the [AI Hub's Korean-English translation corpus dataset](https://aihub.or.kr/aidata/87/download). 
Please understand that this dataset cannot be provided directly due to copyright issues. 
So you have to download it from that site. Sign up is required and sadly you need to apply for permission to download. (It takes about a day for download permission.)

When the download is complete, you need to save it to the *dataset_process/original_dataset* folder. 
When a total of 10 xlsx files are saved, copy-paste each file to the text editor to create 10 txt files. (This can be a tedious task.)

The dataset is now ready for preprocessing. The pre-processing process consists of concatenation, shuffle, split, tokenization, and subword segmentation. 
For pre-processing, you need to install a Korean tokenizer(mecab-ko) and an English tokenizer(mosestokenizer). 
If it is not installed, enter the command below.

```
pip install mosestokenizer
bash ./install_mecab_linux.sh
```

The pre-processing process is automated through the provided shell script. It may take approximately a few minutes. ()
The final results are saved in the *dataset* folder. Several files created in the *dataset_process* folder on the way are not needed when training the translator, so you can delete them. (The code for subword segmentation is based on the [subword-nmt](https://github.com/kh-kim/subword-nmt) project.) 

```
bash ./dataset_process.sh
```
