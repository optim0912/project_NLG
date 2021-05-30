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

```bash
>> pip install mosestokenizer
>> bash ./install_mecab_linux.sh
```

The pre-processing process is automated through the provided shell script. It may take approximately a few minutes.
The final results are saved in the *dataset* folder. Several files created in the *dataset_process* folder on the way are not needed when training the translator, so you can delete them. (The code for subword segmentation is based on the [subword-nmt](https://github.com/kh-kim/subword-nmt) project.) 

```bash
>> bash ./dataset_process.sh
```

## TranslationDataLoader class

This class is specific to the dataset created and preprocessed in the previous section. This class creates a batch from the dataset and loads each batch. For options and usage of the class, refer to *dataloader_test.ipynb* in the *sample* folder. If you are using a personal custom dataset, you may need to change the TranslationDataloader class to suit the dataset. For optimal performance, the options of the provided class (for example, num_workers, lazy, num_parallel_reads, etc.) need to be adjusted according to the user's environment.

When the TranslationDataloader is tested in the Colab environment, the results are shown in the table below. The time taken for initialization (the time taken to create one dataloader instance), the time taken for iteration (the time taken to output sentence pairs in 100 batches from one dataloader), and the memory usage of one dataloader were measured. Except for the three options (lazy, num_workers, num_parallel_reads), the rest of the options are default.

|                                                                   | initialization time (seconds) | iteration time (seconds) | memory usage (MB) |
|:-----------------------------------------------------------------:|:-----------------------------:|:------------------------:|:-----------------:|
|  lazy : True  <br/> num_workers : 0 <br/> num_parallel_reads : 1  |             206.255           |           13.752         |       **263**     |
|  lazy : True  <br/> num_workers : 0 <br/> num_parallel_reads : 2  |             185.152           |           13.521         |       **263**     |
|  lazy : True  <br/> num_workers : 2 <br/> num_parallel_reads : 1  |             199.420           |           10.082         |       **263**     |
|  lazy : True  <br/> num_workers : 2 <br/> num_parallel_reads : 2  |             183.965           |           10.175         |       **263**     |
|  lazy : False <br/> num_workers : 0 <br/> num_parallel_reads : 1  |             188.080           |         **3.127**        |        10156      |
|  lazy : False <br/> num_workers : 0 <br/> num_parallel_reads : 2  |           **158.450**         |         **3.092**        |        10156      |
|  lazy : False <br/> num_workers : 2 <br/> num_parallel_reads : 1  |             171.935           |           3.715          |        10156      |
|  lazy : False <br/> num_workers : 2 <br/> num_parallel_reads : 2  |           **158.063**         |           4.317          |        10156      |

When lazy is True, memory usage is greatly reduced, but iteration time increases. This is because lazy loading stores only the offsets of sentences in memory instead of all the sentences themselves contained in the entire text files. However, since text files must be opened every time a batch is created, I/O overhead may occur and iteration time increases. Therefore, for large datasets, it is better to set the lazy option to True, and if the training time is important, it is better to set it to False. When lazy is False, an additional effect of improving the initialization time can be obtained. This is because inefficient readline() can be avoided when reading text files.

If num_parallel_reads is greater than 1, multiple text files can be read at the same time, thus improving the initialization time. If num_workers is greater than 1, batch generation is executed in parallel using several processes, thus improving iteration time. However, be careful as performance may deteriorate depending on the amount of RAM and the number of CPU or GPU cores. 
