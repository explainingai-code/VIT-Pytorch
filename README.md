Vision Transformer (VIT) Implementation in pytorch on mnist images on textures
========

This repository implements DallE-1 on a synthetic dataset of mnist colored numbers on textures/solid background .

## Vision Transformer Videos
[Patch Embedding Video](https://www.youtube.com/watch?v=lBicvB4iyYU) 

[Attention Block Video](https://www.youtube.com/watch?v=zT_el_cjiJw)

[Building Vision Transformer Video](https://www.youtube.com/watch?v=G6_IA5vKXRI) 


Sample from dataset

<img src="https://github.com/explainingai-code/DallE/assets/144267687/57e3c091-4600-401d-a5a4-52ea5fda3249" width="300">


## Data preparation
For setting up the mnist dataset:
Follow - https://github.com/explainingai-code/Pytorch-VAE#data-preparation

Download Quarter RGB resolution texture data from [ALOT Homepage](https://aloi.science.uva.nl/public_alot/)
In case you want to train on higher resolution, you can download that as well and but you would have to create new imdb.json
Rest of the code should work fine as long as you create valid json files.

Download imdb.json from [Drive](https://drive.google.com/file/d/1dtbFhDCDJVp4OYlAzhVY_mzWTkrYrFlt/view?usp=sharing)
Verify the data directory has the following structure after textures download
```
VIT-Pytorch/data/textures/{texture_number}
	*.png
VIT-Pytorch/data/train/images/{0/1/.../9}
	*.png
VIT-Pytorch/data/test/images/{0/1/.../9}
	*.png
VIT-Pytorch/data/imdb.json
```

# Quickstart
* Create a new conda environment with python 3.8 then run below commands
* ```git clone https://github.com/explainingai-code/VIT-Pytorch.git```
* ```cd VIT-Pytorch```
* ```pip install -r requirements.txt```
* ```python -m tools.train``` for training vit
* ```python -m tools.inference``` for running inference, attention visualizations and positional embedding plots

## Configuration
* ```config/default.yaml``` - Allows you to play with different aspects of VIT 


## Output 
Outputs will be saved according to the configuration present in yaml files.

For every run a folder of ```task_name``` key in config will be created 
* Best Model checkpoint in ```task_name``` directory

During inference the following output will be saved
* Attention map visualization for sample of test set in ```task_name/output``` 
* Positional embedding similarity plots in  ```task_name/output/position_plot.png```


## Sample Output for VIT 

Following is a sample attention map that you should get

Here is a positional embedding similarity plot you should get






