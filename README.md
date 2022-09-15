# Cifar classification with [Maté 🧉](https://github.com/ilex-paraguariensis/yerbamate)
A simple cifar classification task with 30 ViTs and CNNs. All the ViT models are sourced from [lucidrains vit-pytorch](https://github.com/lucidrains/vit-pytorch) repository. This project requires pytorch, pytorch-lightning, and [Maté 🧉](https://github.com/ilex-paraguariensis/yerbamate) and supports pretrained models from any installed python package, for example torchvision and huggingface models. 


# Getting started

You can run mate on your local machine or run a jupyter notebook on google colab.

## Colab

You can run the notebook on colab by clicking on the following badge:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lrhm/cifar-classification/blob/main/vit_mate.ipynb)

A Jupiter notebook is also available in the repository.

## Install locally

First, install the dev version of Maté from lightning branch [link](https://github.com/ilex-paraguariensis/yerbamate/tree/lightning).

Then, install the dependencies:
```bash
pip install -r project/requirements.txt
```


## Running the project
To run the project, you can use Mate to run different configurations. Look at `resnet/hyperparatmers/vanilla.json` and `vit/hyperparameters/vanilla.json` for examples of configurations. Any configuration file can be selected to train. To train a model, run:
```bash
mate train {model_name} {hyperparameter_file}
```
where `{model_name}` can be anything e.g., `resnet` or `vit` and `{hyperparameter_file}` is the name of the hyperparameter file and the experiment.

## Logging
The project by default uses [Weights and Biases](https://wandb.ai/) to log the training process. To log your training with wandb you need to create an account and install the `wandb` package. 
```
pip install wandb
```
Then, you need to login to your account:
```
wandb login
```

You can also select any pytorch lightning loggers, e.g., `TensorBoardLogger` or `CSVLogger`. See `/vit/hyperparateres/tensorboard.json` for an example.

## Training

You can select any combination of your models with hyperparameters, for example:
```bash
mate train resnet fine_tune # fine tune a resnet trained on imagenet on cifar
mate train vit small_datasets #  model from Vision Transformer for Small-Size Datasets paper
mate train vit vanilla # original ViT paper: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
```

You can consequently restart the training with the same configuration by running:
```bash
mate restart vit vanilla
```

## Results
With wandb logger, you can visualize the results, for example fine tuning a resnet on cifar:

<img src="/images/train_accuracy.png" alt="drawing" width="80%"/>
<img src="/images/val_accuracy.png" alt=drawing width="80%" />
<img src="/images/train_loss.png" alt="drawing" width="80%"/>
<img src="/images/val_loss.png" alt="drawing" width="80%"/>

## Experimenting and trying other models
You can try other models by changing the model in the hyperparameters or making new configuration file. Over 30 ViTs are available to experiment with. You can also fork the vit models and change the source code as you wish:
```bash
mate clone vit awesome_vit
```
Then, change the models in `project/models/awesome_vit` and keep on experimenting.

## Customizing the hyperparameters
You can customize the hyperparameters by changing the hyperparameter file. For example, you can change the  model, learning rate, batch size, optimizer, etc. this project is not limited to cifar dataset, with adding a PytorchLightningDataModule, you can train on any dataset. Optimizers, Trainers, Models and Pytorch-Lightning modules are directly created from the arguments in the configuration file and pytorch packages.

## Special thanks
Special thanks to the legend lucidrains for the [vit-pytorch](https://github.com/lucidrains/vit-pytorch) library. His licence applies to the ViT models in this project.