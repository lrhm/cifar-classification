# Cifar classification with Maté
A simple cifar classification task with 30 ViTs and CNNs. All the ViT models are sourced from [lucidrains vit-pytorch](https://github.com/lucidrains/vit-pytorch) repository. 

# Getting started

You can run mate on your local machine or run a jupyter notebook on google colab.

## Colab

You can run the notebook on google colab by clicking on the following badge:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://raw.githubusercontent.com/lrhm/cifar-classification/main/vit_mate.ipynb)

## Run locally

First, install the dev version of Maté from lightning branch [link](https://github.com/ilex-paraguariensis/yerbamate/tree/lightning).

Then, install the dependencies:
```bash
pip install -r project/requirements.txt
```
Keep in mind that the project requires pytorch 1.12.1.

## Running the project
To run the project, you can use Mate to run different configurations. Look at `project/models/resnet/hyperparatmers/cifar10.json` and `project/models/vit/hyperparameters/cifar10.json` for examples of configurations. Any configuration file can be selected to train. To train a model, run:
```bash
mate train {model_name} {hyperparameter_file}
```
where `{model_name}` can be anything e.g., `resnet` or `vit` and `{hyperparameter_file}` is the name of the hyperparameter file.

## Logging
The project uses [Weights and Biases](https://wandb.ai/) to log the training process. To log your training, you need to create an account and install the `wandb` package. 
```
pip install wandb
```
Then, you need to login to your account:
```
wandb login
```

You can also select any pytorch lightning loggers, for example `TensorBoardLogger` or `CSVLogger` in the hyperparameter file. See `models/vit/hyperparateres/cifar-tensorboard.json` for an example.

## Training

You can select any combination of your models with hyperparameters, for example:
```bash
mate train resnet cifar10
mate train vit cifar10
mate train vit cifar_vit_for_small_datasets
```

You can consequently restart the training with the same configuration by running:
```bash
mate restart vit cifar10
```

## Experimenting and trying other models
You can try other models by changing the model in the hyperparameters or making new configuration file. Over 30 ViTs are available to experiment with. You can also fork the vit models and change the models as you wish:
```bash
mate clone vit awesome_vit
```
Then, change the models in `project/models/awesome_vit` and keep on experimenting.

## Customizing the hyperparameters
You can customize the hyperparameters by changing the hyperparameter file. For example, you can change the learning rate, batch size, optimizer, etc. You can also add new hyperparameters to the hyperparameter file. For example, you can add a new hyperparameter `tempature_learning_rate` to the hyperparameter file and use it in the model. The hyperparameter configuration is compatible with python objects, where you need to specify the module, class and parameters for initialization of the object. Optimizers, Trainers, Models and Pytorch-Lightning modules are directly created from the arguments in the configuration file and pytorch packages.

## Special thanks
Special thanks to the legend lucidrains for the [vit-pytorch](https://github.com/lucidrains/vit-pytorch) library. His licence applies to the ViT models in this project.