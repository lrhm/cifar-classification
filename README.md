# cifar-classification
A simple cifar classification task with 30 ViT and CNNs.

# Getting started
First, install the dev version of Mate from this [link](https://github.com/ilex-paraguariensis/yerbamate/tree/lightning).

Then, install the dependencies:
```bash
pip install -r project/requirements.txt
```
Keep in mind that the project requires pytorch 1.12.0, pytorch-lightning 1.7.5 and above while the requirements installs the nightly versions of both.

## Running the project
To run the project, you can use Mate to run different configurations. Look at `project/models/resnet/hyperparatmers/cifar10.json` and `project/models/vit/hyperparameters/cifar10.json` for examples of configurations. Any configuration file can be selected to train. To train a model, run:
```bash
mate train {model_name} {hyperparameter_file}
```
where `{model_name}` is either `resnet` or `vit` and `{hyperparameter_file}` is the name of the hyperparameter file.

## Logging
The project uses [Weights and Biases](https://wandb.ai/) to log the training process. To log your training, you need to create an account and install the `wandb` package. 
```
pip install wandb
```
Then, you need to login to your account:
```
wandb login
```

You can also select any pytorch lightning loggers, for example `TensorBoardLogger` or `CSVLogger` in the hyperparameter file.

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

## Special thanks
Special thanks to the legend lucidrains for the [vit-pytorch](https://github.com/lucidrains/vit-pytorch) library. All the ViTs are based on his work.