# NeurIPS MedFM Challenge: 2nd Place - 2023

This repository contains the code to reproduce the 2nd place submission for the [Foundation Model Prompting for Medical Image Classification Challenge 2023 (MedFM)](https://medfm2023.grand-challenge.org/medfm2023/).

It is based on [MMPreTrain](https://github.com/open-mmlab/mmpretrain), which which uses the backbones [**`ViT-cls`**](./configs/vit-b_vpt/), [**`ViT-eva02`**](./configs/eva-b_vpt/), [**`ViT-dinov2`**](./configs/dinov2-b_vpt/), [**`Swin-cls`**](./configs/swin-b_vpt/) and [**`ViT-clip`**](./configs/clip-b_vpt/). More details could be found in its [document](https://mmpretrain.readthedocs.io/en/latest/index.html).

## Installation

Install requirements by

```bash
$ conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.1 -c pytorch
$ pip install openmim scipy scikit-learn ftfy regex tqdm
$ mim install mmpretrain
```

## Usage

### Preparation

Prepare data following [MMPreTrain](https://github.com/open-mmlab/mmpretrain). Download the dataset and unzip the '.zip' files into {$MedFM}/data/ as following: 

```text
MedFM (root)/
    ├── docker               # dockerfiles
    ├── configs              # all the configs
    │   ├── clip-b_vpt
    │   ├── swin-b_vpt
    │   ├── dinov2-b_vpt
    │   └── ...
    ├── data               
    │   ├── MedFMC_train      # unzip the download file
    │   ├── MedFMC_val
    │   └── ...
    ├── data_anns
    │   ├── MedFMC
    │   |    ├── chest
    │   |    ├── colon
    │   |    ├── endo
    │   ├── result             # sample example to submit 
    │   └── ...
    ├── medfmc                 # all source code
    ├── tools                  # train, test and other tools
    └── ...
```

<details><summary>click to show the detail of data_anns</summary>
Note that the `.txt` files includes data split information for fully supervised learning and few-shot learning tasks.
The public dataset is split into `trainval.txt` and `test_WithLabel.txt`, and `trainval.txt` is also split into `train_20.txt` and `val_20.txt` where `20` means the training data makes up 20% of `trainval.txt`.
`test_WithoutLabel.txt` of each dataset represents the validation set.

Corresponding `.txt` files are stored at `./data_anns/` folder, the few-shot learning data split files `{dataset}_{N_shot}-shot_train/val_exp{N_exp}.txt` could also be generated as below:

```shell
python tools/generate_few-shot_file.py
```

Where `N_shot` is 1,5 and 10, respectively, the shot is of patient(i.e., 1-shot means images of certain one patient are all counted as one), not number of images.

### Training and  Testing

For training and testing, you can directly use the commands listed below:

```bash
# you need to export path in terminal so the `custom_imports` in config would work
export PYTHONPATH=$PWD:$PYTHONPATH
# Training
python tools/train.py $CONFIG

# Evaluation
python tools/test.py $CONFIG $CHECKPOINT 

```

### Generating Submission results

Run

```bash
python tools/infer.py $CONFIF $WEIGHT $IMAGE_FOLDER --batch-size 4 --out $OUT_FILE_PATH
```

## Using the MedFMC repository with Docker

<details><summary>click to show the detail</summary>

More details of Docker could be found in this [tutorial](https://nbviewer.org/github/ericspod/ContainersForCollaboration/blob/master/ContainersForCollaboration.ipynb).

### Preparation of Docker

We provide a [Dockerfile](./docker/Dockerfile) to build an image. Ensure that your [docker version](https://docs.docker.com/engine/install/) >=19.03.

```
# build an image with PyTorch 1.11, CUDA 11.3
# If you prefer other versions, just modified the Dockerfile
docker build -t medfmc docker/
```

Run it with

```
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/medfmc/data medfmc
```

The submitted docker will be evaluated by the following command:

```bash
docker container run --gpus all --shm-size=8g -m 28G -it --name teamname --rm -v $PWD:/medfmc_exp -v $PWD/data:/medfmc_exp/data teamname:latest /bin/bash -c "sh /medfmc_exp/run.sh"
```

- `--gpus`: specify the number of available GPUs during inference
- `-m`: specify the maximum RAM
- `--name`: container name
- `--rm`: remove the container after running
- `-v $PWD:/medfmc_exp`: map local codebase folder to Docker `medfmc_exp` folder.
- `-v $PWD/data:/medfmc_exp/data`: map local codebase folder to Docker `medfmc_exp/data` folder.
- `teamname:latest`: docker image name (should be `teamname`) and its version tag. **The version tag should be the `latest`**. 
- `/bin/bash -c "sh run.sh"`: start the prediction command.

Assuming the team name is `baseline`, the Docker build command is

```shell
docker build -t baseline .
```

> During the inference, please monitor the GPU memory consumption using `watch nvidia-smi`. The GPU memory consumption should be less than 10G. Otherwise, it will run into an OOM error on the official evaluation server.

### 3) Save Docker

```shell
docker save baseline | gzip -c > baseline.tar.gz
```

## License

This project is released under the [Apache 2.0 license](LICENSE).
