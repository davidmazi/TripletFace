Triplet loss for facial recognition.

# Student Report

By playing with the hyperparameters, I realized that changing the architecture from Resnet-18 to Resnet-152, a batch size of 64, with 5 epochs and 16 workers was efficient enough, the training time was shorter than with other tested values, and the visualization appears to be sharper.

The architecture change was motivated by the [PyTorch docs](https://pytorch.org/docs/stable/torchvision/models.html), as they state the 1-crop error rates of this architecture are lower than the Resnet-18's.

## JIT Compile

On Google Colab, the code block to JIT Compile the model is the following :

```python
%cd TripletFace
import torch

from tripletface.core.model import Encoder

model = Encoder(64)
weights = torch.load("/content/TripletFace/model/model.pt")['model']
model.load_state_dict( weights )
#These values were defined by looking at the docs and reading some posts on StackOverflow
jit_model = torch.jit.trace( model, torch.rand(3, 3, 3, 3) )
torch.jit.save( jit_model, "/content/TripletFace/JIT/jitcompile.pt" )

%cd ..
```

This allows some Just In Time compilation to be used afterwards.

## Centroids and Thresholds

Despite my best efforts to implement this script, the results weren't conclusive. I have therefore omitted this part from my report.

# Triplet Face

The repository contains code for the application of triplet loss training to the
task of facial recognition. This code has been produced for a lecture and is not
going to be maintained in any sort.

![TSNE_Latent](TSNE_Latent.png)

## Architecture

The proposed architecture is pretty simple and does not implement state of the
art performances. The chosen architecture is a fine tuning example of the
resnet18 CNN model. The model includes the freezed CNN part of resnet, and its
FC part has been replaced to be trained to output latent variables for the
facial image input.

The dataset needs to be formatted in the following form:

```
dataset/
| test/
| | 0/
| | | 00563.png
| | | 01567.png
| | | ...
| | 1/
| | | 00011.png
| | | 00153.png
| | | ...
| | ...
| train/
| | 0/
| | | 00001.png
| | | 00002.png
| | | ...
| | 1/
| | | 00001.png
| | | 00002.png
| | | ...
| | ...
| labels.csv        # id;label
```

## Install

Install all dependencies ( pip command may need sudo ):

```bash
cd TripletFace/
pip3 install -r requirements.txt
```

## Usage

For training:

```bash
usage: train.py [-h] -s DATASET_PATH -m MODEL_PATH [-i INPUT_SIZE]
                [-z LATENT_SIZE] [-b BATCH_SIZE] [-e EPOCHS]
                [-l LEARNING_RATE] [-w N_WORKERS] [-r N_SAMPLES]

optional arguments:
  -h, --help            show this help message and exit
  -s DATASET_PATH, --dataset_path DATASET_PATH
  -m MODEL_PATH, --model_path MODEL_PATH
  -i INPUT_SIZE, --input_size INPUT_SIZE
  -z LATENT_SIZE, --latent_size LATENT_SIZE
  -b BATCH_SIZE, --batch_size BATCH_SIZE
  -e EPOCHS, --epochs EPOCHS
  -l LEARNING_RATE, --learning_rate LEARNING_RATE
  -w N_WORKERS, --n_workers N_WORKERS
  -r N_SAMPLES, --n_samples N_SAMPLES
```

## References

- Resnet Paper: [Arxiv](https://arxiv.org/pdf/1512.03385.pdf)
- Triplet Loss Paper: [Arxiv](https://arxiv.org/pdf/1503.03832.pdf)
- TripletTorch Helper Module: [Github](https://github.com/TowardHumanizedInteraction/TripletTorch)

## Todo ( For the students )

**Deadline Decembre 13th 2019 at 12pm**

The students are asked to complete the following tasks:

- Fork the Project
- Improve the model by playing with Hyperparameters and by changing the Architecture ( may not use resnet )
- JIT compile the model ( see [Documentation](https://pytorch.org/docs/stable/jit.html#torch.jit.trace) )
- Add script to generate Centroids and Thesholds using few face images from one person
- Generate those for each of the student included in the dataset
- Add inference script in order to use the final model
- Change README.md in order to include the student choices explained and a table containing the Centroids and Thesholds for each student of the dataset with a vizualisation ( See the one above )
- Send the github link by mail
