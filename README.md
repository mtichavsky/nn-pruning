# BIN project - neural net pruning

This project contains an implementation of an evolutionary algorithm that prunes convolutional cores of ResNet-18 
trained on CIFAR-10 dataset.

> **Warning:** it's necessary to fix the fitness calculation - the model size should take into account size of
> each convolutional filter - currently each is counted as 1 no matter how large it is. So that you can say
> how many parameters the resulting models have.

## Usage

To install all the necessary requirements, run

```bash
make venv 
source venv/bin/activate
```

To see the program help, run:

```bash
python main.py
```

Running this program on Metacentrum is explained in [metacentrum/README.md](metacentrum/README.md).
