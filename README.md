Deep Learning on Graphs with Keras
====

Keras-based implementation of graph convolutional networks for semi-supervised classification [1]. This also implements filters from [2] and makes use of the Cora dataset from [3].

Installation
------------

Pip:
```pip install /path/to/repository```

Distutils:
```python setup.py install```

Dependencies
-----

  * Keras (2.0.0 or higher)
  * TensorFlow, Theano, or CNTK (see Keras documentation on how to configure)

Usage
-----

Basic training with default parameters:
```python train.py```

To see help about other parameters:
```python train.py -h```

References
----------

[1] [Kipf & Welling, Semi-Supervised Classification with Graph Convolutional Networks, 2016](https://arxiv.org/abs/1609.02907)

[2] [Defferrard et al., Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering, NIPS 2016](https://arxiv.org/abs/1606.09375)

[3] [Sen et al., Collective Classification in Network Data, AI Magazine 2008](http://linqs.cs.umd.edu/projects/projects/lbc/)
