# NLP-Caffe

NLP-Caffe is a pull request on the Caffe framework developed by Yangqing Jia and Evan Shelhamer, among other members of the BVLC lab at Berkeley and a large number of independent online contributers. 

This fork makes it easier for NLP users to get started without merging C++ code. Currently, there is an example of a language model for the Penn Tree Bank using LSTMs that processes 20,000 words per second and achieves a perplexity of 128. More examples for Machine Translation using the encoder-decoder model and character-level RNNs will be forthcoming. Eventually this code should be merged into the Caffe master branch. This work was performed under the guidance of <a href="http://nlp.stanford.edu/~manning/">Chris Manning</a> and with the invaluable expertise of <a href="http://stanford.edu/~lmthang/">Thang Luong</a>.

# Installation

Please consult the Caffe <a href="http://caffe.berkeleyvision.org/installation.html">installation instructions</a> first. After successfully installing caffe, just clone this repo and run `make -j8 && make pycaffe` from the NLP-Caffe folder.

To run the python LMDB database generation, you will also need to install <a href="https://github.com/dw/py-lmdb/">py-lmdb</a> with:

    pip install py-lmdb

# Tutorial

First, cd to the caffe root directory and download the data for the Penn Tree Bank with:

    ./data/ptb/get_ptb.sh

Using this data, you can generate the LMDB database and the architecture train_val.prototxt with:

    ./examples/ptb/create_ptb.sh
    
You'll notice this generates train, test, and validation databases in examples/ptb. It also generates the train_val.prototxt architecture file and the solver.prototxt hyperparameter file. You can now begin to train the network with:

    ./examples/ptb/train_ptb.sh
