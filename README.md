# NLP-Caffe

NLP-Caffe is a pull request [1] on the Caffe framework developed by Yangqing Jia and Evan Shelhamer, among other members of the BVLC lab at Berkeley and a large number of independent online contributers. 

This fork makes it easier for NLP users to get started without merging C++ code. The current example constructs a mid-size language model for the Penn Tree Bank using LSTMs that processes in excess of 15,000 words per second [2] and achieves a perplexity of 142. More examples for Machine Translation using the encoder-decoder model and character-level RNNs are in the works. Hopefully, this code will eventually be merged into the Caffe master branch. This work was performed under the guidance of <a href="http://nlp.stanford.edu/~manning/">Chris Manning</a> and with the invaluable expertise of <a href="http://stanford.edu/~lmthang/">Thang Luong</a>.

# Installation

Please consult the Caffe <a href="http://caffe.berkeleyvision.org/installation.html">installation instructions</a> first. After successfully installing caffe, just clone this repo and run `make -j8 && make pycaffe` from the NLP-Caffe folder.

To run the python LMDB database generation, you will also need to install <a href="https://github.com/dw/py-lmdb/">py-lmdb</a> with:

    pip install py-lmdb

# Tutorial

First, cd to the caffe root directory and download the data for the Penn Tree Bank with:

    ./data/ptb/get_ptb.sh

Using this data, you can generate the LMDB databases and the architecture train_val.prototxt with:

    python ./examples/ptb/create_ptb.py --make_data

You'll notice this generates train, test, and validation databases in examples/ptb. It also generates the train_val.prototxt architecture file and the solver.prototxt hyperparameter file. By editing this file, you can control the hyperparameters, dataset, and architecture used by NLP-Caffe with a python interface.

You can now begin to train the network with:

    ./examples/ptb/train_ptb.sh

The resulting wordvectors can be viewed with:

    ipython notebook ./examples/ptb/ptb_visualization.ipynb

# Further reading

To get a better general understanding of how Caffe works, you can take advantage of the content in the <a href="http://caffe.berkeleyvision.org/installation.html">caffe tutorials section</a>. In particular, the <a href="http://caffe.berkeleyvision.org/gathered/examples/mnist.html">MNIST tutorial</a> is useful to understand Caffe's command line interface and architecture description text files. To learn how to investigate the weights and performance of a trained model with an IPython Notebook, the <a href="http://nbviewer.ipython.org/github/BVLC/caffe/blob/master/examples/filter_visualization.ipynb">filter visualization tutorial</a> may be helpful.

<br>
<br>

[1] All citations should be addressed to the <a href="https://github.com/BVLC/caffe">main Caffe repository</a>. Licensing is identical to that of a Caffe pull request.

[2] The average sentence has 19.85 words when long sentences are capped to 30 words. We train batches of 128 sentences with an average batch time of 0.128 s on a Nvidia GTX 780 TI.
