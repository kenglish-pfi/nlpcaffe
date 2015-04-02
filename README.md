# NLP-Caffe

NLP-Caffe is a <a href="https://github.com/Russell91/nlp_caffe/blob/master/CHANGES.txt" target="_blank">pull request</a> [1] on the Caffe framework developed by Yangqing Jia and Evan Shelhamer, among other members of the BVLC lab at Berkeley and a large number of independent online contributers. 

This fork makes it easier for NLP users to get started without merging C++ code. The current example constructs a language model for a small subset of Google's Billion Word corpus. It uses a two-layer LSTM architecture that processes in excess of 15,000 words per second [2], and achieves a perplexity of 79. More examples for Machine Translation using the encoder-decoder model and character-level RNNs are in the works. Hopefully, this code will eventually be merged into the Caffe master branch. This work was funded by the <a href="http://nlp.stanford.edu/" target="_blank">Stanford NLP Group</a>, under the guidance of <a href="http://nlp.stanford.edu/~manning/" target="_blank">Chris Manning</a>, and with the invaluable expertise of <a href="http://stanford.edu/~lmthang/" target="_blank">Thang Luong</a>.

# Installation

We recommend consulting the Caffe <a href="http://caffe.berkeleyvision.org/installation.html" target="_blank">installation instructions</a> and compiling the standard Caffe library first. Next, clone this repo and run `make -j8 && make pycaffe` from the NLP-Caffe folder.

NLP-Caffe also requires <a href="https://github.com/dw/py-lmdb/" target="_blank">py-lmdb</a> at runtime, which can be installed with:

    pip install py-lmdb

# Tutorial

First, cd to the caffe root directory and download the data for the model with:

    ./data/language_model/get_lm.sh

Using this data, you can generate the LMDB databases and the architecture train_val.prototxt with:

    python ./examples/launguage_model/create_lm.py --make_data

You'll notice this generates train, test, and validation databases in examples/language_model. It also generates the train_val.prototxt architecture file and the solver.prototxt hyperparameter file. By editing this file, you can control the hyperparameters, dataset, and architecture used by NLP-Caffe with a python interface.

You can now begin to train the network with:

    ./examples/language_model/train_lm.sh

The resulting wordvectors can be viewed with:

    ipython notebook ./examples/language_model/lm_visualization.ipynb

# Further reading

To get a better general understanding of how Caffe works, you can take advantage of the content in the <a href="http://caffe.berkeleyvision.org/installation.html" target="_blank">caffe tutorials section</a>. In particular, the <a href="http://nbviewer.ipython.org/github/BVLC/caffe/blob/master/examples/filter_visualization.ipynb" target="_blank">filter visualization tutorial</a> is a great example of how you can use the IPython notebook to investigate the weights and performance of a trained model. The <a href="http://caffe.berkeleyvision.org/gathered/examples/mnist.html" target="_blank">MNIST tutorial</a> is useful to understand how you can control Caffe over the command line and through architecture description text files.

<br>
<br>

[1] All citations should be addressed to the <a href="https://github.com/BVLC/caffe" target="_blank">main Caffe repository</a>. Licensing is identical to that of a Caffe pull request.

[2] The average sentence has 22.2 words when long sentences are capped to 30 words. We train batches of 128 sentences with an average batch time of 0.166 s on a Nvidia GTX 780 TI.
