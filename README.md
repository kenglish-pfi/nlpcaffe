# NLP-Caffe

NLP-Caffe is a pull request on the Caffe framework developed by Yangqing Jia and Evan Shelhamer, among other members of the BVLC lab at Berkeley and a large number of independent online contributers. This fork makes it easier for NLP users to get started without merging C++ code. Currently, there is an example of a language model for the Penn Tree Bank using LSTMs that processes 20,000 words per second and achieves a perplexity of 128. More examples for Machine Translation using the encoder-decoder model and character-level RNNs will be forthcoming. Eventually this code should be merged into the Caffe master branch.

# Installation

Please consult the Caffe <a href="http://caffe.berkeleyvision.org/installation.html">installation instructions</a> first. After successfully installing caffe, just clone this repo and run `make -j8` from the NLP-Caffe folder.

To run the python LMDB database generation, you will also need to install <a href="https://lmdb.readthedocs.org/en/release/">py-lmdb</a> with:

    pip install py-lmdb
