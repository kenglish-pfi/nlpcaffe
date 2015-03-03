#!/usr/bin/env python

import config
import os
import sys
import lmdb
import random
import subprocess
import itertools
import argparse
import sys
sys.path.append('python/caffe/proto'); import caffe_pb2

from caffe_pb2 import NetParameter, LayerParameter, DataParameter, SolverParameter
from caffe_pb2 import Datum

target_length = 20
target_vocab_size = 41000
num_categories = 1000
num_lstm_stacks = 4
category_size = 41
assert num_categories * category_size == target_vocab_size

t_unknown_symbol = target_vocab_size - 3
t_start_symbol = target_vocab_size - 2
t_zero_symbol = target_vocab_size - 1

data_size_limit = 4*10**4
#data_size_limit = 11 * 10 ** 6
#data_size_limit = 11 * 10 ** 6
rand_skip = min(data_size_limit - 1, 11 * 10 ** 6)
train_batch_size = 128
deploy_batch_size = 10

def make_data():
    #for phase in ['train', 'valid', 'test']:
    for phase in ['train']:
        db_name = './models/rnn/rnn_%s_db' % phase
        subprocess.call(['rm', '-r', db_name])
        env = lmdb.open(db_name, map_size=2147483648*8)


        def vocab_transform(target_input):
            def t_foo(x):
                return x if x < t_unknown_symbol else t_unknown_symbol

            target_line = [t_foo(int(x)) for x in target_input.split(' ')[:target_length]]

            target_line = target_line[:target_length] + [t_zero_symbol] * (target_length - len(target_line[:target_length]))
            assert len(target_line) == target_length
            return target_line

        allX = []
        with open('%s/1bil/shuffled_%s.40k.id.en' % (config.data_dir, phase), 'r') as f1: 
            for en in itertools.islice(f1.readlines(), data_size_limit):
                allX.append(vocab_transform(en))

        assert phase != 'train' or len(allX) > rand_skip


        with env.begin(write=True) as txn:
            for i, target_line in enumerate(allX):
                datum = Datum()
                datum.channels = 2 * target_length
                datum.width = 1
                datum.height = 1
                if i % 1000 == 0:
                    sys.stderr.write('%s\r' % i); sys.stderr.flush()
                for j in range(target_length):
                    if j == 0:
                        datum.float_data.append(t_start_symbol)
                    else:
                        datum.float_data.append(target_line[j - 1])
                for j in range(target_length):
                    datum.float_data.append(target_line[j])
                key = str(i)
                txn.put(key, datum.SerializeToString())

def get_solver():
    solver = SolverParameter()
    solver.net = "models/rnn/train_val.prototxt"
    #solver.test_iter.append(10)
    #solver.test_interval = 500
    solver.base_lr = 1.0
    solver.weight_decay = 0.0000
    solver.lr_policy = "fixed"
    solver.display = 20
    solver.max_iter = 1000000000
    solver.max_grad = 1.0
    solver.snapshot = 10000
    solver.snapshot_prefix = "%s/%s" % (config.snapshot_dir, os.path.dirname(os.path.realpath(__file__)).split('/')[-1])
    solver.random_seed = 17
    solver.solver_mode = SolverParameter.GPU
    return solver

def display_layer(net, name):
    layer = net.layers.add()
    layer.name = 'display_%s' % name
    layer.top.append('display_%s' % name)
    layer.bottom.append(name)
    layer.bottom.append(name)
    layer.type = LayerParameter.ELTWISE
    layer.eltwise_param.coeff.append(0.5)
    layer.eltwise_param.coeff.append(0.5)

def add_weight_filler(param, max_value=0.07):
    param.type = 'uniform'
    param.min = -max_value
    param.max = max_value

def get_net(deploy, batch_size):
    net = NetParameter()
    lstm_num_cells = 1000
    wordvec_length = 1000

    if not deploy:
        train_data = net.layers.add()
        train_data.type = LayerParameter.DATA
        train_data.name = "data"
        train_data.top.append(train_data.name)
        train_data.data_param.source = 'models/rnn/rnn_train_db'
        train_data.data_param.backend = DataParameter.LMDB
        train_data.data_param.batch_size = batch_size
        train_data.data_param.rand_skip = rand_skip

        if True:
            test_data = net.layers.add()
            test_data.type = LayerParameter.DATA
            test_data.name = "data"
            test_data.top.append(test_data.name)
            test_data.data_param.source = 'models/rnn/rnn_valid_db'
            test_data.data_param.backend = DataParameter.LMDB
            test_data.data_param.batch_size = batch_size
            test_data.data_param.rand_skip = rand_skip

            test_data_rule = test_data.include.add()
            test_data_rule.phase = caffe_pb2.TEST
            train_data_rule = train_data.include.add()
            train_data_rule.phase = caffe_pb2.TRAIN


    data_slice_layer = net.layers.add()
    data_slice_layer.name = "data_slice_layer"
    data_slice_layer.type = LayerParameter.SLICE
    data_slice_layer.slice_param.slice_dim = 1
    data_slice_layer.bottom.append('data')
    data_slice_layer.top.append('target_words')
    data_slice_layer.top.append('target')
    data_slice_layer.slice_param.slice_point.append(target_length)

    target_wordvec_layer = net.layers.add()
    target_wordvec_layer.name = "target_wordvec_layer"
    target_wordvec_layer.type = LayerParameter.WORDVEC
    target_wordvec_layer.bottom.append('target_words')
    target_wordvec_layer.top.append(target_wordvec_layer.name)
    target_wordvec_layer.wordvec_param.dimension = wordvec_length
    target_wordvec_layer.wordvec_param.vocab_size = target_vocab_size
    add_weight_filler(target_wordvec_layer.wordvec_param.weight_filler)

    target_wordvec_slice_layer = net.layers.add()
    target_wordvec_slice_layer.name = "target_wordvec_slice_layer"
    target_wordvec_slice_layer.type = LayerParameter.SLICE
    target_wordvec_slice_layer.slice_param.slice_dim = 2
    target_wordvec_slice_layer.bottom.append('target_wordvec_layer')
    for i in range(target_length):
        target_wordvec_slice_layer.top.append('target_wordvec%d' % i)
        if i != 0:
            target_wordvec_slice_layer.slice_param.slice_point.append(i)


    for i in range(target_length):
        if i == 0:
            dummy_layer = net.layers.add()
            dummy_layer.name = 'dummy_layer'
            dummy_layer.top.append(dummy_layer.name)
            dummy_layer.type = LayerParameter.DUMMY_DATA
            dummy_layer.dummy_data_param.num.append(batch_size)
            dummy_layer.dummy_data_param.channels.append(lstm_num_cells)
            dummy_layer.dummy_data_param.height.append(1)
            dummy_layer.dummy_data_param.width.append(1)

            dummy_mem_cell = net.layers.add()
            dummy_mem_cell.name = 'dummy_mem_cell'
            dummy_mem_cell.top.append(dummy_mem_cell.name)
            dummy_mem_cell.type = LayerParameter.DUMMY_DATA
            dummy_mem_cell.dummy_data_param.num.append(batch_size)
            dummy_mem_cell.dummy_data_param.channels.append(lstm_num_cells)
            dummy_mem_cell.dummy_data_param.height.append(1)
            dummy_mem_cell.dummy_data_param.width.append(1)


        for j in range(num_lstm_stacks):
            concat_layer = net.layers.add()
            concat_layer.name = 'concat%d_layer%d' % (j, i)

            concat_layer.top.append(concat_layer.name)
            concat_layer.type = LayerParameter.CONCAT
            if j == 0:
                concat_layer.bottom.append('target_wordvec%d' % i)
            if j >= 1:
                concat_layer.bottom.append('dropout%d_%d' % (j - 1, i))
            if i == 0:
                concat_layer.bottom.append(dummy_layer.name)
            else:
                concat_layer.bottom.append('lstm%d_hidden%d' % (j, i - 1))

            #concat_ip_layer = net.layers.add()
            #concat_ip_layer.name = "concat_ip%d_layer%d" % (j, i)
            #concat_ip_layer.top.append(concat_ip_layer.name)
            #concat_ip_layer.bottom.append('concat%d_layer%d' % (j, i))
            #concat_ip_layer.type = LayerParameter.INNER_PRODUCT
            #concat_ip_layer.inner_product_param.bias_term = False
            #concat_ip_layer.inner_product_param.num_output = 250
            #concat_ip_layer.param.append('concat_ip%d' % (j))
            #add_weight_filler(concat_ip_layer.inner_product_param.weight_filler)

            #relu_layer = net.layers.add()
            #relu_layer.name = 'relu_' + concat_ip_layer.name
            #relu_layer.type = LayerParameter.RELU
            #relu_layer.top.append(concat_ip_layer.top[0])
            #relu_layer.bottom.append(concat_ip_layer.top[0])

            lstm_layer = net.layers.add()
            lstm_layer.name = 'lstm%d_layer%d' % (j, i)
            lstm_layer.type = LayerParameter.LSTM
            lstm_layer.lstm_param.num_cells = lstm_num_cells

            add_weight_filler(lstm_layer.lstm_param.input_weight_filler)
            add_weight_filler(lstm_layer.lstm_param.input_gate_weight_filler)
            add_weight_filler(lstm_layer.lstm_param.forget_gate_weight_filler)
            add_weight_filler(lstm_layer.lstm_param.output_gate_weight_filler)

            for k in range(4):
                lstm_layer.param.append('lstm%d_param_%d' % (j, k))
            lstm_layer.top.append('lstm%d_hidden%d' % (j, i))
            lstm_layer.top.append('lstm%d_mem_cell%d' % (j, i))
            #lstm_layer.bottom.append('concat_ip%d_layer%d' % (j, i))
            lstm_layer.bottom.append('concat%d_layer%d' % (j, i))
            if i == 0:
                lstm_layer.bottom.append('dummy_mem_cell')
            else:
                lstm_layer.bottom.append('lstm%d_mem_cell%d' % (j, i - 1))

            batchnorm_layer = net.layers.add()
            batchnorm_layer.name = 'dropout%d_%d' % (j, i)
            batchnorm_layer.type = LayerParameter.BN
            batchnorm_layer.top.append(batchnorm_layer.name)
            batchnorm_layer.bottom.append('lstm%d_hidden%d' % (j, i))
            #batchnorm_layer.batchnorm_param.norm_dim = 0
            batchnorm_layer.batchnorm_param.norm_dim = 0
            batchnorm_layer.bn_param.scale_filler.value = 1
            for k in range(2):
                batchnorm_layer.param.append('batchnorm_param_%s' % j)

    hidden_concat_layer = net.layers.add()
    hidden_concat_layer.type = LayerParameter.CONCAT
    hidden_concat_layer.name = 'hidden_concat'
    hidden_concat_layer.top.append(hidden_concat_layer.name)
    hidden_concat_layer.concat_param.concat_dim = 0
    for i in range(target_length):
        hidden_concat_layer.bottom.append('dropout%d_%d' % (num_lstm_stacks - 1, i))

    inner_product_layer = net.layers.add()
    inner_product_layer.name = "inner_product"
    inner_product_layer.top.append(inner_product_layer.name)
    inner_product_layer.bottom.append('hidden_concat')
    inner_product_layer.type = LayerParameter.INNER_PRODUCT
    inner_product_layer.inner_product_param.bias_term = False
    inner_product_layer.inner_product_param.num_output = num_categories
    add_weight_filler(inner_product_layer.inner_product_param.weight_filler)

    softmax_product_layer = net.layers.add()
    softmax_product_layer.name = "softmax_product"
    softmax_product_layer.top.append(softmax_product_layer.name)
    softmax_product_layer.top.append('target_local_id')
    softmax_product_layer.top.append('target_category_id')
    softmax_product_layer.bottom.append('hidden_concat')
    softmax_product_layer.bottom.append('target')
    softmax_product_layer.type = LayerParameter.SOFTMAX_PRODUCT
    softmax_product_layer.softmax_product_param.num_output = category_size
    softmax_product_layer.softmax_product_param.num_categories = num_categories
    add_weight_filler(softmax_product_layer.softmax_product_param.weight_filler)

    if deploy:
        category_prob_layer = net.layers.add()
        category_prob_layer.name = "category_prob"
        category_prob_layer.type = LayerParameter.SOFTMAX
        category_prob_layer.bottom.append("inner_product")
        category_prob_layer.top.append(category_prob_layer.name)

        local_prob_layer = net.layers.add()
        local_prob_layer.name = "local_prob"
        local_prob_layer.type = LayerParameter.SOFTMAX
        local_prob_layer.bottom.append("softmax_product")
        local_prob_layer.top.append(local_prob_layer.name)
    else:
        category_loss_layer = net.layers.add()
        category_loss_layer.name = "category_loss"
        category_loss_layer.type = LayerParameter.SOFTMAX_LOSS
        category_loss_layer.bottom.append("inner_product")
        category_loss_layer.bottom.append("target_category_id")
        category_loss_layer.bottom.append("target")
        category_loss_layer.top.append(category_loss_layer.name)
        category_loss_layer.softmax_loss_param.empty_word = t_zero_symbol

        local_loss_layer = net.layers.add()
        local_loss_layer.name = "local_loss"
        local_loss_layer.type = LayerParameter.SOFTMAX_LOSS
        local_loss_layer.bottom.append("softmax_product")
        local_loss_layer.bottom.append("target_local_id")
        local_loss_layer.bottom.append("target")
        local_loss_layer.top.append(local_loss_layer.name)
        category_loss_layer.softmax_loss_param.empty_word = t_zero_symbol

    silence_layer = net.layers.add()
    silence_layer.name = "silence"
    silence_layer.type = LayerParameter.SILENCE
    for j in range(num_lstm_stacks):
        silence_layer.bottom.append("lstm%d_mem_cell%d" % (j, target_length - 1))
    for j in range(num_lstm_stacks - 1):
        silence_layer.bottom.append("dropout%d_%d" % (j, target_length - 1))

    return net

def write_solver():
    with open('./models/rnn/solver.prototxt', 'w') as f:
        f.write(str(get_solver()))

def write_net():
    with open('./models/rnn/train_val.prototxt', 'w') as f:
        f.write('name: "RussellNet"\n')
        f.write(str(get_net(False, train_batch_size)))

    with open('./models/rnn/deploy.prototxt', 'w') as f:
        f.write('name: "RussellNet"\n')
        f.write('''
input: "data"
input_dim: %s
input_dim: %s
input_dim: 1
input_dim: 1

''' % (deploy_batch_size, 2 * target_length))
        f.write(str(get_net(True, deploy_batch_size)))

parser = argparse.ArgumentParser()
parser.add_argument('--make_data', action='store_true')
args = parser.parse_args()

def main():
    if args.make_data:
        make_data()
    write_solver()
    write_net()

if __name__ == '__main__':
    main()
