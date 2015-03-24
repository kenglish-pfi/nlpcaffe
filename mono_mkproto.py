#!/usr/bin/env python

from copy import deepcopy
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

from caffe_pb2 import NetParameter, LayerParameter, DataParameter, SolverParameter, ParamSpec
from caffe_pb2 import Datum

def make_data(param):
    for phase in ['train', 'valid', 'test']:
        db_name = './models/rnn/rnn_%s_db' % phase
        subprocess.call(['rm', '-r', db_name])
        env = lmdb.open(db_name, map_size=2147483648*8)


        def vocab_transform(target_input):
            def t_foo(x):
                return x if x < param['t_unknown_symbol'] else param['t_unknown_symbol']

            target_line = [t_foo(int(x)) for x in target_input.split(' ')[:param['target_length']]]

            target_line = target_line[:param['target_length']] + [param['t_zero_symbol']] * (param['target_length'] - len(target_line[:param['target_length']]))
            assert len(target_line) == param['target_length']
            return target_line

        allX = []
        #with open('%s/1bil/shuffled_%s.40k.id.en' % (config.data_dir, phase), 'r') as f1: 
        with open('%s/penn/%s_indices.txt' % (config.data_dir, phase), 'r') as f1: 
            for en in itertools.islice(f1.readlines(), param['data_size_limit']):
                allX.append(vocab_transform(en))

        print len(allX)
        assert phase != 'train' or len(allX) > param['rand_skip']


        with env.begin(write=True) as txn:
            for i, target_line in enumerate(allX):
                datum = Datum()
                datum.channels = 2 * param['target_length']
                datum.width = 1
                datum.height = 1
                if i % 1000 == 0:
                    sys.stderr.write('%s\r' % i); sys.stderr.flush()
                for j in range(param['target_length']):
                    if j == 0:
                        datum.float_data.append(param['t_start_symbol'])
                    else:
                        datum.float_data.append(target_line[j - 1])
                for j in range(param['target_length']):
                    datum.float_data.append(target_line[j])
                key = str(i)
                txn.put(key, datum.SerializeToString())

def get_solver(param):
    solver = SolverParameter()
    solver.net = param['file_train_val_net']
    solver.test_interval = param['solver_test_interval']
    solver.base_lr = param['solver_base_lr']
    solver.weight_decay = param['solver_weight_decay']
    solver.lr_policy = param['solver_lr_policy']
    solver.display = param['solver_display']
    solver.max_iter = param['solver_max_iter']
    solver.clip_gradients = param['solver_clip_gradients']
    solver.snapshot = param['solver_snapshot']
    solver.lr_policy = param['solver_lr_policy']
    solver.stepsize = param['solver_stepsize']
    solver.gamma = param['solver_gamma']
    solver.snapshot_prefix = param['solver_snapshot_prefix']
    solver.random_seed = param['solver_random_seed']
    solver.solver_mode = param['solver_solver_mode']
    solver.test_iter.append(param['solver_test_iter'])
    return solver


def display_layer(net, name):
    layer = net.layer.add()
    layer.name = 'display_%s' % name
    layer.top.append('display_%s' % name)
    layer.bottom.append(name)
    layer.bottom.append(name)
    layer.type = "Eltwise"
    layer.eltwise_param.coeff.append(0.5)
    layer.eltwise_param.coeff.append(0.5)

def add_weight_filler(param, max_value=0.07):
    param.type = 'uniform'
    param.min = -max_value
    param.max = max_value

def get_net(param, deploy, batch_size):
    net = NetParameter()
    lstm_num_cells = param['lstm_num_cells']
    wordvec_length = param['wordvec_length']

    if not deploy:
        train_data = net.layer.add()
        train_data.type = "Data"
        train_data.name = "data"
        train_data.top.append(train_data.name)
        train_data.data_param.source = 'models/rnn/rnn_train_db'
        train_data.data_param.backend = DataParameter.LMDB
        train_data.data_param.batch_size = batch_size
        train_data.data_param.rand_skip = param['rand_skip']

        if True:
            test_data = net.layer.add()
            test_data.type = "Data"
            test_data.name = "data"
            test_data.top.append(test_data.name)
            test_data.data_param.source = 'models/rnn/rnn_test_db'
            test_data.data_param.backend = DataParameter.LMDB
            test_data.data_param.batch_size = batch_size
            test_data.data_param.rand_skip = param['rand_skip']

            test_data_rule = test_data.include.add()
            test_data_rule.phase = caffe_pb2.TEST
            train_data_rule = train_data.include.add()
            train_data_rule.phase = caffe_pb2.TRAIN


    data_slice_layer = net.layer.add()
    data_slice_layer.name = "data_slice_layer"
    data_slice_layer.type = "Slice"
    data_slice_layer.slice_param.slice_dim = 1
    data_slice_layer.bottom.append('data')
    data_slice_layer.top.append('target_words')
    data_slice_layer.top.append('target')
    data_slice_layer.slice_param.slice_point.append(param['target_length'])

    target_wordvec_layer = net.layer.add()
    target_wordvec_layer.name = "target_wordvec_layer"
    target_wordvec_layer.type = "Wordvec"
    target_wordvec_layer.bottom.append('target_words')
    target_wordvec_layer.top.append(target_wordvec_layer.name)
    target_wordvec_layer.wordvec_param.dimension = wordvec_length
    target_wordvec_layer.wordvec_param.vocab_size = param['target_vocab_size']
    add_weight_filler(target_wordvec_layer.wordvec_param.weight_filler)

    target_wordvec_slice_layer = net.layer.add()
    target_wordvec_slice_layer.name = "target_wordvec_slice_layer"
    target_wordvec_slice_layer.type = "Slice"
    target_wordvec_slice_layer.slice_param.slice_dim = 2
    target_wordvec_slice_layer.slice_param.fast_wordvec_slice = True
    target_wordvec_slice_layer.bottom.append('target_wordvec_layer')
    for i in range(param['target_length']):
        target_wordvec_slice_layer.top.append('target_wordvec%d' % i)
        if i != 0:
            target_wordvec_slice_layer.slice_param.slice_point.append(i)


    for i in range(param['target_length']):
        if i == 0:
            dummy_layer = net.layer.add()
            dummy_layer.name = 'dummy_layer'
            dummy_layer.top.append(dummy_layer.name)
            dummy_layer.type = "DummyData"
            dummy_layer.dummy_data_param.num.append(batch_size)
            dummy_layer.dummy_data_param.channels.append(lstm_num_cells)
            dummy_layer.dummy_data_param.height.append(1)
            dummy_layer.dummy_data_param.width.append(1)

            dummy_mem_cell = net.layer.add()
            dummy_mem_cell.name = 'dummy_mem_cell'
            dummy_mem_cell.top.append(dummy_mem_cell.name)
            dummy_mem_cell.type = "DummyData"
            dummy_mem_cell.dummy_data_param.num.append(batch_size)
            dummy_mem_cell.dummy_data_param.channels.append(lstm_num_cells)
            dummy_mem_cell.dummy_data_param.height.append(1)
            dummy_mem_cell.dummy_data_param.width.append(1)


        for j in range(param['num_lstm_stacks']):
            concat_layer = net.layer.add()
            concat_layer.name = 'concat%d_layer%d' % (j, i)

            concat_layer.top.append(concat_layer.name)
            concat_layer.type = "Concat"
            concat_layer.concat_param.fast_lstm_concat = True
            if j == 0:
                concat_layer.bottom.append('target_wordvec%d' % i)
            if j >= 1:
                concat_layer.bottom.append('dropout%d_%d' % (j - 1, i))
            if i == 0:
                concat_layer.bottom.append(dummy_layer.name)
            else:
                concat_layer.bottom.append('lstm%d_hidden%d' % (j, i - 1))

            lstm_layer = net.layer.add()
            lstm_layer.name = 'lstm%d_layer%d' % (j, i)
            lstm_layer.type = "Lstm"
            lstm_layer.lstm_param.num_cells = lstm_num_cells

            add_weight_filler(lstm_layer.lstm_param.input_weight_filler)
            add_weight_filler(lstm_layer.lstm_param.input_gate_weight_filler)
            add_weight_filler(lstm_layer.lstm_param.forget_gate_weight_filler)
            add_weight_filler(lstm_layer.lstm_param.output_gate_weight_filler)

            for k in range(4):
                param_spec = lstm_layer.param.add()
                param_spec.name = 'lstm%d_param_%d' % (j, k)
            lstm_layer.top.append('lstm%d_hidden%d' % (j, i))
            lstm_layer.top.append('lstm%d_mem_cell%d' % (j, i))
            lstm_layer.bottom.append('concat%d_layer%d' % (j, i))
            if i == 0:
                lstm_layer.bottom.append('dummy_mem_cell')
            else:
                lstm_layer.bottom.append('lstm%d_mem_cell%d' % (j, i - 1))

            dropout_layer = net.layer.add()
            dropout_layer.name = 'dropout%d_%d' % (j, i)
            dropout_layer.type = "Dropout"
            dropout_layer.top.append(dropout_layer.name)
            dropout_layer.bottom.append('lstm%d_hidden%d' % (j, i))
            dropout_layer.dropout_param.dropout_ratio = 0.5

    hidden_concat_layer = net.layer.add()
    hidden_concat_layer.type = "Concat"
    hidden_concat_layer.name = 'hidden_concat'
    hidden_concat_layer.top.append(hidden_concat_layer.name)
    hidden_concat_layer.concat_param.concat_dim = 0
    for i in range(param['target_length']):
        hidden_concat_layer.bottom.append('dropout%d_%d' % (param['num_lstm_stacks'] - 1, i))

    inner_product_layer = net.layer.add()
    inner_product_layer.name = "inner_product"
    inner_product_layer.top.append(inner_product_layer.name)
    inner_product_layer.bottom.append('hidden_concat')
    inner_product_layer.type = "InnerProduct"
    inner_product_layer.inner_product_param.bias_term = False
    inner_product_layer.inner_product_param.num_output = param['num_categories']
    add_weight_filler(inner_product_layer.inner_product_param.weight_filler)

    softmax_product_layer = net.layer.add()
    softmax_product_layer.name = "softmax_product"
    softmax_product_layer.top.append(softmax_product_layer.name)
    softmax_product_layer.top.append('target_local_id')
    softmax_product_layer.top.append('target_category_id')
    softmax_product_layer.bottom.append('hidden_concat')
    softmax_product_layer.bottom.append('target')
    softmax_product_layer.type = "SoftmaxProduct"
    softmax_product_layer.softmax_product_param.num_output = param['category_size']
    softmax_product_layer.softmax_product_param.num_categories = param['num_categories']
    add_weight_filler(softmax_product_layer.softmax_product_param.weight_filler)

    if deploy:
        category_prob_layer = net.layer.add()
        category_prob_layer.name = "category_prob"
        category_prob_layer.type = "Softmax"
        category_prob_layer.bottom.append("inner_product")
        category_prob_layer.top.append(category_prob_layer.name)

        local_prob_layer = net.layer.add()
        local_prob_layer.name = "local_prob"
        local_prob_layer.type = "Softmax"
        local_prob_layer.bottom.append("softmax_product")
        local_prob_layer.top.append(local_prob_layer.name)
    else:
        category_loss_layer = net.layer.add()
        category_loss_layer.name = "category_loss"
        category_loss_layer.type = "SoftmaxWithLoss"
        category_loss_layer.bottom.append("inner_product")
        category_loss_layer.bottom.append("target_category_id")
        category_loss_layer.top.append(category_loss_layer.name)
        #category_loss_layer.softmax_loss_param.empty_word = param['t_zero_symbol']

        local_loss_layer = net.layer.add()
        local_loss_layer.name = "local_loss"
        local_loss_layer.type = "SoftmaxWithLoss"
        local_loss_layer.bottom.append("softmax_product")
        local_loss_layer.bottom.append("target_local_id")
        local_loss_layer.top.append(local_loss_layer.name)
        #category_loss_layer.softmax_loss_param.empty_word = param['t_zero_symbol']

    silence_layer = net.layer.add()
    silence_layer.name = "silence"
    silence_layer.type = "Silence"
    for j in range(param['num_lstm_stacks']):
        silence_layer.bottom.append("lstm%d_mem_cell%d" % (j, param['target_length'] - 1))
    for j in range(param['num_lstm_stacks'] - 1):
        silence_layer.bottom.append("dropout%d_%d" % (j, param['target_length'] - 1))

    return net

def write_solver(param):
    with open(param['file_solver'], 'w') as f:
        f.write(str(get_solver(param)))

def write_net(param):
    with open(param['file_train_val_net'], 'w') as f:
        f.write('name: "%s"\n' % param['net_name'])
        f.write(str(get_net(param, deploy=False, batch_size = param['train_batch_size'])))

    with open(param['file_deploy_net'], 'w') as f:
        f.write('name: "%s"\n' % param['net_name'])
        f.write('''
input: "data"
input_dim: %s
input_dim: %s
input_dim: 1
input_dim: 1

''' % (param['deploy_batch_size'], 2 * param['target_length']))
        f.write(str(get_net(param, deploy=True, batch_size = param['deploy_batch_size'])))

parser = argparse.ArgumentParser()
parser.add_argument('--make_data', action='store_true')
args = parser.parse_args()


def run(idx, param):
    base_param = {}

    base_param['net_name'] = "RussellNet"
    base_param['target_length'] = 30
    base_param['target_vocab_size'] = 11000
    base_param['num_categories'] = 11000
    base_param['num_lstm_stacks'] = 1
    base_param['category_size'] = 1
    assert base_param['num_categories'] * base_param['category_size'] == base_param['target_vocab_size']

    base_param['t_unknown_symbol'] = base_param['target_vocab_size'] - 3
    base_param['t_start_symbol'] = base_param['target_vocab_size'] - 2
    base_param['t_zero_symbol'] = base_param['target_vocab_size'] - 1

    base_param['data_size_limit'] = 42068
    #base_param['data_size_limit'] = 11 * 10 ** 6
    base_param['rand_skip'] = min(base_param['data_size_limit'] - 1, 3 * 10 ** 7)
    base_param['train_batch_size'] = 128
    base_param['deploy_batch_size'] = 32
    base_param['lstm_num_cells'] = 1000
    base_param['wordvec_length'] = 1000

    base_param['file_solver'] = "models/rnn/solver%d.prototxt" % idx
    base_param['file_train_val_net'] = "models/rnn/train_val%d.prototxt" % idx
    base_param['file_deploy_net'] = "models/rnn/deploy%d.prototxt" % idx
    base_param['solver_test_interval'] = 500
    base_param['solver_base_lr'] = 20
    base_param['solver_weight_decay'] = 0.0000
    base_param['solver_lr_policy'] = "fixed"
    base_param['solver_display'] = 20
    base_param['solver_max_iter'] = 30000
    base_param['solver_clip_gradients'] = 20.0
    base_param['solver_snapshot'] = 10000
    base_param['solver_lr_policy'] = 'step'
    base_param['solver_stepsize'] = 5000
    base_param['solver_gamma'] = 0.8
    base_param['solver_snapshot_prefix'] = "%s/%s.%d" % (config.snapshot_dir, os.path.dirname(os.path.realpath(__file__)).split('/')[-1], idx)
    base_param['solver_random_seed'] = 17
    base_param['solver_solver_mode'] = SolverParameter.GPU
    base_param['solver_test_iter'] = 10
    param.update(base_param)

    if args.make_data:
        make_data(param)
    write_solver(param)
    write_net(param)

def main():
    run(1, {})

if __name__ == '__main__':
    main()
