#!/usr/bin/env python
import caffe_pb2
from caffe_pb2 import NetParameter, LayerParameter, DataParameter

import sys
import lmdb
import random
from caffe_pb2 import Datum
import subprocess
import itertools

source_length = 30
target_length = 20
source_vocab_size = 41000
target_vocab_size = 41000
num_categories = 1000
category_size = 41
assert num_categories * category_size == target_vocab_size

s_unknown_symbol = target_vocab_size + source_vocab_size - 3
s_start_symbol = target_vocab_size + source_vocab_size - 2
s_zero_symbol = target_vocab_size + source_vocab_size - 1

t_unknown_symbol = target_vocab_size - 3
t_start_symbol = target_vocab_size - 2
t_zero_symbol = target_vocab_size - 1

rand_skip = 11 * 10 ** 6
train_batch_size = 64
deploy_batch_size = 10


def make_data():
    for phase in ['train', 'valid', 'test']:
        db_name = './models/rnn/rnn_%s_db' % phase
        subprocess.call(['rm', '-r', db_name])
        env = lmdb.open(db_name, map_size=2147483648*8)


        def vocab_transform(source_input, target_input):
            def s_foo(x):
                return x if x < s_unknown_symbol else s_unknown_symbol
            def t_foo(x):
                return x if x < t_unknown_symbol else t_unknown_symbol

            source_line = [s_foo(int(x)) for x in source_input.split(' ')[:source_length]]
            target_line = [t_foo(int(x)) for x in target_input.split(' ')[:target_length]]

            source_line = source_line[:source_length] + [s_zero_symbol] * (source_length - len(source_line[:source_length]))
            target_line = target_line[:target_length] + [t_zero_symbol] * (target_length - len(target_line[:target_length]))
            assert len(source_line) == source_length
            assert len(target_line) == target_length
            return [source_line, target_line]

        allX = []
        with open('/home/stewartr/data/zhen/shuffled_%s.40k.id.en' % phase, 'r') as f1: 
            with open('/home/stewartr/data/zhen/shuffled_%s.40k.id.zh' % phase, 'r') as f2: 
                for en, zh in itertools.izip(f1.readlines(), f2.readlines()):
                #for en, zh in itertools.islice(itertools.izip(f1.readlines(), f2.readlines()), 100000):
                    allX.append(vocab_transform(en, zh))

        assert phase != 'train' or len(allX) > rand_skip


        with env.begin(write=True) as txn:
            for i, (source_line, target_line) in enumerate(allX):
                datum = Datum()
                datum.channels = source_length + 2 * target_length
                datum.width = 1
                datum.height = 1
                if i % 1000 == 0:
                    sys.stderr.write('%s\r' % i); sys.stderr.flush()
                for j in range(source_length):
                    if j == 0:
                        datum.float_data.append(s_start_symbol)
                    else:
                        datum.float_data.append(source_line[::-1][j - 1])
                for j in range(target_length):
                    if j == 0:
                        datum.float_data.append(t_start_symbol)
                    else:
                        datum.float_data.append(target_line[j - 1])
                for j in range(target_length):
                    datum.float_data.append(target_line[j])
                key = str(i)
                txn.put(key, datum.SerializeToString())

def display_layer(net, name):
    layer = net.layers.add()
    layer.name = 'display_%s' % name
    layer.top.append('display_%s' % name)
    layer.bottom.append(name)
    layer.bottom.append(name)
    layer.type = LayerParameter.ELTWISE
    layer.eltwise_param.coeff.append(0.5)
    layer.eltwise_param.coeff.append(0.5)

def add_weight_filler(param):
    param.type = 'uniform'
    param.min = -0.1
    param.max = 0.1

def get_net(deploy, batch_size):
    net = NetParameter()
    lstm_num_cells = 250
    wordvec_length = 200

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
            test_data.data_param.source = 'models/rnn/rnn_test_db'
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
    data_slice_layer.top.append('words')

    data_slice_layer.top.append('target')
    data_slice_layer.slice_param.slice_point.append(source_length + target_length)

    wordvec_layer = net.layers.add()
    wordvec_layer.name = "wordvec_layer"
    wordvec_layer.type = LayerParameter.WORDVEC
    wordvec_layer.bottom.append('words')
    wordvec_layer.top.append(wordvec_layer.name)
    wordvec_layer.wordvec_param.dimension = wordvec_length
    wordvec_layer.wordvec_param.vocab_size = source_vocab_size + target_vocab_size
    add_weight_filler(wordvec_layer.wordvec_param.weight_filler)


    input_slice_layer = net.layers.add()
    input_slice_layer.name = "input_slice_layer"
    input_slice_layer.type = LayerParameter.SLICE
    input_slice_layer.slice_param.slice_dim = 0
    input_slice_layer.bottom.append('wordvec_layer')

    for i in range(source_length + target_length):
        input_slice_layer.top.append('wordvec%d' % i)
        if i != 0:
            input_slice_layer.slice_param.slice_point.append(i * batch_size)

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

        concat_layer0 = net.layers.add()
        concat_layer0.name = 'concat0_layer%d' % i
        lstm_layer0 = net.layers.add()
        lstm_layer0.name = 'lstm0_layer%d' % i

        for j, (concat_layer, lstm_layer) in enumerate([(concat_layer0, lstm_layer0)]):
            concat_layer.top.append(concat_layer.name)
            concat_layer.type = LayerParameter.CONCAT
            concat_layer.bottom.append('wordvec%d' % i)
            if j == 1:
                concat_layer.bottom.append('lstm0_hidden%d' % i)
            if i == 0:
                concat_layer.bottom.append(dummy_layer.name)
            else:
                concat_layer.bottom.append('lstm%d_hidden%d' % (j, i - 1))

            lstm_layer.type = LayerParameter.LSTM
            lstm_layer.lstm_param.num_cells = lstm_num_cells

            add_weight_filler(lstm_layer.lstm_param.input_weight_filler)
            add_weight_filler(lstm_layer.lstm_param.input_gate_weight_filler)
            add_weight_filler(lstm_layer.lstm_param.forget_gate_weight_filler)
            add_weight_filler(lstm_layer.lstm_param.output_gate_weight_filler)

            for k in range(4):
                lstm_layer.param.append('lstm%d_param%d' % (j, k))
            lstm_layer.top.append('lstm%d_hidden%d' % (j, i))
            lstm_layer.top.append('lstm%d_mem_cell%d' % (j, i))
            lstm_layer.bottom.append('concat%d_layer%d' % (j, i))
            if i == 0:
                lstm_layer.bottom.append('dummy_mem_cell')
            else:
                lstm_layer.bottom.append('lstm%d_mem_cell%d' % (j, i - 1))

    hidden_concat_layer = net.layers.add()
    hidden_concat_layer.type = LayerParameter.CONCAT
    hidden_concat_layer.name = 'hidden_concat'
    hidden_concat_layer.top.append(hidden_concat_layer.name)
    hidden_concat_layer.concat_param.concat_dim = 0
    for i in range(source_length, source_length + target_length):
        hidden_concat_layer.bottom.append('lstm0_hidden%d' % i)

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
    silence_layer.name = "silence%d" % (source_length-1)
    silence_layer.type = LayerParameter.SILENCE
    silence_layer.bottom.append("lstm0_mem_cell%d" % (source_length + target_length - 1))

    return net

def main():
    if '--make_data' in sys.argv:
        make_data()

    with open('./models/rnn/train_val.prototxt', 'w') as f:
        f.write('name: "RussellNet"\n')
        f.write(str(get_net(False, train_batch_size)));

    with open('./models/rnn/deploy.prototxt', 'w') as f:
        f.write('name: "RussellNet"\n')
        f.write('''
input: "data"
input_dim: %s
input_dim: %s
input_dim: 1
input_dim: 1

''' % (deploy_batch_size, source_length + 2 * target_length))
        f.write(str(get_net(True, deploy_batch_size)))
if __name__ == '__main__':
    main()
