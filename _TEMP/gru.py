#!encoding:utf-8
import sys
import math
import numpy as np
import tensorflow as tf

class GRU(object):

    def __init__(self, input_dim, hidden_dim, seq_len, scope = 'rnn'):
        self.inp_dim = input_dim
        self.hdd_dim = hidden_dim
        self.seq_len = seq_len
        self.kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
        self.bias_initializer=tf.zeros_initializer()
        with tf.variable_scope(scope):
            #basic rnn
            self.b_w_kernal = tf.get_variable('rnn_w_kernal', (self.hdd_dim, self.inp_dim), initializer = self.kernel_initializer)
            self.b_w_bias = tf.get_variable('rnn_w_bias', (self.hdd_dim, ), initializer = self.kernel_initializer)
            self.b_u_kernal = tf.get_variable('rnn_u_kernal', (self.hdd_dim, self.hdd_dim), initializer = self.kernel_initializer)
            self.b_u_bias = tf.get_variable('rnn_u_bias', (self.hdd_dim, ), initializer = self.bias_initializer)
            #reset gate
            self.r_w_kernal = tf.get_variable('reset_w_kernal', (self.hdd_dim, self.inp_dim), initializer = self.kernel_initializer)
            self.r_w_bias = tf.get_variable('reset_w_bias', (self.hdd_dim, ), initializer = self.bias_initializer)
            self.r_u_kernal = tf.get_variable('reset_u_kernal', (self.hdd_dim, self.hdd_dim), initializer = self.kernel_initializer)
            self.r_u_bias = tf.get_variable('reset_u_bias', (self.hdd_dim, ), initializer = self.bias_initializer)
            #update gate
            self.z_w_kernal = tf.get_variable('update_w_kernal', (self.hdd_dim, self.inp_dim), initializer = self.kernel_initializer)
            self.z_w_bias = tf.get_variable('update_w_bias', (self.hdd_dim, ), initializer = self.bias_initializer)
            self.z_u_kernal = tf.get_variable('update_u_kernal', (self.hdd_dim, self.hdd_dim), initializer = self.kernel_initializer)
            self.z_u_bias = tf.get_variable('update_u_bias', (self.hdd_dim, ), initializer = self.bias_initializer)

    def step_forward(self, inp, hdd):
        inp_shape = inp.get_shape().as_list()
        hdd_shape = hdd.get_shape().as_list()
        assert(len(inp_shape) == 2 and inp_shape[1] == self.inp_dim) #stepwise
        assert(len(hdd_shape) == 2 and hdd_shape[1] == self.hdd_dim) #stepwise
        assert(inp_shape[0] == hdd_shape[0]) #batch
        r1 = tf.matmul(inp, self.r_w_kernal, transpose_b=True) + self.r_w_bias
        r2 = tf.matmul(hdd, self.r_u_kernal, transpose_b=True) + self.r_u_bias
        r = tf.nn.sigmoid(r1 + r2)
        z1 = tf.matmul(inp, self.z_w_kernal, transpose_b=True) + self.z_w_bias
        z2 = tf.matmul(hdd, self.z_u_kernal, transpose_b=True) + self.z_u_bias
        z = tf.nn.sigmoid(z1 + z2)
        #print 'inp.shape:', inp.shape
        #print 'hdd.shape:', hdd.shape
        #print 'r.shape:', r.shape
        #print 'z.shape:', z.shape
        hh1 = tf.matmul(inp, self.b_w_kernal, transpose_b=True) + self.b_w_bias
        hh2 = tf.matmul(r * hdd, self.b_u_kernal, transpose_b=True) + self.b_u_bias
        hh = tf.nn.tanh(hh1 + hh2)
        #print 'hh.shape:', hh.shape
        h = z * hdd + (1.0 - z) * hh
        #print 'h.shape:', h.shape
        return h

    def do_infer(self, inp):
        inp_shape = inp.get_shape().as_list()
        assert(len(inp_shape) == 3 and inp_shape[1] == self.seq_len and inp_shape[2] == self.inp_dim)
        state = tf.zeros(shape=[inp_shape[0], self.hdd_dim], dtype=tf.float32)
        outputs = []
        for i in range(self.seq_len):
            state = self.step_forward(inp[:,i,:], state)
            outputs.append(tf.expand_dims(state, axis=1))
            #print outputs[-1].shape
        outputs = tf.concat(outputs, axis=1)
        return outputs

if __name__ == '__main__':
    batch_size, input_dim, hidden_dim, seq_len = 2, 3, 5, 6
    gru = GRU(input_dim, hidden_dim, seq_len)
    sess = tf.Session()
    vs = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    #print 'global variables list:'
    #for vi in vs:
    #    print vi.name, vi.get_shape().as_list()
    #print '\nbatch size:', batch_size
    #print 'input_dim:', input_dim
    #print 'hidden_dim:', hidden_dim
    #print 'seq_len:', seq_len
    #inp = tf.random_uniform([batch_size, input_dim], -0.1, 0.1)
    #hdd = tf.random_uniform([batch_size, hidden_dim], -0.1, 0.1)
    #gru.step_forward(inp, hdd)
    inp = tf.random_uniform([batch_size, seq_len, input_dim], -0.1, 0.1)
    out = gru.do_infer(inp)
    sess.run(tf.global_variables_initializer())
    print sess.run(out)
    
