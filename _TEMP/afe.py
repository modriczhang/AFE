#!encoding=utf-8
"""
    A Peep into the Future: Adversarial Future Encoding in Recommendation
    @2021-02-05
    by modriczhang
"""
import os
import sys
import math
import time
import copy
import json
import random
import hashlib
import datetime
import subprocess
import numpy as np
import tensorflow as tf
from layer_util import *
from data_util import read_data
from hyper_param import param_dict as pd
from gru import GRU
from replay_buffer import RB

# parameters for distributed training
flags = tf.app.flags
flags.DEFINE_string("model_path", "train_model", "Saved model dir.")
flags.DEFINE_string("ps_hosts", "", "Comma-separated list of hostname:port pairs.")
flags.DEFINE_string("worker_hosts", "", "Comma-separated list of hostname:port pairs.")
flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = flags.FLAGS
# global training info
g_batch_counter = 0
g_working_mode = None
g_training = None
# replay buffer
g_rb = None

class FGAN(object):
    def __init__(self, global_step):
        self.global_step = global_step
        # cumulative loss
        self.b_loss, self.l_loss, self.h_loss = 0.0, 0.0, 0.0
        # network parameter
        self.n_batch, self.clk_len, self.rnn_len = pd['batch_size'], pd['clk_seq_len'], pd['rnn_seq_len']
        self.n_user, self.n_doc, self.n_con = pd['user_field_num'], pd['doc_field_num'], pd['con_field_num']
        self.n_b_dim, self.n_l_dim, self.hint_dim = pd['booster_feat_dim'], pd['lite_feat_dim'], pd['hint_dim']
        self.b_rnn, self.l_rnn = pd['booster_rnn_dim'], pd['lite_rnn_dim']
        self.gamma = pd['rl_gamma']
        self.b_lr, self.l_lr = pd['booster_lr'], pd['lite_lr']
        # placeholder
        self.sph_clk_seq = tf.sparse_placeholder(tf.int32, name='sph_clk_seq')
        self.sph_user = tf.sparse_placeholder(tf.int32, name='sph_user')
        self.sph_doc = tf.sparse_placeholder(tf.int32, name='sph_doc')
        self.sph_zdoc = tf.sparse_placeholder(tf.int32, name='sph_zdoc')
        self.sph_con = tf.sparse_placeholder(tf.int32, name='sph_con')
        self.ph_reward = tf.placeholder(tf.float32, shape=(self.n_batch * self.rnn_len), name='ph_reward')
        self.ph_nbq = tf.placeholder(tf.float32, shape=(self.n_batch, self.rnn_len), name='ph_nbq')
        self.ph_nlq = tf.placeholder(tf.float32, shape=(self.n_batch, self.rnn_len), name='ph_nlq')
        self.ph_guide_weight = tf.placeholder(tf.float32, shape=(self.n_batch * self.rnn_len), name='ph_guide_weight')
        self.ph_guide_q = tf.placeholder(tf.float32, shape=(self.n_batch, self.rnn_len), name='ph_guide_q')
        self.ph_hint = tf.placeholder(tf.float32, shape=(self.n_batch, self.rnn_len, self.hint_dim), name='ph_hint')
        self.gan_reward = tf.placeholder(tf.float32, name='gan_reward')
        # booster loss
        print '\n======\nbuilding booster main q network ...'
        self.bmh, self.bmq = self.build_booster_network('main')
        print '\n======\nbuilding booster target q network ...'
        _, self.btq = self.build_booster_network('target')
        yt = tf.reshape(self.ph_reward, [-1]) + tf.scalar_mul(tf.constant(self.gamma), tf.reshape(self.ph_nbq, [-1]))
        diff = yt - tf.reshape(self.bmq, [-1])
        self.booster_loss = tf.reduce_mean(tf.square(diff))
        # guide weight
        ee = tf.square(diff)
        maxe = tf.reduce_max(tf.reshape(ee, [-1]))
        mine = tf.reduce_min(tf.reshape(ee, [-1]))
        self.guide_weights = 1.0 - (ee-mine) / (maxe-mine+1e-3)
        
        vs = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='booster/main')
        vs.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='booster/embedding'))
        self.b_grads = tf.clip_by_global_norm(tf.gradients(self.booster_loss, vs), pd['grad_clip'])[0]
        with tf.variable_scope('opt_booster'):
            optimizer = tf.train.AdamOptimizer(self.b_lr)
            self.booster_opt = optimizer.apply_gradients(zip(self.b_grads, vs), global_step=global_step)
        m_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="booster/main")
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="booster/target")
        alpha = pd['double_networks_sync_step']
        self.booster_sync_op = [tf.assign(t, (1.0-alpha)*t + alpha*m) for t, m in zip(t_params, m_params)]
        
        # lite loss
        print '\n======\nbuilding lite main q network ...'
        self.lmh, self.lmq = self.build_lite_network('main')
        print '\n======\nbuilding lite target q network ...'
        _, self.ltq = self.build_lite_network('target')
        yt = tf.reshape(self.ph_reward, [-1]) + tf.scalar_mul(tf.constant(self.gamma), tf.reshape(self.ph_nlq, [-1]))
        self_diff = tf.square(yt - tf.reshape(self.lmq, [-1]))
        guide_diff = tf.multiply(tf.reshape(self.ph_guide_weight, [-1]), tf.square(tf.reshape(self.lmq-self.ph_guide_q, [-1])))
        hint_diff = tf.square(tf.reshape(self.lmh - self.ph_hint, [-1]))

        self.l_prob = tf.sigmoid(self.lmq)
        self.b_prob = tf.sigmoid(self.bmq)
        self.b_reward = 2 * self.b_prob - 1
        print 'self.lmq shape:', self.lmq.shape, 'self.l_prob shape:', self.l_prob.shape

        #loss_weights = [0.45, 0.30, 0.25]
        loss_weights = [0.40, 0.10, 0.10]
        self.lite_loss = loss_weights[0]*tf.reduce_mean(self_diff)
        if pd['enable_distill']:
            self.lite_loss = loss_weights[0]*tf.reduce_mean(self_diff) + \
                             loss_weights[1]*tf.reduce_mean(guide_diff) + \
                             loss_weights[2]*tf.reduce_mean(hint_diff)
        if pd['enable_gan']:
            self.lite_loss = loss_weights[0]*tf.reduce_mean(self_diff) + \
                             (loss_weights[1]*tf.reduce_mean(guide_diff) + \
                             loss_weights[2]*tf.reduce_mean(hint_diff)) / ((tf.reduce_mean(self_diff) ** 0.5))
            self.lite_loss += -0.05 * tf.reduce_mean(tf.multiply(tf.log(1 + self.l_prob), self.gan_reward)) / ((tf.reduce_mean(self_diff) ** 0.5))

        vs = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='lite/main')
        vs.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='lite/embedding'))
        self.l_grads = tf.clip_by_global_norm(tf.gradients(self.lite_loss, vs), pd['grad_clip'])[0]
        with tf.variable_scope('opt_lite'):
            optimizer = tf.train.AdamOptimizer(self.l_lr)
            self.lite_opt = optimizer.apply_gradients(zip(self.l_grads, vs), global_step=global_step)
        m_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="lite/main")
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="lite/target")
        self.lite_sync_op = [tf.assign(t, (1.0-alpha)*t + alpha*m) for t, m in zip(t_params, m_params)]
        
    def encode(self, clk_embed, usr_embed):
        global g_training
        q = tf.layers.dropout(usr_embed, rate=pd['dropout'], training=g_training)
        kv = tf.layers.dropout(clk_embed, rate=pd['dropout'], training=g_training)
        for i in range(pd['encoder_layer']):
            with tf.variable_scope('encoder_%d' % (i+1)):
                #self-attention
                enc = multihead_attention(queries = q,
                                          keys = kv,
                                          values = kv,
                                          num_heads = pd['head_num'],
                                          dropout_rate = pd['dropout'],
                                          training = g_training, 
                                          causality = False,
                                          scope='mha')
                #feed forward
                last_dim = q.get_shape().as_list()[-1]
                enc = feed_forward(enc, num_units=[last_dim, last_dim], activation=tf.nn.tanh, scope='ff')
        return enc
    
    def field_interact(self, fields):
        global g_training
        qkv = tf.layers.dropout(fields, rate=pd['dropout'], training=g_training)
        with tf.variable_scope('fi'):
            return multihead_attention(queries = qkv,
                                       keys = qkv,
                                       values = qkv,
                                       num_heads = pd['head_num'],
                                       dropout_rate = pd['dropout'],
                                       training = g_training,
                                       causality = False,
                                       scope='mha')
    
    def build_embedding_layer(self, sub_net, scope, feat_size, feat_dim, need_zdoc=False):
        # 必须设置zero_pad=False，这是因为PartitionedVariable不支持相关操作
        with tf.variable_scope(sub_net, reuse=tf.AUTO_REUSE):
            feat_dict = get_embeddings(feat_size, 
                                       feat_dim,
                                       scope='embedding', 
                                       zero_pad=False, 
                                       partitioner=tf.fixed_size_partitioner(pd['embed_dict_split_part_cnt'], 0))
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                # click sequence postional embedding
                csp_dict = get_embeddings(self.clk_len,
                                          self.n_doc*feat_dim,
                                          scope='click_pos_embedding',
                                          zero_pad=False)
                clk_pos_ids = [[i for i in range(self.clk_len)] for k in range(self.n_batch)]
                pos_embed = tf.nn.embedding_lookup(csp_dict, clk_pos_ids)
                pos_embed = tf.identity(pos_embed, name='pos_embed')
                clk_embed = tf.nn.embedding_lookup_sparse(feat_dict, 
                                                          self.sph_clk_seq, 
                                                          sp_weights=None, 
                                                          partition_strategy='div', 
                                                          combiner='mean')
                clk_embed = tf.reshape(clk_embed, shape=[self.n_batch, self.clk_len, self.n_doc*feat_dim])
                clk_embed = tf.identity(clk_embed, name='clk_embed')
                clk_seq_embed = clk_embed + pos_embed
                #print 'clk_seq_embed.shape:', clk_seq_embed.shape
                user_embed = tf.nn.embedding_lookup_sparse(feat_dict, 
                                                           self.sph_user, 
                                                           sp_weights=None, 
                                                           partition_strategy='div', 
                                                           combiner='mean')
                user_fields = tf.reshape(user_embed, shape=[self.n_batch, self.rnn_len, self.n_user, feat_dim])
                doc_embed = tf.nn.embedding_lookup_sparse(feat_dict, 
                                                          self.sph_doc, 
                                                          sp_weights=None, 
                                                          partition_strategy='div', 
                                                          combiner='mean')
                doc_fields = tf.reshape(doc_embed, shape=[self.n_batch, self.rnn_len, self.n_doc, feat_dim])
                con_embed = tf.nn.embedding_lookup_sparse(feat_dict, 
                                                          self.sph_con, 
                                                          sp_weights=None, 
                                                          partition_strategy='div', 
                                                          combiner='mean')
                con_fields = tf.reshape(con_embed, shape=[self.n_batch, self.rnn_len, self.n_con, feat_dim])
                if need_zdoc:
                    zdoc_embed = tf.nn.embedding_lookup_sparse(feat_dict, 
                                                              self.sph_zdoc, 
                                                              sp_weights=None, 
                                                              partition_strategy='div', 
                                                              combiner='mean')
                    zdoc_fields = tf.reshape(zdoc_embed, shape=[self.n_batch, self.rnn_len, 2, feat_dim])
                    return clk_seq_embed, user_fields, doc_fields, con_fields, zdoc_fields
                return clk_seq_embed, user_fields, doc_fields, con_fields

    def build_lite_network(self, scope):
        global g_training
        clk_seq_embed, user_fields, doc_fields, con_fields = self.build_embedding_layer('lite', scope, pd['feat_size'], self.n_l_dim)
        with tf.variable_scope('lite', reuse=tf.AUTO_REUSE):
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                outputs = []
                state_dim = self.l_rnn+(self.n_user+self.n_doc)*self.n_l_dim
                mlp_dims = [state_dim/2, self.hint_dim]
                inp_dim = (self.n_doc + self.n_con) * self.n_l_dim
                gru = GRU(inp_dim, self.l_rnn, self.rnn_len)
                step_h = tf.zeros(shape=[self.n_batch, self.l_rnn], dtype=tf.float32)
                h_layers = []
                for i in range(self.rnn_len):
                    step_u = tf.identity(user_fields[:,i,:,:], name='user_embed')
                    step_m = self.encode(clk_seq_embed, tf.reshape(step_u, shape=(self.n_batch, 1, self.n_user*self.n_l_dim)))
                    step_m = tf.identity(tf.reshape(step_m, (self.n_batch, -1)), name='memory')
                    step_d = tf.identity(doc_fields[:,i,:,:], name='doc_embed')
                    step_c = tf.identity(con_fields[:,i,:,:], name='con_embed')
                    step_x = tf.reshape(tf.concat([step_d, step_c], axis = 2), (self.n_batch, -1))
                    step_h = tf.identity(gru.step_forward(step_x, step_h), name='gru_h' + str(i))
                    step_s = tf.concat([tf.reshape(step_u, (self.n_batch, -1)),
                                        step_h,
                                        step_m], axis = 1)
                    step_fc = step_s
                    for k in range(len(mlp_dims)):
                        step_fc = tf.layers.dense(step_fc, mlp_dims[k], activation=tf.nn.tanh, name='stepwise_fc'+str(k), reuse=tf.AUTO_REUSE)
                        step_fc = tf.layers.dropout(step_fc, rate=pd['dropout'], training=g_training)
                    l_h_layer = step_fc
                    qt = tf.layers.dense(l_h_layer, 1, name='qt', reuse=tf.AUTO_REUSE)
                    qt = tf.identity(qt, name='stepwise_q' + str(i))
                    outputs.append(qt)
                    h_layers.append(l_h_layer)
                hint = tf.reshape(tf.concat(h_layers, axis=1), (self.n_batch, self.rnn_len, -1))
                print 'hint.shape:', hint.shape
                q = tf.concat(outputs, axis=1)
                print 'q.shape:', q.shape
                return hint, q
    
    def build_booster_network(self, scope):
        global g_training
        if pd['enable_future']:
            clk_seq_embed, user_fields, doc_fields, con_fields, zdoc_fields = self.build_embedding_layer('booster', scope, pd['feat_size'], self.n_b_dim, pd['enable_future'])
        else:
            clk_seq_embed, user_fields, doc_fields, con_fields = self.build_embedding_layer('booster', scope, pd['feat_size'], self.n_b_dim)
        with tf.variable_scope('booster', reuse=tf.AUTO_REUSE):
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                # rnn seq embedding
                seq_fields = tf.concat([tf.reshape(doc_fields, (-1, self.n_doc, self.n_b_dim)),
                                        tf.reshape(con_fields, (-1, self.n_con, self.n_b_dim))], axis=2)
                print 'seq shape:', seq_fields.shape
                if pd['enable_future']:
                    seq_fields = tf.concat([tf.concat([tf.reshape(doc_fields, (-1, self.n_doc, self.n_b_dim)),
                                                       tf.reshape(zdoc_fields, (-1, 2, self.n_b_dim))], axis=1),
                                            tf.concat([tf.reshape(con_fields, (-1, self.n_con, self.n_b_dim)),
                                                       tf.reshape(zdoc_fields, (-1, 2, self.n_b_dim))], axis=1)], axis=2)
                print 'seq shape:', seq_fields.shape
                inter = self.field_interact(seq_fields)
                print 'interact shape:', inter.shape
                seq_embed = tf.reshape(inter, (self.n_batch, self.rnn_len, -1))
                print 'seq_embed shape:', seq_embed.shape
                inp_dim = seq_embed.get_shape().as_list()[-1]
                print 'inp_dim:', inp_dim, 'self.b_rnn:', self.b_rnn, 'self.rnn_len:', self.rnn_len
                gru = GRU(inp_dim, self.b_rnn, self.rnn_len)
                seq_embed = gru.do_infer(seq_embed)
                print 'seq_embed.shape:', seq_embed.shape
                # memory embedding
                mem_embed = self.encode(clk_seq_embed, tf.reshape(user_fields, (self.n_batch, self.rnn_len, -1)))
                print 'mem_embed.shape:', mem_embed.shape
                # state embedding
                state_emebd = tf.concat([seq_embed, 
                                         mem_embed, 
                                         tf.reshape(user_fields, (self.n_batch, self.rnn_len, -1))],
                                         axis = 2)
                print 'state_embed.shape:', state_emebd.shape
                state_dim = state_emebd.get_shape().as_list()[-1]
                mlp_dims = [state_dim/2, state_dim/4, self.hint_dim]
                fc = state_emebd
                for i in range(len(mlp_dims)):
                    fc = tf.layers.dense(fc, mlp_dims[i], activation = tf.nn.tanh)
                    fc = tf.layers.dropout(fc, rate=pd['dropout'], training=g_training)
                b_h_layer = fc
                print 'hint_layer.shape:', b_h_layer.shape
                q = tf.reshape(tf.layers.dense(b_h_layer, 1), (self.n_batch, self.rnn_len))
                print 'q.shape:', q.shape
                return b_h_layer, q

    #call for temporal-diffenence learning
    def target_bq(self, sess, ph_dict):
        return sess.run(self.btq, feed_dict={self.sph_clk_seq   :   ph_dict['clk_seq'],
                                            self.sph_user       :   ph_dict['user'],
                                            self.sph_doc        :   ph_dict['doc'],
                                            self.sph_con        :   ph_dict['con'],
                                            self.sph_zdoc       :   ph_dict['zdoc']})
    #call for calc normalized guide weight and q
    def guide_wnq(self, sess, ph_dict):
        tq = self.target_bq(sess, ph_dict)
        nq = np.append(tq[:,1:], np.array([[0] for i in range(self.n_batch)], dtype=np.float32), 1)
        return sess.run([self.guide_weights, self.bmq],
                            feed_dict={self.sph_clk_seq   :   ph_dict['clk_seq'],
                                       self.ph_nbq        :   nq,
                                       self.ph_reward     :   ph_dict['reward'],
                                       self.sph_user      :   ph_dict['user'],
                                       self.sph_doc       :   ph_dict['doc'],
                                       self.sph_con       :   ph_dict['con'],
                                       self.sph_zdoc      :   ph_dict['zdoc']})
    #call for calc booter hint layer
    def bhint(self, sess, ph_dict):
        return sess.run(self.bmh, feed_dict={self.sph_clk_seq   :   ph_dict['clk_seq'],
                                            self.sph_user       :   ph_dict['user'],
                                            self.sph_doc        :   ph_dict['doc'],
                                            self.sph_con        :   ph_dict['con'],
                                            self.sph_zdoc       :   ph_dict['zdoc']})
    #call for temporal-diffenence learning
    def target_lq(self, sess, ph_dict):
        return sess.run(self.ltq, feed_dict={self.sph_clk_seq   :   ph_dict['clk_seq'],
                                            self.sph_user       :   ph_dict['user'],
                                            self.sph_doc        :   ph_dict['doc'],
                                            self.sph_con        :   ph_dict['con']})
    #call for evaluating booster networks
    def main_bq(self, sess, ph_dict):
        return sess.run(self.bmq, feed_dict={self.sph_clk_seq   :   ph_dict['clk_seq'],
                                            self.sph_user       :   ph_dict['user'],
                                            self.sph_doc        :   ph_dict['doc'],
                                            self.sph_con        :   ph_dict['con'],
                                            self.sph_zdoc       :   ph_dict['zdoc']})
    #call for evaluating booster networks
    def main_lq(self, sess, ph_dict):
        return sess.run(self.lmq, feed_dict={self.sph_clk_seq   :   ph_dict['clk_seq'],
                                            self.sph_user       :   ph_dict['user'],
                                            self.sph_doc        :   ph_dict['doc'],
                                            self.sph_con        :   ph_dict['con']})

    def get_lprob(self, sess, ph_dict):
        return sess.run(self.l_prob, feed_dict={self.sph_clk_seq   :   ph_dict['clk_seq'],
                                            self.sph_user       :   ph_dict['user'],
                                            self.sph_doc        :   ph_dict['doc'],
                                            self.sph_con        :   ph_dict['con']})
    def get_bprob(self, sess, ph_dict):
        return sess.run(self.b_prob, feed_dict={self.sph_clk_seq   :   ph_dict['clk_seq'],
                                            self.sph_user       :   ph_dict['user'],
                                            self.sph_doc        :   ph_dict['doc'],
                                            self.sph_con        :   ph_dict['con'],
                                            self.sph_zdoc       :   ph_dict['zdoc']})
    def get_breward(self, sess, ph_dict):
        return sess.run(self.b_reward, feed_dict={self.sph_clk_seq   :   ph_dict['clk_seq'],
                                            self.sph_user       :   ph_dict['user'],
                                            self.sph_doc        :   ph_dict['doc'],
                                            self.sph_con        :   ph_dict['con'],
                                            self.sph_zdoc       :   ph_dict['zdoc']})
    #call for learning from data
    def booster_learn(self, sess, ph_dict):
        tq = self.target_bq(sess, ph_dict)
        nq = np.append(tq[:,1:], np.array([[0] for i in range(self.n_batch)], dtype=np.float32), 1)
        loss, _ = sess.run([self.booster_loss, self.booster_opt], feed_dict={self.ph_nbq        :   nq,
                                                                             self.ph_reward     :   ph_dict['reward'],
                                                                             self.sph_clk_seq   :   ph_dict['clk_seq'],
                                                                             self.sph_user      :   ph_dict['user'],
                                                                             self.sph_doc       :   ph_dict['doc'],
                                                                             self.sph_con       :   ph_dict['con'],
                                                                             self.sph_zdoc      :   ph_dict['zdoc']})
        self.b_loss += loss
        global g_batch_counter
        if g_batch_counter % 1 == 0:
            print sess.run(self.global_step), ' ---Booster Network Loss: ', self.b_loss / g_batch_counter

    #call for learning from data
    def lite_learn(self, sess, ph_dict):
        old_ph = copy.deepcopy(ph_dict)
        if pd['enable_gan']:
            prob = self.get_lprob(sess, ph_dict).reshape(-1)
            now_sum = sum(prob)
            if now_sum <= 0:
                now_sum = 1
            prob = prob / now_sum
            alpha = 0.9
            ll = list(prob.shape)[0]
            new_prob = prob * alpha + (1 - alpha) / ll
            new_prob = new_prob / np.sum(new_prob)
            sample = np.random.choice(ll, ll, p=new_prob, replace=True)
            #self.take_ph_dict(ph_dict, sample)
        tq = self.target_lq(sess, ph_dict)
        nq = np.append(tq[:,1:], np.array([[0] for i in range(self.n_batch)], dtype=np.float32), 1)
        guide_w, guide_q = self.guide_wnq(sess, ph_dict)
        hint = self.bhint(sess, ph_dict)
        gan_reward = self.get_breward(sess, ph_dict)
        #print gan_reward
        #print gan_reward.shape
        loss, _ = sess.run([self.lite_loss, self.lite_opt], feed_dict={self.ph_nlq          :   nq,
                                                                       self.ph_hint         :   hint,
                                                                       self.ph_guide_weight :   guide_w,
                                                                       self.ph_guide_q      :   guide_q,
                                                                       self.gan_reward      :   gan_reward,
                                                                       self.ph_reward       :   ph_dict['reward'],
                                                                       self.sph_clk_seq     :   ph_dict['clk_seq'],
                                                                       self.sph_user        :   ph_dict['user'],
                                                                       self.sph_doc         :   ph_dict['doc'],
                                                                       self.sph_con         :   ph_dict['con']})
        self.l_loss += loss
        global g_batch_counter
        if g_batch_counter % 1 == 0:
            print sess.run(self.global_step), ' ---Lite Network Loss: ', self.l_loss / g_batch_counter
        ph_dict = copy.deepcopy(old_ph)

    def take_array(self, arr, take_list):
        #arr = arr.reshape((pd['batch_size'], -1))
        if isinstance(arr, list):
            tmp = [arr[i] for i in take_list]
            return tmp
        l = pd['batch_size']*pd['clk_seq_len']
        shape = arr.dense_shape
        left = len(arr.indices) / l
        #shape[0] = shape[0] / 2
        #print shape
        #print arr.indices[0]
        #print arr.values[0]
        ind = []
        val = []
        for i in take_list:
            for j in range(left):
                ind.append(arr.indices[i * left + j])
                val.append(arr.values[i * left + j])
        return tf.SparseTensorValue(ind, val, shape)

    def take_ph_dict(self, ph_dict, take_list):
        take_list = range(len(ph_dict['reward']))
        #ph_dict['reward'] = [ph_dict['reward'][i] for i in take_list]
        ph_dict['reward'] = self.take_array(ph_dict['reward'], take_list)
        #print dir(ph_dict['clk_seq'])
        print ph_dict['clk_seq'].dense_shape
        print ph_dict['clk_seq'].indices[0]
        print len(ph_dict['clk_seq'].indices)
        print ph_dict['clk_seq'].values[0]
        ph_dict['clk_seq'] = self.take_array(ph_dict['clk_seq'], take_list)
        print ph_dict['clk_seq'].dense_shape
        print ph_dict['clk_seq'].indices[0]
        print len(ph_dict['clk_seq'].indices)
        print ph_dict['clk_seq'].values[0]
        ph_dict['user'] = self.take_array(ph_dict['user'], take_list)
        ph_dict['doc'] = self.take_array(ph_dict['doc'], take_list)
        ph_dict['con'] = self.take_array(ph_dict['con'], take_list)
        ph_dict['zdoc'] = self.take_array(ph_dict['zdoc'], take_list)

#####   global variables for predict samples  ######
q_pred_op = None
p_pred_op = None

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(max(min(-x, 1e4), -1e4)))

def handle(sess, net, sess_data):
    def gen_sparse_tensor(fs):
        kk, vv = [], []
        for i in range(len(fs)):
            ff = fs[i]
            assert(isinstance(ff, set))
            ff = list(ff)
            for k in range(len(ff)):
                kk.append(np.array([i, k], dtype=np.int32))
                vv.append(ff[k])
        return tf.SparseTensorValue(kk, vv, [len(fs), pd['feat_size']])
    
    global g_rb
    g_rb.save(sess_data)
    while g_rb.has_batch():
        clk, user, doc, con, rwd, rtn, zdoc = g_rb.next_batch()
        clk = np.array(clk).reshape(pd['batch_size']*pd['clk_seq_len']*pd['doc_field_num'])
        phd = {}
        phd['clk_seq'] = gen_sparse_tensor(clk)
        #print np.array(user).shape
        user = np.array(user).reshape(pd['batch_size']*pd['rnn_seq_len']*pd['user_field_num'])
        phd['user'] = gen_sparse_tensor(user)
        doc = np.array(doc).reshape(pd['batch_size']*pd['rnn_seq_len']*pd['doc_field_num'])
        phd['doc'] = gen_sparse_tensor(doc)
        con = np.array(con).reshape(pd['batch_size']*pd['rnn_seq_len']*pd['con_field_num'])
        phd['con'] = gen_sparse_tensor(con)
        zdoc = np.array(zdoc).reshape(pd['batch_size']*pd['rnn_seq_len']*2)
        phd['zdoc'] = gen_sparse_tensor(zdoc)
        phd['reward'] = rwd
        global g_batch_counter, g_training
        print datetime.datetime.now(), 'start to handle batch', g_batch_counter
        g_batch_counter += 1
        if g_training:
            net.booster_learn(sess, phd)
            net.lite_learn(sess, phd)
            #net.hint_learn(sess, phd)
            if g_batch_counter % pd['double_networks_sync_freq'] == 0:
                print 'Run soft replacement for main networks and target networks...'
                sess.run(net.booster_sync_op)
                sess.run(net.lite_sync_op)
        else:
            qout = net.main_bq(sess, phd).reshape([-1])
            global g_working_mode
            for i in range(len(rtn)):
                if 'local_predict' == g_working_mode:
                    print('%s %s' % (rwd[i], sigmoid(qout[i])))
                else:
                    q_pred_op.write('%s %s\n' % (rwd[i], sigmoid(qout[i])))
            pout = net.main_lq(sess, phd).reshape([-1])
            for i in range(len(rtn)):
                if 'local_predict' == g_working_mode:
                    print('%s %s' % (rwd[i], sigmoid(pout[i])))
                else:
                    p_pred_op.write('%s %s\n' % (rwd[i], sigmoid(pout[i])))
        print datetime.datetime.now(), 'batch finish, ', g_rb.dump()
    
def do_train(sess, net):
    print 'start do train...'
    worker_hosts = FLAGS.worker_hosts.split(',')
    print 'HADOOP_BIN: ', os.environ['HADOOP_BIN']
    print 'fea_path: ', os.environ['fea_path']
    hdp_cmd = os.environ['HADOOP_BIN'] + ' fs' + ' -ls ' + os.environ['fea_path']
    print "hadoop command: ", hdp_cmd
    filelist = subprocess.Popen(hdp_cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    print 'filelist: ', filelist
    filelist = map(lambda x: x.split()[-1], filter(lambda x: 'part' in x, filelist.strip().split('\n')))
    train_sess_cnt = 0
    train_file_idx = 0
    global g_training
    for fname in filelist:
        dit = int(hashlib.md5(fname).hexdigest(), 16) % len(worker_hosts)
        dit = int(fname[-8:-3]) % len(worker_hosts)
        if int(dit) != int(FLAGS.task_index):
            continue
        train_file_idx += 1
        print datetime.datetime.now(), fname
        sys.stdout.flush()
        train_ratio = 7
        #if len(filelist) >= 200 and ((g_training and train_file_idx % 10 >= train_ratio) or (not g_training and train_file_idx % 10 < train_ratio)):
        if (g_training and train_file_idx % 3 != 0) or (not g_training and train_file_idx % 3 != 1):
            print 'to speed up, skip...'
            continue
        part = fname.split('/')[-1][:-3]
        os.system('rm train_data.txt')
        os.system('timeout 1800 %s fs -text %s > train_data.txt' % (os.environ['HADOOP_BIN'], fname))
        if not os.path.exists('train_data.txt'):
            continue
        global q_pred_op, p_pred_op
        if not g_training:
            print 'create new file pred_c.txt'
            q_pred_op = open('pred_c.txt', 'w')
            print 'create new file pred_ac.txt'
            p_pred_op = open('pred_ac.txt', 'w')
        for data in read_data('train_data.txt', pd['clk_seq_len'], pd['user_field_num'], pd['doc_field_num'], pd['con_field_num'], pd['feat_prime']):
            handle(sess, net, data)
            train_sess_cnt += 1
            if g_training and train_sess_cnt % 1000 == 0:
                print '------Global Step: ', sess.run(net.global_step)
                print '------Train Session Count: ', train_sess_cnt
        if not g_training:
            print 'close new file pred.txt'
            q_pred_op.close()
            p_pred_op.close()
            os.system('%s fs -put pred_c.txt %s/%s' % (os.environ['HADOOP_BIN'], os.environ['c_pred_path'], part))
            os.system('%s fs -put pred_ac.txt %s/%s' % (os.environ['HADOOP_BIN'], os.environ['ac_pred_path'], part))
            os.system('rm pred_c.txt')
            os.system('rm pred_ac.txt')

def distributed_run():
    is_chief = (FLAGS.task_index == 0)
    worker_hosts = FLAGS.worker_hosts.split(',')
    ps_hosts = FLAGS.ps_hosts.split(',')
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    if FLAGS.job_name == "ps":
        if not os.path.exists(FLAGS.model_path):
            os.system('mkdir %s' % FLAGS.model_path)
        server.join()
    elif FLAGS.job_name == "worker":
        global g_training
        if is_chief and g_training == False:
            os.system('%s fs -rmr %s' % (os.environ['HADOOP_BIN'], os.environ['c_pred_path']))
            os.system('%s fs -mkdir %s' % (os.environ['HADOOP_BIN'], os.environ['c_pred_path']))
            os.system('%s fs -rmr %s' % (os.environ['HADOOP_BIN'], os.environ['ac_pred_path']))
            os.system('%s fs -mkdir %s' % (os.environ['HADOOP_BIN'], os.environ['ac_pred_path']))
        num_ps_tasks = len(ps_hosts)
        ps_stg = tf.contrib.training.GreedyLoadBalancingStrategy(num_ps_tasks, tf.contrib.training.byte_size_load_fn)
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index, 
                                                      ps_device="/job:ps/cpu:0", 
                                                      cluster=cluster, 
                                                      ps_strategy=ps_stg)):
            #build networks
            global_step = tf.train.get_or_create_global_step()
            net = FGAN(global_step)
        ckpt_dir = 'hdfs://searchteamcluster%s' % os.environ['all_model_path']
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=is_chief,
                                               checkpoint_dir=ckpt_dir,
                                               save_checkpoint_secs=1800,
                                               save_checkpoint_steps=None,
                                               save_summaries_steps=None,
                                               save_summaries_secs=None) as sess:
            print 'start', datetime.datetime.now(), FLAGS.job_name, FLAGS.task_index
            sys.stdout.flush()
            for _ in range(pd['num_epochs']):
                do_train(sess, net)
            print 'finish train/predict', datetime.datetime.now(), FLAGS.job_name, FLAGS.task_index
            print '===='
            #dump model
            if is_chief and (not g_training):
                vs = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='lite/main')
                vs.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='lite/embedding'))
                ns = [v.name for v in vs]
                var_info = []
                part_var_info = []
                for n in ns:
                    w = sess.run(n)
                    if 'embedding/w/part' not in n:
                        var_info.append([n, w.shape, np.ndarray.tolist(w)])
                    else:
                        part_var_info.append(w)
                fe = np.concatenate(part_var_info, 0)
                var_info.append(['lite/embedding/w:0', fe.shape, np.ndarray.tolist(fe)])
                for vi in var_info:
                    print vi[0], vi[1]
                with open('weight_model.json', 'w') as op:
                    op.write(json.dumps(var_info))
                os.system('%s fs -rmr %s' % (os.environ['HADOOP_BIN'], os.environ['json_model_path']))
                os.system('%s fs -mkdir %s' % (os.environ['HADOOP_BIN'], os.environ['json_model_path']))
                os.system('%s fs -put weight_model.json %s' % (os.environ['HADOOP_BIN'], os.environ['json_model_path']))
                os.system('rm weight_model.json')
            print 'start to exit process...'
            sys.stdout.flush()
            os._exit(0)

def local_run():
    global_step = tf.train.get_or_create_global_step()
    sess = tf.Session()
    net = FGAN(global_step)
    vs = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='')
    #for v in vs:
    #    print v.name, v.shape
    #print '=====bye'
    #sys.exit(0)
    saver = tf.train.Saver(max_to_keep=1)
    g_init_op = tf.global_variables_initializer()
    if os.path.exists('./ckpt'):
        model_file = tf.train.latest_checkpoint('ckpt/')
        saver.restore(sess, model_file)
    else:
        sess.run(g_init_op)
        os.system('mkdir ckpt')
    print '>>>local model...'
    pull_cnt = 0
    for data in read_data('sample.in', pd['clk_seq_len'], pd['user_field_num'], pd['doc_field_num'], pd['con_field_num'], pd['feat_prime']):
        handle(sess, net, data)
        pull_cnt += 1
    saver.save(sess, 'ckpt/kdq.ckpt')

def local_dump_model():
    global_step= tf.train.get_or_create_global_step()
    sess = tf.Session()
    #build networks
    net = FGAN(global_step)
    saver = tf.train.Saver(max_to_keep=1)
    if os.path.exists('./ckpt'):
        model_file = tf.train.latest_checkpoint('ckpt/')
        saver.restore(sess, model_file)
    else:
        print 'no model exists!'
    vs = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='lite/main')
    vs.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='lite/embedding'))
    ns = [v.name for v in vs]
    var_info = []
    part_var_info = []
    for n in ns:
        w = sess.run(n)
        if 'embedding/w/part' not in n:
            var_info.append([n, w.shape, np.ndarray.tolist(w)])
        else:
            print n, w.shape
            part_var_info.append(w)
    fe = np.concatenate(part_var_info, 0)
    var_info.append(['lite/embedding/w:0', fe.shape, np.ndarray.tolist(fe)])
    for vi in var_info:
        print vi[0], vi[1]

if __name__ == '__main__':
    assert(pd['clk_seq_len'] == pd['rnn_seq_len'])
    g_working_mode = os.environ['working_mode']
    
    if 'enable_gan' in os.environ:
        pd['enable_gan'] = os.environ['enable_gan']
    if 'enable_distill' in os.environ:
        pd['enable_distill'] = os.environ['enable_distill']
    if 'enable_future' in os.environ:
        pd['enable_future'] = os.environ['enable_future']

    g_rb = RB(pd['batch_size'], pd['clk_seq_len'], pd['rnn_seq_len'], pd['user_field_num'], pd['doc_field_num'], pd['con_field_num'])
    commander = {
        'local_train' : local_run,
        'local_predict' : local_run,
        'local_dump_model' : local_dump_model,
        'distributed_train' : distributed_run,
        'distributed_predict' : distributed_run,
    }
    if g_working_mode not in commander:
        print 'your working mode(%s) not recognized!!!' % g_working_mode
        sys.exit(1)
    g_training = (g_working_mode  == 'local_train' or g_working_mode == 'distributed_train')
    print '>>> working_model:', g_working_mode
    print '>>> is_training:', g_training
    commander[g_working_mode]()
