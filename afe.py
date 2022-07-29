#!encoding=utf-8
"""
    A Peep into the Future: Adversarial Future Encoding in Recommendation
    @2022-07-27
    by Modric Zhang
"""
import os
import sys
import copy
import datetime
import numpy as np
import tensorflow.compat.v1 as tf
from layer_util import *
from data_reader import DataReader
from hyper_param import param_dict as pd
from replay_buffer import RB

tf.disable_eager_execution()
tf.logging.set_verbosity(tf.logging.ERROR)

g_working_mode = 'local_train'
g_training = False
g_batch_counter = 0
# replay buffer
g_rb = None
# data reader
g_dr = DataReader(pd['batch_size'])


class FutureGAN(object):
    def __init__(self, global_step):
        self.global_step = global_step
        # cumulative loss
        self.b_loss, self.l_loss, self.h_loss = 0.0, 0.0, 0.0
        # network parameter
        self.n_batch, self.clk_len, self.rnn_len = pd['batch_size'], pd['clk_seq_len'], pd['rnn_seq_len']
        self.n_user, self.n_doc, self.n_con, self.n_future = pd['user_field_num'], pd['doc_field_num'], pd[
            'con_field_num'], pd['future_field_num']
        self.n_b_dim, self.n_l_dim, self.hint_dim = pd['booster_feat_dim'], pd['lite_feat_dim'], pd['hint_dim']
        self.b_rnn, self.l_rnn = pd['booster_rnn_dim'], pd['lite_rnn_dim']
        self.gamma = pd['rl_gamma']
        self.b_lr, self.l_lr = pd['booster_lr'], pd['lite_lr']
        # placeholder
        self.sph_clk_seq = tf.sparse_placeholder(tf.int32, name='sph_clk_seq')
        self.sph_user = tf.sparse_placeholder(tf.int32, name='sph_user')
        self.sph_doc = tf.sparse_placeholder(tf.int32, name='sph_doc')
        self.sph_future = tf.sparse_placeholder(tf.int32, name='sph_future')
        self.sph_con = tf.sparse_placeholder(tf.int32, name='sph_con')
        self.ph_reward = tf.placeholder(tf.float32, shape=(self.n_batch * self.rnn_len), name='ph_reward')
        self.ph_nbq = tf.placeholder(tf.float32, shape=(self.n_batch, self.rnn_len), name='ph_nbq')
        self.ph_nlq = tf.placeholder(tf.float32, shape=(self.n_batch, self.rnn_len), name='ph_nlq')
        self.ph_guide_weight = tf.placeholder(tf.float32, shape=(self.n_batch * self.rnn_len), name='ph_guide_weight')
        self.ph_guide_q = tf.placeholder(tf.float32, shape=(self.n_batch, self.rnn_len), name='ph_guide_q')
        self.ph_hint = tf.placeholder(tf.float32, shape=(self.n_batch, self.rnn_len, self.hint_dim), name='ph_hint')
        self.gan_reward = tf.placeholder(tf.float32, name='gan_reward')
        # booster loss
        print('\n======\nbuilding booster main q network ...')
        self.bmh, self.bmq = self.build_booster_network('main')
        print('\n======\nbuilding booster target q network ...')
        _, self.btq = self.build_booster_network('target')
        yt = tf.reshape(self.ph_reward, [-1]) + tf.scalar_mul(tf.constant(self.gamma), tf.reshape(self.ph_nbq, [-1]))
        diff = yt - tf.reshape(self.bmq, [-1])
        self.booster_loss = tf.reduce_mean(tf.square(diff))
        # guide weight
        ee = tf.square(diff)
        maxe = tf.reduce_max(tf.reshape(ee, [-1]))
        mine = tf.reduce_min(tf.reshape(ee, [-1]))
        self.guide_weights = 1.0 - (ee - mine) / (maxe - mine + 1e-3)

        vs = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='booster/main')
        vs.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='booster/embedding'))
        self.b_grads = tf.clip_by_global_norm(tf.gradients(self.booster_loss, vs), pd['grad_clip'])[0]
        with tf.variable_scope('opt_booster'):
            optimizer = tf.train.AdamOptimizer(self.b_lr)
            self.booster_opt = optimizer.apply_gradients(zip(self.b_grads, vs), global_step=global_step)
        m_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="booster/main")
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="booster/target")
        alpha = pd['double_networks_sync_step']
        self.booster_sync_op = [tf.assign(t, (1.0 - alpha) * t + alpha * m) for t, m in zip(t_params, m_params)]

        # lite loss
        print('\n======\nbuilding lite main q network ...')
        self.lmh, self.lmq = self.build_lite_network('main')
        print('\n======\nbuilding lite target q network ...')
        _, self.ltq = self.build_lite_network('target')
        yt = tf.reshape(self.ph_reward, [-1]) + tf.scalar_mul(tf.constant(self.gamma), tf.reshape(self.ph_nlq, [-1]))
        self_diff = tf.square(yt - tf.reshape(self.lmq, [-1]))
        guide_diff = tf.multiply(tf.reshape(self.ph_guide_weight, [-1]),
                                 tf.square(tf.reshape(self.lmq - self.ph_guide_q, [-1])))
        hint_diff = tf.square(tf.reshape(self.lmh - self.ph_hint, [-1]))

        self.l_prob = tf.sigmoid(self.lmq)
        self.b_prob = tf.sigmoid(self.bmq)
        self.b_reward = 2 * self.b_prob - 1
        print('self.lmq shape:', self.lmq.shape, 'self.l_prob shape:', self.l_prob.shape)

        loss_weights = [0.40, 0.10, 0.10]
        self.lite_loss = loss_weights[0] * tf.reduce_mean(self_diff)
        if pd['enable_distill']:
            self.lite_loss = loss_weights[0] * tf.reduce_mean(self_diff) + \
                             loss_weights[1] * tf.reduce_mean(guide_diff) + \
                             loss_weights[2] * tf.reduce_mean(hint_diff)
        if pd['enable_gan']:
            self.lite_loss = loss_weights[0] * tf.reduce_mean(self_diff) + \
                             (loss_weights[1] * tf.reduce_mean(guide_diff) + \
                              loss_weights[2] * tf.reduce_mean(hint_diff)) / ((tf.reduce_mean(self_diff) ** 0.5))
            self.lite_loss += -0.05 * tf.reduce_mean(tf.multiply(tf.log(1 + self.l_prob), self.gan_reward)) / (
                (tf.reduce_mean(self_diff) ** 0.5))

        vs = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='lite/main')
        vs.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='lite/embedding'))
        self.l_grads = tf.clip_by_global_norm(tf.gradients(self.lite_loss, vs), pd['grad_clip'])[0]
        with tf.variable_scope('opt_lite'):
            optimizer = tf.train.AdamOptimizer(self.l_lr)
            self.lite_opt = optimizer.apply_gradients(zip(self.l_grads, vs), global_step=global_step)
        m_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="lite/main")
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="lite/target")
        self.lite_sync_op = [tf.assign(t, (1.0 - alpha) * t + alpha * m) for t, m in zip(t_params, m_params)]

    def encode(self, clk_embed, usr_embed):
        global g_training
        q = tf.layers.dropout(usr_embed, rate=pd['dropout'], training=g_training)
        kv = tf.layers.dropout(clk_embed, rate=pd['dropout'], training=g_training)
        for i in range(pd['encoder_layer']):
            with tf.variable_scope('encoder_%d' % (i + 1)):
                # self-attention
                enc = multihead_attention(queries=q,
                                          keys=kv,
                                          values=kv,
                                          num_heads=pd['head_num'],
                                          dropout_rate=pd['dropout'],
                                          training=g_training,
                                          causality=False,
                                          scope='mha')
                # feed forward
                last_dim = q.get_shape().as_list()[-1]
                enc = feed_forward(enc, num_units=[last_dim, last_dim], activation=tf.nn.tanh, scope='ff')
        return enc

    def field_interact(self, fields):
        global g_training
        qkv = tf.layers.dropout(fields, rate=pd['dropout'], training=g_training)
        with tf.variable_scope('fi'):
            return multihead_attention(queries=qkv,
                                       keys=qkv,
                                       values=qkv,
                                       num_heads=pd['head_num'],
                                       dropout_rate=pd['dropout'],
                                       training=g_training,
                                       causality=False,
                                       scope='mha')

    def build_embedding_layer(self, sub_net, scope, feat_dim, has_future=False):
        with tf.variable_scope(sub_net, reuse=tf.AUTO_REUSE):
            feat_dict = get_embeddings(g_dr.unique_feature_num(),
                                       feat_dim,
                                       scope='embedding',
                                       zero_pad=False)
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                # position embedding for click docs
                csp_dict = get_embeddings(self.clk_len,
                                          self.n_doc * feat_dim,
                                          scope='click_pos_embedding',
                                          zero_pad=False)
                clk_pos_ids = [[i for i in range(self.clk_len)] for k in range(self.n_batch)]
                pos_embed = tf.nn.embedding_lookup(csp_dict, clk_pos_ids)
                clk_embed = tf.nn.embedding_lookup_sparse(feat_dict,
                                                          self.sph_clk_seq,
                                                          sp_weights=None,
                                                          partition_strategy='div',
                                                          combiner='mean')
                clk_embed = tf.reshape(clk_embed, shape=[self.n_batch, self.clk_len, self.n_doc * feat_dim])
                clk_seq_embed = clk_embed + pos_embed
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
                if has_future:
                    future_embed = tf.nn.embedding_lookup_sparse(feat_dict,
                                                                 self.sph_future,
                                                                 sp_weights=None,
                                                                 partition_strategy='div',
                                                                 combiner='mean')
                    future_fields = tf.reshape(future_embed,
                                               shape=[self.n_batch, self.rnn_len, self.n_future, feat_dim])
                    return clk_seq_embed, user_fields, doc_fields, con_fields, future_fields
                return clk_seq_embed, user_fields, doc_fields, con_fields

    def build_lite_network(self, scope):
        global g_training
        clk_embed, u_fields, d_fields, c_fields = self.build_embedding_layer('lite', scope, self.n_l_dim)
        with tf.variable_scope('lite', reuse=tf.AUTO_REUSE):
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                # rnn seq embedding
                seq_fields = tf.concat([tf.reshape(d_fields, (-1, self.n_doc, self.n_l_dim)),
                                        tf.reshape(c_fields, (-1, self.n_con, self.n_l_dim))], axis=2)
                print('seq shape:', seq_fields.shape)
                inter = self.field_interact(seq_fields)
                print('interact shape:', inter.shape)
                seq_embed = tf.reshape(inter, (self.n_batch, self.rnn_len, -1))
                print('seq_embed shape:', seq_embed.shape)
                inp_dim = seq_embed.get_shape().as_list()[-1]
                print('inp_dim:', inp_dim, 'self.l_rnn:', self.l_rnn, 'self.rnn_len:', self.rnn_len)
                gru = tf.nn.rnn_cell.GRUCell(self.l_rnn)
                drop = tf.nn.rnn_cell.DropoutWrapper(gru, output_keep_prob=1.0 - pd['dropout'] if g_training else 1.)
                cell = tf.nn.rnn_cell.MultiRNNCell([drop for _ in range(pd['rnn_layer'])])
                rnn_out, _ = tf.nn.dynamic_rnn(cell=cell, dtype=tf.float32, inputs=seq_embed,
                                               time_major=False)
                print('rnn_output.shape:', rnn_out.shape)
                # memory embedding
                mem_embed = self.encode(clk_embed, tf.reshape(u_fields, (self.n_batch, self.rnn_len, -1)))
                print('mem_embed.shape:', mem_embed.shape)
                # state embedding
                state_embed = tf.concat([rnn_out,
                                         mem_embed,
                                         tf.reshape(u_fields, (self.n_batch, self.rnn_len, -1))],
                                        axis=2)
                print('state_embed.shape:', state_embed.shape)
                state_dim = state_embed.get_shape().as_list()[-1]
                mlp_dims = [state_dim / 2, self.hint_dim]
                fc = state_embed
                for i in range(len(mlp_dims)):
                    fc = tf.layers.dense(fc, mlp_dims[i], activation=tf.nn.tanh)
                    fc = tf.layers.dropout(fc, rate=pd['dropout'], training=g_training)
                l_h_layer = fc
                print('hint_layer.shape:', l_h_layer.shape)
                q = tf.reshape(tf.layers.dense(l_h_layer, 1), (self.n_batch, self.rnn_len))
                print('q.shape:', q.shape)
                return l_h_layer, q

    def build_booster_network(self, scope):
        if pd['enable_future']:
            clk_embed, u_fields, d_fields, c_fields, f_fields = self.build_embedding_layer('booster',
                                                                                           scope,
                                                                                           self.n_b_dim,
                                                                                           pd['enable_future'])
        else:
            clk_embed, u_fields, d_fields, c_fields = self.build_embedding_layer('booster', scope, self.n_b_dim)
        with tf.variable_scope('booster', reuse=tf.AUTO_REUSE):
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                # rnn seq embedding
                seq_fields = tf.concat([tf.reshape(d_fields, (-1, self.n_doc, self.n_b_dim)),
                                        tf.reshape(c_fields, (-1, self.n_con, self.n_b_dim))], axis=2)
                print('seq shape:', seq_fields.shape)
                if pd['enable_future']:
                    seq_fields = tf.concat([tf.concat([tf.reshape(d_fields, (-1, self.n_doc, self.n_b_dim)),
                                                       tf.reshape(f_fields, (-1, self.n_future, self.n_b_dim))],
                                                      axis=1),
                                            tf.concat([tf.reshape(c_fields, (-1, self.n_con, self.n_b_dim)),
                                                       tf.reshape(f_fields, (-1, self.n_future, self.n_b_dim))],
                                                      axis=1)],
                                           axis=2)
                print('seq shape:', seq_fields.shape)
                inter = self.field_interact(seq_fields)
                print('interact shape:', inter.shape)
                seq_embed = tf.reshape(inter, (self.n_batch, self.rnn_len, -1))
                print('seq_embed shape:', seq_embed.shape)
                inp_dim = seq_embed.get_shape().as_list()[-1]
                print('inp_dim:', inp_dim, 'self.b_rnn:', self.b_rnn, 'self.rnn_len:', self.rnn_len)
                gru = tf.nn.rnn_cell.GRUCell(self.b_rnn)
                drop = tf.nn.rnn_cell.DropoutWrapper(gru, output_keep_prob=1.0 - pd['dropout'] if g_training else 1.)
                cell = tf.nn.rnn_cell.MultiRNNCell([drop for _ in range(pd['rnn_layer'])])
                rnn_out, _ = tf.nn.dynamic_rnn(cell=cell, dtype=tf.float32, inputs=seq_embed,
                                               time_major=False)
                print('rnn_output.shape:', rnn_out.shape)
                # memory embedding
                mem_embed = self.encode(clk_embed, tf.reshape(u_fields, (self.n_batch, self.rnn_len, -1)))
                print('mem_embed.shape:', mem_embed.shape)
                # state embedding
                state_embed = tf.concat([rnn_out,
                                         mem_embed,
                                         tf.reshape(u_fields, (self.n_batch, self.rnn_len, -1))],
                                        axis=2)
                print('state_embed.shape:', state_embed.shape)
                state_dim = state_embed.get_shape().as_list()[-1]
                mlp_dims = [state_dim / 2, state_dim / 4, self.hint_dim]
                fc = state_embed
                for i in range(len(mlp_dims)):
                    fc = tf.layers.dense(fc, mlp_dims[i], activation=tf.nn.tanh)
                    fc = tf.layers.dropout(fc, rate=pd['dropout'], training=g_training)
                b_h_layer = fc
                print('hint_layer.shape:', b_h_layer.shape)
                q = tf.reshape(tf.layers.dense(b_h_layer, 1), (self.n_batch, self.rnn_len))
                print('q.shape:', q.shape)
                return b_h_layer, q

    # call for temporal-difference learning
    def target_bq(self, sess, ph_dict):
        return sess.run(self.btq, feed_dict={self.sph_clk_seq: ph_dict['clk_seq'],
                                             self.sph_user: ph_dict['user'],
                                             self.sph_doc: ph_dict['doc'],
                                             self.sph_con: ph_dict['con'],
                                             self.sph_future: ph_dict['future']})

    # call for calc normalized guide weight and q
    def guide_wnq(self, sess, ph_dict):
        tq = self.target_bq(sess, ph_dict)
        nq = np.append(tq[:, 1:], np.array([[0] for i in range(self.n_batch)], dtype=np.float32), 1)
        return sess.run([self.guide_weights, self.bmq],
                        feed_dict={self.sph_clk_seq: ph_dict['clk_seq'],
                                   self.ph_nbq: nq,
                                   self.ph_reward: ph_dict['reward'],
                                   self.sph_user: ph_dict['user'],
                                   self.sph_doc: ph_dict['doc'],
                                   self.sph_con: ph_dict['con'],
                                   self.sph_future: ph_dict['future']})

    # call for calc booter hint layer
    def bhint(self, sess, ph_dict):
        return sess.run(self.bmh, feed_dict={self.sph_clk_seq: ph_dict['clk_seq'],
                                             self.sph_user: ph_dict['user'],
                                             self.sph_doc: ph_dict['doc'],
                                             self.sph_con: ph_dict['con'],
                                             self.sph_future: ph_dict['future']})

    # call for temporal-difference learning
    def target_lq(self, sess, ph_dict):
        return sess.run(self.ltq, feed_dict={self.sph_clk_seq: ph_dict['clk_seq'],
                                             self.sph_user: ph_dict['user'],
                                             self.sph_doc: ph_dict['doc'],
                                             self.sph_con: ph_dict['con']})

    # call for evaluating booster networks
    def main_bq(self, sess, ph_dict):
        return sess.run(self.bmq, feed_dict={self.sph_clk_seq: ph_dict['clk_seq'],
                                             self.sph_user: ph_dict['user'],
                                             self.sph_doc: ph_dict['doc'],
                                             self.sph_con: ph_dict['con'],
                                             self.sph_future: ph_dict['future']})

    # call for evaluating booster networks
    def main_lq(self, sess, ph_dict):
        return sess.run(self.lmq, feed_dict={self.sph_clk_seq: ph_dict['clk_seq'],
                                             self.sph_user: ph_dict['user'],
                                             self.sph_doc: ph_dict['doc'],
                                             self.sph_con: ph_dict['con']})

    def get_lprob(self, sess, ph_dict):
        return sess.run(self.l_prob, feed_dict={self.sph_clk_seq: ph_dict['clk_seq'],
                                                self.sph_user: ph_dict['user'],
                                                self.sph_doc: ph_dict['doc'],
                                                self.sph_con: ph_dict['con']})

    def get_bprob(self, sess, ph_dict):
        return sess.run(self.b_prob, feed_dict={self.sph_clk_seq: ph_dict['clk_seq'],
                                                self.sph_user: ph_dict['user'],
                                                self.sph_doc: ph_dict['doc'],
                                                self.sph_con: ph_dict['con'],
                                                self.sph_future: ph_dict['future']})

    def get_breward(self, sess, ph_dict):
        return sess.run(self.b_reward, feed_dict={self.sph_clk_seq: ph_dict['clk_seq'],
                                                  self.sph_user: ph_dict['user'],
                                                  self.sph_doc: ph_dict['doc'],
                                                  self.sph_con: ph_dict['con'],
                                                  self.sph_future: ph_dict['future']})

    # call for learning from data
    def booster_learn(self, sess, ph_dict):
        tq = self.target_bq(sess, ph_dict)
        nq = np.append(tq[:, 1:], np.array([[0] for i in range(self.n_batch)], dtype=np.float32), 1)
        loss, _ = sess.run([self.booster_loss, self.booster_opt], feed_dict={self.ph_nbq: nq,
                                                                             self.ph_reward: ph_dict['reward'],
                                                                             self.sph_clk_seq: ph_dict['clk_seq'],
                                                                             self.sph_user: ph_dict['user'],
                                                                             self.sph_doc: ph_dict['doc'],
                                                                             self.sph_con: ph_dict['con'],
                                                                             self.sph_future: ph_dict['future']})
        self.b_loss += loss
        global g_batch_counter
        if g_batch_counter % 5 == 0:
            print(sess.run(self.global_step), ' ---Booster Network Loss: ', self.b_loss / g_batch_counter)

    # call for learning from data
    def lite_learn(self, sess, ph_dict):
        old_ph = copy.deepcopy(ph_dict)
        tq = self.target_lq(sess, ph_dict)
        nq = np.append(tq[:, 1:], np.array([[0] for i in range(self.n_batch)], dtype=np.float32), 1)
        guide_w, guide_q = self.guide_wnq(sess, ph_dict)
        hint = self.bhint(sess, ph_dict)
        gan_reward = self.get_breward(sess, ph_dict)
        loss, _ = sess.run([self.lite_loss, self.lite_opt], feed_dict={self.ph_nlq: nq,
                                                                       self.ph_hint: hint,
                                                                       self.ph_guide_weight: guide_w,
                                                                       self.ph_guide_q: guide_q,
                                                                       self.gan_reward: gan_reward,
                                                                       self.ph_reward: ph_dict['reward'],
                                                                       self.sph_clk_seq: ph_dict['clk_seq'],
                                                                       self.sph_user: ph_dict['user'],
                                                                       self.sph_doc: ph_dict['doc'],
                                                                       self.sph_con: ph_dict['con']})
        self.l_loss += loss
        global g_batch_counter
        if g_batch_counter % 5 == 0:
            print(sess.run(self.global_step), ' ---Lite Network Loss: ', self.l_loss / g_batch_counter)
        ph_dict = copy.deepcopy(old_ph)


def handle(sess, net, sess_data):
    def gen_sparse_tensor(fs):
        global g_dr
        kk, vv = [], []
        for i in range(len(fs)):
            ff = fs[i]
            assert (isinstance(ff, set))
            ff = list(ff)
            for k in range(len(ff)):
                kk.append(np.array([i, k], dtype=np.int32))
                vv.append(ff[k])
        return tf.SparseTensorValue(kk, vv, [len(fs), g_dr.unique_feature_num()])

    global g_rb
    g_rb.save(sess_data)
    while g_rb.has_batch():
        clk, user, doc, con, future, rwd, rtn = g_rb.next_batch()
        clk = np.array(clk).reshape(pd['batch_size'] * pd['clk_seq_len'] * pd['doc_field_num'])
        phd = {}
        phd['clk_seq'] = gen_sparse_tensor(clk)
        user = np.array(user).reshape(pd['batch_size'] * pd['rnn_seq_len'] * pd['user_field_num'])
        phd['user'] = gen_sparse_tensor(user)
        doc = np.array(doc).reshape(pd['batch_size'] * pd['rnn_seq_len'] * pd['doc_field_num'])
        phd['doc'] = gen_sparse_tensor(doc)
        con = np.array(con).reshape(pd['batch_size'] * pd['rnn_seq_len'] * pd['con_field_num'])
        phd['con'] = gen_sparse_tensor(con)
        future = np.array(future).reshape(pd['batch_size'] * pd['rnn_seq_len'] * pd['future_field_num'])
        phd['future'] = gen_sparse_tensor(future)
        phd['reward'] = rwd
        global g_batch_counter, g_training
        print(datetime.datetime.now(), 'start to handle batch', g_batch_counter)
        g_batch_counter += 1
        if g_training:
            net.booster_learn(sess, phd)
            net.lite_learn(sess, phd)
            # net.hint_learn(sess, phd)
            if g_batch_counter % pd['double_networks_sync_freq'] == 0:
                print('Run soft replacement for main networks and target networks...')
                sess.run(net.booster_sync_op)
                sess.run(net.lite_sync_op)
        else:
            qout = net.main_bq(sess, phd).reshape([-1])
            global g_working_mode
            for i in range(len(rtn)):
                if 'local_predict' == g_working_mode:
                    print('%s %s' % (rwd[i], qout[i]))
            pout = net.main_lq(sess, phd).reshape([-1])
            for i in range(len(rtn)):
                if 'local_predict' == g_working_mode:
                    print('%s %s' % (rwd[i], pout[i]))
        print(datetime.datetime.now(), 'batch finish, ', g_rb.dump())


def local_run():
    global_step = tf.train.get_or_create_global_step()
    sess = tf.Session()
    net = FutureGAN(global_step)
    saver = tf.train.Saver(max_to_keep=1)
    g_init_op = tf.global_variables_initializer()
    if os.path.exists('./ckpt'):
        model_file = tf.train.latest_checkpoint('ckpt/')
        saver.restore(sess, model_file)
    else:
        sess.run(g_init_op)
        os.system('mkdir ckpt')
    print('>>> local model...')
    global g_batch_counter
    for k in range(pd['num_epochs'] if g_training else 1):
        if k > 0:
            g_dr.load('sample.data')
        data = g_dr.next()
        while data is not None:
            handle(sess, net, data)
            data = g_dr.next()
            if g_training and g_batch_counter % 10 == 0:
                print(
                    '>>> epoch %d --- batch %d --- teacher net loss = %f --- student net loss = %f --- top-k distill loss = %f' % (
                        k, g_batch_counter, net.teacher_loss_val / (g_batch_counter + 1e-6),
                        net.student_loss_val / (g_batch_counter + 1e-6),
                        net.topk_loss_val / (g_batch_counter + 1e-6)))
    saver.save(sess, 'ckpt/afe.ckpt')


if __name__ == '__main__':

    g_working_mode = 'local_train'
    commander = {
        'local_train': local_run,
        'local_predict': local_run
    }
    if g_working_mode not in commander:
        print('your working mode(%s) not recognized!!!' % g_working_mode)
        sys.exit(1)

    g_training = True if g_working_mode == 'local_train' else False

    print('>>> working_model:%s\n>>> is_training:%s\nenable_gan:%s\nenable_distill:%s\nenable_future:%s' % (
        g_working_mode, g_training, pd['enable_gan'], pd['enable_distill'], pd['enable_future']))

    g_dr.load('sample.data')

    g_rb = RB()

    commander[g_working_mode]()
