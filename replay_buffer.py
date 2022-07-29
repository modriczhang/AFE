#!encoding:utf-8
import sys

"""
    Static Replay Buffer
    @2020-02-24
    by Modric Zhang
"""

from hyper_param import param_dict as pd


class RB(object):

    def __init__(self):
        self.batch_num = pd['batch_size']
        self.clk_seq_len = pd['clk_seq_len']
        self.rnn_seq_len = pd['rnn_seq_len']
        self.clk_step = self.batch_num * self.clk_seq_len
        self.clk_step = self.batch_num * self.rnn_seq_len
        self.rec_step = self.batch_num * self.rnn_seq_len
        self.user_field_num = pd['user_field_num']
        self.doc_field_num = pd['doc_field_num']
        self.con_field_num = pd['con_field_num']
        self.future_field_num = pd['future_field_num']
        self.clk_buffer = []
        self.user_buffer = []
        self.doc_buffer = []
        self.con_buffer = []
        self.future_buffer = []
        self.reward_buffer = []
        self.return_buffer = []

    def save(self, sess_data):
        for data in sess_data:
            if len(data) != 4:
                print('sample format error, colum should be 4 not %d' % len(data))
                return False
        for data in sess_data:
            for clk_doc in data[2]:
                if len(clk_doc) != self.doc_field_num:
                    raise Exception('sample format error, doc_field_num=%d' % len(clk_doc))
                self.clk_buffer.append(clk_doc)
                # print 'clk_doc:', clk_doc
            for rec_doc in data[3]:
                if len(rec_doc) != 6:
                    raise Exception('sample format error, len(rec_doc)=%d' % len(rec_doc))
                if len(rec_doc[0]) != self.user_field_num:
                    raise Exception('sample format error, len(user)=%d' % len(rec_doc[0]))
                self.user_buffer.append(rec_doc[0])
                # print 'user:', rec_doc[0]
                if len(rec_doc[1]) != self.doc_field_num:
                    raise Exception('sample format error, len(doc)=%d' % len(rec_doc[1]))
                self.doc_buffer.append(rec_doc[1])
                # print 'doc:', rec_doc[1]
                if len(rec_doc[2]) != self.con_field_num:
                    raise Exception('sample format error, len(con)=%d' % len(rec_doc[2]))
                self.con_buffer.append(rec_doc[2])
                if len(rec_doc[3]) != self.future_field_num:
                    raise Exception('sample format error, len(future)=%d' % len(rec_doc[3]))
                self.future_buffer.append(rec_doc[3])
                # print 'con:', rec_doc[2]
                self.reward_buffer.append(rec_doc[4])
                self.return_buffer.append(rec_doc[5])

    def has_batch(self):
        return len(self.clk_buffer) >= self.clk_step and len(self.reward_buffer) >= self.rec_step

    def dump(self):
        return 'buffer_size:' + str(len(self.clk_buffer)) + ',' \
               + str(len(self.user_buffer)) + ',' \
               + str(len(self.doc_buffer)) + ',' \
               + str(len(self.con_buffer)) + ',' \
               + str(len(self.future_buffer))

    def next_batch(self):
        if not self.has_batch():
            raise Exception('replay buffer is almost empty')
        clk = self.clk_buffer[:self.clk_step]
        self.clk_buffer = self.clk_buffer[self.clk_step:]
        user = self.user_buffer[:self.rec_step]
        self.user_buffer = self.user_buffer[self.rec_step:]
        doc = self.doc_buffer[:self.rec_step]
        self.doc_buffer = self.doc_buffer[self.rec_step:]
        con = self.con_buffer[:self.rec_step]
        self.con_buffer = self.con_buffer[self.rec_step:]
        rwd = self.reward_buffer[:self.rec_step]
        self.reward_buffer = self.reward_buffer[self.rec_step:]
        rtn = self.return_buffer[:self.rec_step]
        self.return_buffer = self.return_buffer[self.rec_step:]
        future = self.future_buffer[:self.rec_step]
        self.future_buffer = self.future_buffer[self.rec_step:]
        return clk, user, doc, con, future, rwd, rtn
