#!encoding:utf-8
"""
    Data Reader
    @2022-07-26
    by Modric Zhang
"""

import sys
from id_allocator import IdAllocator
from hyper_param import param_dict as pd


def normalize_recommend(rec_list, max_len):
    """
    If len(rec_list) < max_len, rec_list will be dropped.
    Otherwise, rec_list will be truncated.
    """
    if len(rec_list) < max_len:
        return []
    else:
        return rec_list[-max_len:]


class DataReader(object):
    def __init__(self, batch_num):
        self._id_tool = IdAllocator()
        self._data = []
        self._batch = batch_num * pd['rnn_seq_len']

    def unique_feature_num(self):
        return self._id_tool.unique_id_num()

    def parse_feature(self, raw_feature):
        feature = set()
        for f in raw_feature.split(','):
            feature.add(self._id_tool.allocate(f))
        if len(feature) == 0:
            feature.add(0)
        return feature

    def load(self, sample_path):
        with open(sample_path, 'r') as fp:
            for line in fp:
                info = line.strip().split('\t')
                if len(info) != 4:
                    raise Exception('invalid data!')
                sid = ''
                offset = 0
                clk_seq, rec_list = [], []
                for ii in info:
                    pos = ii.find(':')
                    if pos <= 0:
                        raise Exception('invalid data!')
                    kk, vv = ii[:pos], ii[pos + 1:]
                    if kk == 'sid':
                        sid = vv
                    elif kk == 'offset':
                        offset = vv
                    elif kk == 'click_docs':
                        for doc in vv.split('|'):
                            if len(doc) == 0:
                                continue
                            feats = []
                            for ff in doc.split(' '):
                                pos = ff.find(':')
                                if pos <= 0:
                                    raise Exception('invalid data')
                                fk, fv = ff[:pos], ff[pos + 1:]
                                if 'f' in fk:
                                    feats.append(self.parse_feature(fv))
                                else:
                                    print(fk, fv)
                                    sys.exit(1)
                            clk_seq.append(feats)
                    elif kk == 'rec_list':
                        for doc in vv.split('|'):
                            feats, rwd, rtn = [], 0.0, 0.0
                            for ff in doc.split(' '):
                                pos = ff.find(':')
                                if pos <= 0:
                                    raise Exception('invalid data')
                                fk, fv = ff[:pos], ff[pos + 1:]
                                if fk == 'reward':
                                    rwd = float(fv)
                                elif fk == 'return':
                                    rtn = float(fv)
                                elif 'uf' in fk or 'df' in fk or 'cf' in fk or 'zf' in fk:
                                    feats.append(self.parse_feature(fv))
                                elif fk == 'idx':
                                    continue
                                else:
                                    print(fk, fv)
                                    sys.exit(0)
                            rec_list.append([feats, rwd, rtn])
                nrl = normalize_recommend(rec_list, pd['rnn_seq_len'])
                if len(nrl):
                    rec_seq = []
                    for doc in rec_list:
                        offset1 = pd['user_field_num']
                        offset2 = offset1 + pd['doc_field_num']
                        offset3 = offset2 + pd['con_field_num']
                        rec_seq.append(
                            [doc[0][:offset1], doc[0][offset1:offset2], doc[0][offset2:offset3], doc[0][offset3:],
                             doc[1], doc[2]])
                    self._data.append([sid, offset, clk_seq, rec_seq])

    def next(self):
        nb = None
        if len(self._data) <= 0:
            return nb
        else:
            idx = len(self._data) if len(self._data) <= self._batch else self._batch
            nb = self._data[:idx]
            self._data = self._data[idx:]
        return nb
