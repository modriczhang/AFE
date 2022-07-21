#!encoding:utf-8
import sys
import datetime
import time

#len(samples)<max_len时，会进行padding补充
#len(samples)>max_len时，会进行截断
def parse_clk_seq(samples, max_len, doc_field_num, fea_limit):
    def parse_block(ss):
        ff = []
        vec = ss.split(' ')
        if len(vec) != doc_field_num:
            raise Exception('click seq data formate error 1!')
        for vi in vec:
            kk, vv = vi.split(':')
            if 'f' not in kk:
                raise Exception('click seq data formate error 2!')
            hash_feat = set()
            if len(vv):
                hash_feat = set([int(x) % fea_limit for x in vv.split(',') if int(x) % fea_limit > 0])
            if len(hash_feat) == 0:
                hash_feat = set([0])
            ff.append(hash_feat)
        return ff
    #for s in samples:
    #    print s
    samples = [x for x in samples if len(x) > 0]
    feat_info = []
    padding_len = max_len - len(samples)
    if padding_len > 0:
        for i in range(padding_len):
            feat_info.append([set([0]) for _ in range(doc_field_num)])
        for sample in samples:
            feat_info.append(parse_block(sample))
    else:
        idx = len(samples) - max_len
        while idx < len(samples):
            feat_info.append(parse_block(samples[idx]))
            idx += 1
    #for fi in feat_info:
    #    print fi
    #sys.exit(0)
    return feat_info

def parse_line(line, clk_seq_max_len, user_field_num, doc_field_num, con_field_num, fea_limit):    
    info = line.strip('\n').split('\t')
    if len(info) != 6:
        raise Exception('Sample Data Formate Error')
    tag = True
    sid = 0
    offset = 0
    clk_seq, rec_list = [], []
    for kv in info:
        pos = kv.find(':')
        if pos < 0:
            tag = False
            break
        kk, vv = kv[:pos], kv[pos+1:]
        if kk == 'uin' or kk == 'channel':
            continue
        if kk == 'sid':
            sid = int(vv)
        elif kk == 'offset':
            offset = int(vv)
        elif kk == 'clk_seq':
            clk_seq = vv.split('|')
        elif kk == 'rec_list':
            rec_list = vv.split('|')
        else:
            tag = False
            break
    if not tag:
        sys.stderr.write("data formate error!!!" + line)
        raise Exception('Sample Data Fromate Error')
    clk_seq = parse_clk_seq(clk_seq, clk_seq_max_len, doc_field_num, fea_limit)
    
    def gen_ff(fv):
        ff = set()
        if len(fv.strip()):
            ff = set([int(x) % fea_limit for x in fv.split(',') if int(x) % fea_limit > 0])
        if len(ff) == 0:
            ff = set([0])
        return ff

    new_rec_list = []
    for rli in rec_list:
        reward, returns = 0., 0.
        uff, dff, cff = [], [], []
        zff = []
        for kv in rli.split(' '):
            pos = kv.find(':')
            if pos < 0:
                tag = False
                break
            kk, vv = kv[:pos], kv[pos+1:]
            if 'uf' in kk:
                uff.append(gen_ff(vv))
            elif 'df' in kk:
                dff.append(gen_ff(vv))
            elif 'cf' in kk:
                cff.append(gen_ff(vv))
            elif kk == 'reward':
                reward = max(min(3.0, float(vv)), 0.0)
            elif kk == 'return':
                returns = float(vv)
            elif 'zf' in kk:
                zff.append(gen_ff(vv))
        if not tag or len(uff) != user_field_num or len(dff) != doc_field_num or len(cff) != con_field_num:
            sys.stderr.write('parse rec list error,' + str(tag) + '--' + str(len(uff)) + ',' + str(len(dff)) + ',' + str(len(cff)) + '\n')
            raise Exception('Parse Rec List Error!!!')
        new_rec_list.append([uff, dff, cff, reward, returns, zff])
    rec_list = new_rec_list
    rec_len = len(rec_list)
    if rec_len > 10:
        rec_list = rec_list[rec_len-10:]
    elif rec_len < 10:
        padding_user = [set([0]) for _ in range(user_field_num)]
        padding_doc = [set([0]) for _ in range(doc_field_num)]
        padding_con = [set([0]) for _ in range(con_field_num)]
        padding_zdoc = [set([0]) for _ in range(2)]
        padding_rec_list = []
        for i in range(10-rec_len):
            padding_rec_list.append([padding_user, padding_doc, padding_con, 0., 0., padding_zdoc])
        padding_rec_list.extend(rec_list)
        rec_list = padding_rec_list
    assert(len(rec_list) == 10)
    return sid, offset, clk_seq, rec_list

def read_data(fname, clk_seq_max_len, user_field_num, doc_field_num, con_field_num, fea_limit):
    last_sid = ''
    sess_data = []
    rcnt = 0
    with open(fname, 'r') as fp:
        for line in fp:
            rcnt += 1
            if rcnt % 100 == 0:
                print datetime.datetime.now(), 'read ' + fname + ', lines =', rcnt
                sys.stdout.flush()
            try:
                sid, offset, clk_seq, rec_list = parse_line(line, clk_seq_max_len, user_field_num, doc_field_num, con_field_num, fea_limit)
                ##简化decoder, 固定长度为10
                #if len(rec_list) >= clk_seq_max_len:
                #    rec_list = rec_list[-clk_seq_max_len:]
                #else:
                #    continue  #忽略推荐结果小于10条的数据
            except Exception, e:
                sys.stderr.write('exception:' + str(e) + '\n')
                #sys.stderr.write(line)
                continue
            if last_sid != '' and last_sid != sid:
                yield sess_data
                sess_data = []
            sess_data.append([sid, offset, clk_seq, rec_list])
            last_sid = sid
        if len(sess_data):
            yield sess_data

#for pulls in read_data('sample.in', 10, 5, 5, 5, 8388593):
#    #print 'pull_num:', len(pulls)
#    for data in pulls:
#        #print 'sid:', data[0]
#        #print 'offset:', data[1]
#        #print 'short_history_len:', len(data[2])
#        #for s in data[2]:
#        #    print s
#        #print 'rec_list_len:', len(data[3])
#        #print data[2]
#        for r in data[3]:
#            print data[0], data[1], r[-2], r[-1]
#        break
#    break
#
