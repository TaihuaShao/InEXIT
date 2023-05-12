# coding: utf-8

import sys, os, random
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer

def load_sents(token, args):
    filepath = args.data_path

    if token == 'geek':
        max_feature_num = args.geek_max_feature_num
    elif token == 'job':
        max_feature_num = args.job_max_feature_num
    else:
        raise AssertionError("Not defined")

    sents = {}
    sent_num = 0
    filepath = os.path.join(filepath, '{}'.format(token))

    print('\nLoading from {}'.format(filepath))
    sys.stdout.flush()  
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in tqdm(file):
            idx = line.strip().split('\t')[0]
            sent = line.strip().split('\t')[1:]
            if idx not in sents:
                sent_num = 0
                sents[idx] = []
            if sent_num == max_feature_num: continue
            for s in sent:
                s = ''.join(s.strip().split(' '))
                sents[idx].append(s)   
                sent_num += 1
                if sent_num == max_feature_num: break
    return sents

class PJFDataset(Dataset):
    '''
    geek_sents: dict[str, LongTensor(1, max_sent_len['geek'])]
    job_sents: dict[str, LongTensor(1, max_sent_len['job'])]
    '''
    def __init__(self, geek_sents, job_sents, args, token):
        super(PJFDataset, self).__init__()
        filepath = args.data_path
        self.geek_sents = geek_sents
        self.job_sents = job_sents
        self.pairs, self.labels = self.load_pairs(filepath, token)
        self.bert_tokenizer =  AutoTokenizer.from_pretrained(args.bert_path)
        self.max_feat_len = args.max_feat_len  
        self.max_sent_len = args.max_sent_len  

    def load_pairs(self, filepath, token):
        pairs = []
        labels = []
        assert token in ['train', 'test', 'valid']
        filepath = os.path.join(filepath, '{}.data'.format(token))

        print('\nLoading from {}'.format(filepath))
        sys.stdout.flush()
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in tqdm(file):
                geek_id, job_id, label = line.strip().split('\t')
                if geek_id not in self.geek_sents or job_id not in self.job_sents: continue
                pairs.append([geek_id, job_id])
                labels.append(int(label))
        return pairs, torch.FloatTensor(labels)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        geek_sent = self.geek_sents[pair[0]]
        job_sent = self.job_sents[pair[1]]

        geek = []
        geek_taxon = '简历信息'
        geek_feat_0 = '居住城市'
        geek_feat_1 = '期望工作城市'
        geek_feat_2 = '期望工作行业'
        geek_feat_3 = '期望工作类型'
        geek_feat_4 = '期望薪资'
        geek_feat_5 = '当前工作行业'
        geek_feat_6 = '当前工作类型'
        geek_feat_7 = '当前薪资'
        geek_feat_8 = '学历'
        geek_feat_9 = '年龄'
        geek_feat_10 = '开始工作时间'
        geek_feat_11 = '工作经验'
        geek.append(geek_taxon)
        geek.append(geek_feat_0)
        geek.append(geek_feat_1)
        geek.append(geek_feat_2)
        geek.append(geek_feat_3)
        geek.append(geek_feat_4)
        geek.append(geek_feat_5)
        geek.append(geek_feat_6)
        geek.append(geek_feat_7)
        geek.append(geek_feat_8)
        geek.append(geek_feat_9)
        geek.append(geek_feat_10)
        geek.append(geek_feat_11)
        geek.append(geek_sent[0])
        geek.append(geek_sent[1])
        geek.append(geek_sent[2])
        geek.append(geek_sent[3])
        geek.append(geek_sent[4])
        geek.append(geek_sent[5])
        geek.append(geek_sent[6])
        geek.append(geek_sent[7])
        geek.append(geek_sent[8])
        geek.append(geek_sent[9])
        geek.append(geek_sent[10])

        job = []
        job_taxon = '工作信息'
        job_feat_0 = '工作名称'
        job_feat_1 = '工作城市'
        job_feat_2 = '工作类型'
        job_feat_3 = '招聘人数'
        job_feat_4 = '薪资'
        job_feat_5 = '招聘开始时间'
        job_feat_6 = '招聘结束时间'
        job_feat_7 = '是否出差'
        job_feat_8 = '工作年限'
        job_feat_9 = '最低学历'
        job_feat_10 = '工作描述'
        
        job.append(job_taxon)
        job.append(job_feat_0)
        job.append(job_feat_1)
        job.append(job_feat_2)
        job.append(job_feat_3)
        job.append(job_feat_4)
        job.append(job_feat_5)
        job.append(job_feat_6)
        job.append(job_feat_7)
        job.append(job_feat_8)
        job.append(job_feat_9)
        job.append(job_feat_10)
        job.append(job_sent[0])
        job.append(job_sent[1])
        job.append(job_sent[2])
        job.append(job_sent[3])
        job.append(job_sent[4])
        job.append(job_sent[5])
        job.append(job_sent[6])
        job.append(job_sent[7])
        job.append(job_sent[8])
        job.append(job_sent[9])

        geek_tokens = self.bert_tokenizer(geek, padding='max_length', truncation=True, max_length=self.max_feat_len, return_tensors='pt')

        job_tokens = self.bert_tokenizer(job, padding='max_length', truncation=True, max_length=self.max_feat_len, return_tensors='pt')

        geek_sent_tokens = self.bert_tokenizer(geek_sent[11], padding='max_length', truncation=True, max_length=self.max_sent_len, return_tensors='pt')

        job_sent_tokens = self.bert_tokenizer(job_sent[10], padding='max_length', truncation=True, max_length=self.max_sent_len, return_tensors='pt')

        return geek_tokens['input_ids'], geek_tokens['token_type_ids'], geek_tokens['attention_mask'], job_tokens['input_ids'], job_tokens['token_type_ids'], job_tokens['attention_mask'], geek_sent_tokens['input_ids'], geek_sent_tokens['token_type_ids'], geek_sent_tokens['attention_mask'], job_sent_tokens['input_ids'], job_sent_tokens['token_type_ids'], job_sent_tokens['attention_mask'], self.labels[index]