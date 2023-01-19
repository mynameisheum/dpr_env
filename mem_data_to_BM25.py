import json
import os
import argparse
import csv
import sys

from collections import defaultdict
from tqdm import tqdm
from rank_bm25 import BM25Okapi

import numpy as np
import pandas as pd
import gc
import time
import re
from konlpy.tag import Mecab

mecab = Mecab()
print(mecab.pos('최근 기분이 요즘 많이안좋아'))
print(mecab.pos('나는 영화보는 것을 좋아한다 나는 중국어에 관심 있지만 배울 시간이 없다'))


def get_bm25_ranking_memory( rankling_list, memory_map, begin=0, n=5):


    memory_bm25_ranking = []
    for ix, score in rankling_list[begin : begin + n]:
        if score != 0:
            # memory_bm25_ranking = [{'memory_idx':ix, 'score':score, 'memory':memory_map[ix]["memory"]}]
            memory_bm25_ranking.append({'memory_idx':ix, 'score':score, 'memory':memory_map[ix]["memory"].strip()})

        else :
            pass

    return memory_bm25_ranking

def get_bm25(passages):
    temp4 = []
    stopwords = ['최근','요즘'] # 시간 군대 친구
    for passage in passages:
        temp3 = []
        for tag in mecab.pos(passage):
            # if str(tag[1]) in ['NNG', 'NNP', 'NNB', 'NNBC', 'NR', 'NP', 'VV', 'VA', 'VX', 'VCP', 'VCN']: # N,V
            # if str(tag[1]) in ['NNG', 'NNP', 'NNB', 'NNBC', 'NR', 'NP']:# N
            if str(tag[1]) in ['NNG', 'NNP', 'NNBC', 'NR', 'NP']:# N- NNB(의존명사)
            # if str(tag[1]) in ['NNG', 'NNP', 'NNBC', 'NR', 'NP', 'VV', 'VA', 'VX']: # N- NNB(의존명사) + 소수의 V
                if str(tag[0]) not in stopwords:
                    temp3.append(tag[0])

        temp4.append(temp3)
    bm25 = BM25Okapi(temp4)

    return bm25

# def get_bm25(passages):
#     temp4 = []
#     for passage in passages:
#         temp3 = []
#         for tag in mecab.nouns(passage):
#             temp3.append(tag)
#         temp4.append(temp3)
#     bm25 = BM25Okapi(temp4)
#
#     return bm25


shard_id = 0
shard_size = 11000 #573900 291000
# file_path = '/home/tjdgma/tjdgma/Memory_data/MSC_data/bm25_top_5_test'
file_path = '/home/tjdgma/tjdgma/Memory_data/data_dir_light/bm25_top_5'

json_data = []

bm25_count = 0
print('reading memory row data')
print('shard_id : ',shard_id)
print('shard_size : ',shard_size)

# train data
# with open('/home/tjdgma/tjdgma/Memory_data/MSC_data/train_v1.source', 'r') as f: # 홍진
with open('/home/tjdgma/tjdgma/Memory_data/data_dir_light/val.source', 'r') as f: # 빛나
    memory_data = f.readlines()

# train id data
# with open('/home/tjdgma/tjdgma/Memory_data/MSC_data/train_v1.id', 'r') as ff:
with open('/home/tjdgma/tjdgma/Memory_data/data_dir_light/val.id', 'r') as ff:
    memory_id = ff.readlines()


print('data count : ', len(memory_data))
print('id count : ', len(memory_id))
time_start = time.time()
for index, data in enumerate(tqdm(memory_data[shard_size*shard_id:shard_size*(shard_id+1)], desc='Creating..')):
    if index == 10:
        print('g2')
    if  (data.find('<agent>') != -1 or data.find('<user>') != -1): # <user>, <agent> token 있는 line만 가져오기, 즉 메모리만 있는 line 그대로 두기
        data_split = re.split(r'<agent>|<user>', data) # [[memory1,2,3,4][user][agent][user][agent]...]
        query = data_split[1:] # 앞의 memory부분 버리고 turn들만 얻기
        bm25_query = query[-1:] # 마지막으로부터 3 or 1 turn의 발화만 가져오기
        assert len(bm25_query) == 1 or 2 or 3
        bm25_query = " ".join(bm25_query)

        memory_split = re.split(r'<user_mem>', data_split[0]) # agent, user 메모리만 적혀있는 부분을 <user_mem>으로 split하기
        user_memory = memory_split[1:]

        agent_memory = re.split(r'<agent_mem>', memory_split[0])# agent 메모리만 있는 부분을 <agent_mem>으로 split하기
        agent_memory = agent_memory[1:] # ''생긴걸 슬라이스로 제거

        agent_memory_map = {}
        for idx , memory in enumerate(agent_memory):
            agent_memory_map[idx] = {'memory' : str(memory)}

        user_memory_map = {}
        for idx, memory in enumerate(user_memory):
            user_memory_map[idx] = {'memory' : str(memory)}

        agent_bm25 = get_bm25(agent_memory)
        user_bm25 = get_bm25(user_memory)

        # space 단위
        # space_split_query = bm25_query.strip().lower().split()
        
        mecab_tokenizing_query = mecab.pos(bm25_query)
        # 용언(n),체언(v) 단위
        # filter_N_V_query = [tag[0] for tag in mecab_tokenizing_query if str(tag[1]) in ['NNG','NNP', 'NNB','NNBC','NR','NP','VV','VA','VX','VCP','VCN']]
        # 용언(n) 단위
        # filter_N_V_query = [tag[0] for tag in mecab_tokenizing_query if str(tag[1]) in ['NNG','NNP', 'NNB','NNBC','NR','NP']]
        # 용언(n)- 의존명사(nnb) 단위
        filter_N_V_query = [tag[0] for tag in mecab_tokenizing_query if str(tag[1]) in ['NNG','NNP','NNBC','NR','NP']]
        # 용언(n)- 의존명사(nnb) +동사 단위
        # filter_N_V_query = [tag[0] for tag in mecab_tokenizing_query if str(tag[1]) in ['NNG', 'NNP', 'NNBC', 'NR', 'NP', 'VV', 'VA', 'VX']]
        # 명사 단위
        # filter_N_V_query = mecab.nouns(bm25_query)

        # 한 글자 제거
        temp2 = []
        for idx, split in enumerate(filter_N_V_query):
            if len(split) == 1:
                temp2.append(idx)
        temp2.reverse()
        for i in temp2:
            del filter_N_V_query[i]
        
        
        # 통계기반 score 내기
        agent_memory_score = agent_bm25.get_scores(filter_N_V_query)
        user_memory_score = user_bm25.get_scores(filter_N_V_query)

        # <agent memory> top-3 가져오기
        agent_memory_score_idx = [(i, score) for i, score in enumerate(agent_memory_score)]
        agent_memory_ranking_list = sorted(agent_memory_score_idx, key=lambda x: x[1], reverse=True)
        top3_agent_memory = get_bm25_ranking_memory(agent_memory_ranking_list,agent_memory_map )

        # <user memory> top-3 가져오기
        user_memory_score_idx = [(i, score) for i, score in enumerate(user_memory_score)]
        user_memory_ranking_list = sorted(user_memory_score_idx, key=lambda x: x[1], reverse=True)
        top3_user_memory = get_bm25_ranking_memory(user_memory_ranking_list, user_memory_map)

        # 전체 데이터중 몇개의 데이터가 top-3랭킹을 가지고 있는지
        total_memory_score = agent_memory_score_idx+user_memory_score_idx
        for i in total_memory_score:
            if i[1] != 0:
                bm25_count += 1
                break
        # query가 있는 메모리 데이터
        json_data.append({
            'data number' : index,
            'data id' : memory_id[index].strip(),
            'query' : bm25_query.strip(),
            'top-5 agent memory' : top3_agent_memory,
            'top-5 user memory' : top3_user_memory
        })
        # query가 없는 메모리 데이터
    else:
        json_data.append({
            'data number' : index,
            'data id': memory_id[index].strip(),
            'query' : '',
            'top-5 agent memory' : [],
            'top-5 user memory' : []
        })

# cp = os.path.join(cfg.output_dir, cfg.checkpoint_file_name + "." + str(epoch))

time_end = time.time()-time_start
with open(os.path.join(file_path + '_' + str(bm25_count) + '.' + str(shard_id)), 'w') as outfile:
    json.dump(json_data, outfile, indent=4, ensure_ascii=False)

print(bm25_count)
print('yes')

#####################################################################









