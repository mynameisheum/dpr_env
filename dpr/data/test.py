# import json
#
# with open('/home/tjdgma/multidoc2dial_project/data_renew/mdd_dpr_renew/dpr.multidoc2dial_all.structure.train_renew(both5-15).json', encoding='utf-8' ) as f: # train:8516, val:2117
#     train_renew = json.load(f)
# with open('/home/tjdgma/multidoc2dial_project/data_renew/mdd_dpr_renew_doc/dpr.multidoc2dial_all.structure.train_renew_doc-both5-15.json', encoding='utf-8' ) as f: # train:8516, val:2117
#     train_renew_doc = json.load(f)
#
#
# num = 0
# for idx, context in enumerate(train_renew):
#     num +=1
#     context['negative_ctxs'] = train_renew_doc[idx]['hard_negative_ctxs']
#
# print(num)
#
#
# with open('/home/tjdgma/multidoc2dial_project/data_renew/multidoc2dial_renew_hard_mix/dpr.multidoc2dial_all.structure.train_renew_hard_mix(both5-15).json', 'w') as ffff:  # train:8516, val:2117
#     json.dump( train_renew,ffff, indent=4)


# with open('/home/tjdgma/multidoc2dial_project/data_renew/mdd_dpr_renew_doc/dpr.multidoc2dial_all.structure.train_renew_+_doc(both5-15).json', encoding='utf-8' ) as f: # train:8516, val:2117
#     train_renew_doc_dioc = json.load(f)
#
# num = 0
# for i in train_renew_doc_dioc:
#     num +=1
#
# print(num)


# wiki passage 50 split 하기
import pandas as pd
import math

num_shard = 50
shard_id = 5

shard_size = math.ceil(21015324 / num_shard)
start_index = shard_id * shard_size
end_index = start_index + shard_size

print('num shard = ',num_shard)
print('shard id = ',shard_id)
print('shard size = ', shard_size)
print('start index = ',start_index)
print('end index = ', end_index)

a = pd.read_csv('/home/tjdgma/DPR/dpr/data/download_data/data/retriever/wikipedia_split/psgs_w100.tsv',
                # nrows= end_index,
                skiprows=shard_size*shard_id+1,
                nrows=shard_size,
                sep = '\t',
                names = ['id','text','title'])

a.to_csv('/home/tjdgma/DPR/dpr/data/download_data/data/retriever/wikipedia_split/psgs_w100_split_{}.tsv'.format(shard_id), index = False,header = True, sep = '\t')

