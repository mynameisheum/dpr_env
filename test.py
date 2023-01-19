import json
import csv
import pandas as pd

with open('/home/tjdgma/DPR/dpr/data/download_data/data/retriever/NQ/nq-train.json','r') as f:
    train_data = json.load(f)


d_1 = []
d_2 = []
for idx, data in enumerate(train_data):
    print(data['question'])
    d_1.append(data['question'].strip())
    print(data['answers'])
    d_1.append(data['answers'])
    d_2.append(d_1)
    d_1 = []

    # if idx == 10:
    #     break

# print(d_2)

df = pd.DataFrame(d_2)
# df.to_csv('/home/tjdgma/DPR/dpr/data/download_data/data/retriever/NQ/train_csv.csv', index=False, sep='\t', header=False)
df.to_pickle('/home/tjdgma/DPR/dpr/data/download_data/data/retriever/NQ/train_pickle.pkl')











