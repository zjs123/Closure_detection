import pickle
import imp
import torch
import random
import numpy as np
from tqdm import tqdm
from Dataset import Dataset
from sklearn import metrics
import matplotlib.pyplot as plt
from RecurrentGCN import Basic_GNN_LSTM, LSTM
from torch_geometric.loader import DataLoader
from TemporalDataLoader import DataLoader
from sklearn.metrics import precision_recall_curve
import warnings
warnings.filterwarnings("ignore")

print("loading dataset...")
result_path = '../Dataset/traj_uv_basic_101415.pickle'
#dataset = Dataset('../Dataset/data_all_uv_0908_35_with_feature.pickle', '../Dataset/data_all_uv_0920_10_with_feature.pickle', 10, 0.7)
dataset = Dataset('../Dataset/test_data_101415_41.pickle', '../Dataset/test_data_101415_41.pickle', 10, 0.7)
#train_dataset, train_ids = dataset.get_train_set()
test_dataset, test_ids = dataset.get_test_set()

#train_dataLoader = DataLoader(train_dataset, batch_size=20, shuffle=True)
test_dataLoader = DataLoader(test_dataset, batch_size=20, shuffle=False)

DEVICE = torch.device('cuda:0') # cuda
#train_size = len(train_dataset)
test_size = len(test_dataset)

print("Done")
#print("Train num: " + str(len(train_dataset)))
print("Test num: " + str(len(test_dataset)))

model = Basic_GNN_LSTM(150,1,10,1).to(DEVICE)
model.load_state_dict(torch.load("traj_uv_basic.pt"))


all_prob = np.array([])
all_y = np.array([])
all_target = np.array([])
print("eval in test_dataset")
right_list = []
fp_list = []
fn_list = []
model.eval()
for batch in tqdm(test_dataLoader):
    h_classcify, target_classcify, traj_aux_loss = model(batch)
    y = h_classcify.detach().cpu()
    target = target_classcify.float().cpu().numpy()

    all_prob = np.append(all_prob, y)
    all_target = np.append(all_target, target)

precision, recall, thresholds = precision_recall_curve(all_target, all_prob)
pickle.dump([[precision, recall, thresholds], [test_ids, all_prob, all_target]], open(result_path,"wb"))
p_np = np.array(precision)
r_np = np.array(recall)
t_np = np.array(thresholds)
p_list = []
r_list  =[]
t_list = []

for i in [0.95, 0.9, 0.8, 0.7]:
    p_filter = p_np[p_np >=i]
    if len(p_filter) !=0:
        r_filter = r_np[-len(p_filter):]
        t_filter = t_np[-len(p_filter):]
        p_list.append(i)
        r_list.append(r_filter[0])
        t_list.append(t_filter[0])
    else:
        p_list.append(0)
        r_list.append(0)
        t_list.append(0)
print(p_list)
print(r_list)
print(t_list)