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
result_path = '../Dataset/traj_uv_basic_edgegnn_only_traj_test_2att_sort.pickle'
#dataset = Dataset('../Dataset/data_all_uv_0908_35_with_feature.pickle', '../Dataset/data_all_uv_0920_10_with_feature.pickle', 10, 0.7)
dataset = Dataset('../Dataset/train_data_basic_41.csv', '../Dataset/test_data_101415_41.pickle', 10, 50)
print("get_train_set")
train_dataset, train_ids = dataset.get_train_set()
print("get_test_set")
test_dataset, test_ids = dataset.get_test_set()

train_dataLoader = DataLoader(train_dataset, batch_size=20, shuffle=True)
test_dataLoader = DataLoader(test_dataset, batch_size=20, shuffle=False)

DEVICE = torch.device('cuda:0') # cuda
train_size = len(train_dataset)
test_size = len(test_dataset)

print("Done")
print("Train num: " + str(len(train_dataset)))
print("Test num: " + str(len(test_dataset)))

model = Basic_GNN_LSTM(110,1,10,1).to(DEVICE)
#model = LSTM(10,1,5).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCELoss()

r_best = 0
for epoch in range(101):
    print("Epoch" + str(epoch))
    epoch_Loss = 0
    model.train()
    for batch in train_dataLoader:
        h_classcify, target_classcify, traj_aux_loss = model(batch)
        Loss = criterion(h_classcify, target_classcify.to(DEVICE))
        #Loss += 0.1*traj_aux_loss
        Loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_Loss += Loss
    print("Loss: " + str(epoch_Loss/train_size))
    
    if epoch %10 == 0:
        model.eval()
        all_prob = np.array([])
        all_y = np.array([])
        all_target = np.array([])
        print("eval in test_dataset")
        right_list = []
        fp_list = []
        fn_list = []
        for batch in test_dataLoader:
            h_classcify, target_classcify, traj_aux_loss = model(batch)
            y = h_classcify.detach().cpu()
            target = target_classcify.float().cpu().numpy()

            all_prob = np.append(all_prob, y)
            all_target = np.append(all_target, target)

        if epoch % 10 == 0:
            precision, recall, thresholds = precision_recall_curve(all_target, all_prob)
            p_np = np.array(precision)
            r_np = np.array(recall)
            t_np = np.array(thresholds)
            p_list = []
            r_list  =[]
            t_list = []
            p_filter = p_np[p_np >=0.9]
            if len(p_filter) != 0:
                r_now = r_np[-len(p_filter)]
            else:
                r_now = 0
            if r_now >= r_best:
                r_best = r_now
                pickle.dump([[precision, recall, thresholds], [test_ids, all_prob, all_target]], open(result_path,"wb"))
                torch.save(model.state_dict(), 'traj_uv_basic_edgegnn_only_traj_test_2att_sort.pt')

            for i in [0.95, 0.9, 0.8, 0.7]:
                p_filter = p_np[p_np >=i]
                if len(p_filter) !=0:
                    r_filter = r_np[-len(p_filter):]
                    t_filter = t_np[-len(p_filter):]
                    p_list.append(p_filter[0])
                    r_list.append(r_filter[0])
                    t_list.append(t_filter[0])
                else:
                    p_list.append(0)
                    r_list.append(0)
                    t_list.append(0)
            print(p_list)
            print(r_list)
            print(t_list)
        
    
    