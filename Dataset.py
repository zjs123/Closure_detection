from unicodedata import unidata_version
from tqdm import tqdm
import string
from unittest import result
import torch
import pickle
import pandas
import numpy as np
import random as rd
import pandas as pd
from torch_geometric_temporal.signal import StaticGraphTemporalSignal, DynamicGraphTemporalSignal
 
class Dataset(object):
    def __init__(self, raw_data_dir, test_dir, his_len, point_sample):
        super(Dataset, self).__init__()
        self.raw_data_dir = raw_data_dir
        self.test_dir = test_dir
        self.his_length = his_len
        self.point_sample = 50

        self.train_raw = None
        self.test_raw = None
        self.node_f = 0
        self.graph_f = 0
        self.label_num = 0
        self.label_weights = [1,1,1]
        self.read_raw_data()

    def read_raw_data(self):
        if 'csv'  not in self.raw_data_dir:
            f = open(self.raw_data_dir, 'rb')
            dataset = pickle.load(f, encoding='iso-8859-1')
        else:
            dataset = pd.read_csv(self.raw_data_dir)


        self.graph_f = 6+20 #len(dataset.loc[0]['feature'][:-1]) #len(dataset.loc[0]['feature'][:-1])
        self.node_f = 6 #len(dataset.loc[0]['feature'][-1])
        self.label_num = 1
        
        if self.test_dir == None:
            self.train_raw = dataset.sample(frac=0.7)
            self.test_raw = dataset[~dataset.index.isin(self.train_raw.index)]
            self.train_raw = self.train_raw.reset_index(drop=True)
            self.test_raw = self.test_raw.reset_index(drop=True)
        else:
            test_f = open(self.test_dir, 'rb')
            test_dataset = pickle.load(test_f, encoding='iso-8859-1')
            self.train_raw = dataset.reset_index(drop=True)
            self.test_raw = test_dataset.reset_index(drop=True)
        return 0
    
    def combine_graphs(self, graph_list):
        link_set = set()
        combined_graph_dict = {}
        sub_dict_list = []
        new_graph_list = []
        for index in range(len(graph_list)):
            sub_graph = graph_list[index]
            
            link_set = link_set | set(sub_graph[0])

            tmp_dict = {}
            for in_index in range(len(sub_graph[0])):
                tmp_dict[in_index] = sub_graph[0][in_index]
            sub_dict_list.append(tmp_dict)
        
        combined_graph_dict[sub_graph[0][0]] = 0
        new_index = 1
        for link in link_set:
            if link not in combined_graph_dict.keys():
                combined_graph_dict[link] = new_index
                new_index += 1
        
        all_nodes = [0]*len(link_set)
        for link in combined_graph_dict.keys():
            all_nodes[combined_graph_dict[link]] = link

        for index in range(len(graph_list)):
            sub_graph = graph_list[index]
            nodes = sub_graph[0]
            node_num = sub_graph[1]
            new_src = [combined_graph_dict[sub_dict_list[index][in_index]] for in_index in sub_graph[2]]
            new_dst = [combined_graph_dict[sub_dict_list[index][in_index]] for in_index in sub_graph[3]]
            features = sub_graph[4]
            uv = sub_graph[5]
            static = sub_graph[6]
            edge_type = sub_graph[-1]

            new_graph_list.append([nodes, node_num, new_src, new_dst, features, uv, static, edge_type])
        
        return new_graph_list, all_nodes


    def get_uv_f(self, now_uv, day_uv, week_uv, Type):
        now_uv_np = np.array(now_uv)
        day_uv_np = np.array(day_uv)
        week_uv_np = np.array(week_uv)

        now_uv_last_2 = now_uv_np[-2:]
        day_uv_last_2 = day_uv_np[-2:]
        week_uv_last_2 = week_uv_np[-2:]

        now_uv_last_5 = now_uv_np[-10:]
        day_uv_last_5 = day_uv_np[-10:]
        week_uv_last_5 = week_uv_np[-10:]

        now_mean = np.mean(now_uv_last_2)
        now_day_sub = np.mean(now_uv_last_2) / (np.mean(day_uv_last_2)+1)
        now_week_sub = np.mean(now_uv_last_2) / (np.mean(week_uv_last_2)+1)

        low_uv_ratio_now = float(len(now_uv_last_5[now_uv_last_5 <= 2]))/float(len(now_uv_last_5))
        low_uv_ratio_now_day_sub = float(len(now_uv_last_5[now_uv_last_5 <= 2]))/float(len(now_uv_last_5)) - float(len(day_uv_last_5[day_uv_last_5 <= 2]))/float(len(day_uv_last_5))
        low_uv_ratio_now_week_sub = float(len(now_uv_last_5[now_uv_last_5 <= 2]))/float(len(now_uv_last_5)) - float(len(week_uv_last_5[week_uv_last_5 <= 2]))/float(len(week_uv_last_5))

        trade_now = np.mean(now_uv_np[:5])/np.mean(now_uv_last_2) if np.mean(now_uv_last_2) != 0 else np.mean(now_uv_np[:5])
        trade_day = np.mean(day_uv_np[:5])/np.mean(day_uv_last_2) if np.mean(day_uv_last_2) != 0 else np.mean(day_uv_np[:5])
        trade_week = np.mean(week_uv_np[:5])/np.mean(week_uv_last_2) if np.mean(week_uv_last_2) != 0 else np.mean(week_uv_np[:5])

        #print(np.array(uv_f).shape)
        uv_f = list(now_uv_last_5)+list(now_uv_last_5 - day_uv_last_5)+list(now_uv_last_5 - week_uv_last_5)
        #print(np.array(uv_f).shape)

        return np.array(uv_f)

    def traj_padding(self, f):
        f_list = list(f)
        new_f_list = []
        padding_list = []
        if len(f_list) < self.point_sample:
            f_tmp = f_list+[0]*self.point_sample+f_list
            padding_tmp =[0]*len(f_list)+[1]*self.point_sample

            new_f_list = f_tmp[:self.point_sample]
            padding_list = padding_tmp[:self.point_sample]
        else:
            split_ratio = self.point_sample//5
            f_pre_list = f_list[:split_ratio]
            f_aft_list = f_list[-split_ratio:]
            f_mid_list = f_list[split_ratio:-split_ratio]

            index_sample = np.random.choice(a=len(f_mid_list), size=self.point_sample - len(f_pre_list) - len(f_aft_list), replace=False, p=None)
            f_mid_smaple = list(np.array(f_mid_list)[index_sample])

            new_f_list = list(f_pre_list)+list(f_mid_smaple)+list(f_aft_list)
            padding_list = [0]*len(new_f_list)
        return new_f_list,padding_list

    def get_graph_feature(self, sub_graph, all_nodes, link_dir, Type): # dim: N*f
        # get kind raw feature
        cat_f = np.array([[sub_graph[4][i][0]] for i in range(len(sub_graph[4]))]) # n*1
        link_ids = sub_graph[0] # n*1

        # get static raw feature
        static_f = np.array(sub_graph[6]) # n*static_f
        try:
            link_length = static_f[:,2] # n*1
            link_direction = static_f[:,1] # n*1
        except:
            print(sub_graph)
            print(static_f)

        # get traj raw feature
        lng = np.array([self.traj_padding(sub_graph[4][i][1])[0] for i in range(len(sub_graph[4]))]) # n*point_sample
        lat = np.array([self.traj_padding(sub_graph[4][i][2])[0] for i in range(len(sub_graph[4]))])
        lng = []
        lat = []
        for i in range(len(sub_graph[4])):
            new_lng = []
            new_lat = []

            raw_lng = sub_graph[4][i][1]
            raw_lat = sub_graph[4][i][2]

            lng_middle = np.mean(raw_lng)
            lat_middle = np.mean(raw_lat)

            for j in range(len(raw_lng)):
                new_lng.append(raw_lng[j] - lng_middle*100000)
                new_lat.append(raw_lat[j] - lat_middle*100000)
            lng.append(self.traj_padding(new_lng)[0])
            lat.append(self.traj_padding(new_lat)[0])

        speed = np.array([np.abs(self.traj_padding(sub_graph[4][i][3])[0]) for i in range(len(sub_graph[4]))])
        direction = []
        for i in range(len(sub_graph[4])):
            new_direction = []
            raw_direction = sub_graph[4][i][4]
            try:
                real_direction = float(link_dir[link_ids[i]])
            except:
                try:
                    real_direction = raw_direction[0]
                except:
                    real_direction = 0
            for j in raw_direction:
                direction_change = abs(j - real_direction)
                if direction_change > 180:
                    direction_change_ratio = abs(360 - direction_change)
                else:
                    direction_change_ratio = direction_change
                new_direction.append(direction_change_ratio)
            direction.append(self.traj_padding(new_direction)[0])
        
        order_status = np.array([self.traj_padding(sub_graph[4][i][5])[0] for i in range(len(sub_graph[4]))])
        dist = np.array([self.traj_padding(sub_graph[4][i][6])[0]/(link_length[i]+1)  for i in range(len(sub_graph[4]))])
        padding = np.array([self.traj_padding(sub_graph[4][i][1])[1] for i in range(len(sub_graph[4]))]) # n*point_sample

        # get uv raw feature
        uv_f = [] # n*uv_f_dim
        for index in range(len(sub_graph[5])):
            now_uv = sub_graph[5][index][0]
            day_uv = sub_graph[5][index][1]
            week_uv = sub_graph[5][index][2]

            uv_f.append(self.get_uv_f(now_uv, day_uv, week_uv, Type))

        # combine features
        cat_uv_f = np.concatenate((cat_f, uv_f, static_f), -1) # n*(1+uv_f_dim+static_f)
        traj_f = [[lng[i], lat[i], speed[i], direction[i], dist[i], order_status[i]] for i in range(len(lng))] # n*6*point_sample

        
        # pad empty features for unseen nodes
        all_cat_uv_f = [] # N*(uv_f_dim+2)
        all_traj_f = [] # N*6*point_sample
        all_padding = [] # N*point_sample

        for node in  all_nodes:
            if node in sub_graph[0]:
                node_index = np.where(np.array(sub_graph[0]) == node)[0][0]
                all_cat_uv_f.append(cat_uv_f[node_index])
                all_traj_f.append(traj_f[node_index])
                all_padding.append(padding[node_index])

            else:
                all_cat_uv_f.append([0]*38)
                all_traj_f.append([[0]*self.point_sample, [0]*self.point_sample, [0]*self.point_sample, [0]*self.point_sample, [0]*self.point_sample, [0]*self.point_sample])
                all_padding.append([1]*self.point_sample)

        return np.array(all_cat_uv_f), np.array(all_traj_f), all_padding
    
    def get_traj_aux_task_sample(self, raw_traj, all_nodes, Type): # m*3
        # get link_id to combined graph index dict
        id_2_index_dict = {}
        for i in range(len(all_nodes)):
            id_2_index_dict[all_nodes[i]] = i
        
        # get positive and negative traj segement samples [target_seg, pos_next_seg, neg_next_seg]
        project_links = raw_traj[7]
        project_links_union = sorted(set(project_links),key=project_links.index)

        sample_list = []
        if len(project_links_union) <= 2:
            return sample_list
        
        for i in range(len(project_links_union)):
            if i != len(project_links_union)-1:
                try:
                    tmp_sample = [id_2_index_dict[project_links_union[i]], id_2_index_dict[project_links_union[i+1]]]
                    neg_list = project_links_union.copy()
                    neg_list.remove(project_links_union[i])
                    neg_list.remove(project_links_union[i+1])
                    neg_sample = rd.choice(neg_list)
                    tmp_sample.append(id_2_index_dict[neg_sample])
                    sample_list.append(tmp_sample)
                except:
                    continue
        return sample_list
    
    def get_traj_graphs(self, sample, Type):
        if 'csv'  not in self.raw_data_dir or Type != 'Train':
            graph_combined, all_node = self.combine_graphs(sample['order_2_traj_plan_graph_dict_list'])
        else:
            graph_combined, all_node = self.combine_graphs(eval(sample['order_2_traj_plan_graph_dict_list']))
        seq_padding_index = 0
        traj_padding = [] # T*n*point_sample
        uv_cat_f_list = [] # T*n*(uv_f_dim+1)
        traj_f_list = [] # T*n*6*point_sample
        target_list = [] # T*n*1
        order_target_list = [] #T*1

        edge_index_list = [] # T*2*edge_num
        edge_weight_list = [] # T*edge_num

        traj_aux_task_sample_list = [] # T*m(different for each step)*3

        seq_span_list = [] # T*1

        pre_time = 0
        order_times = []
        order_ids = []
        order_types = []
        i = 0
        j = 0

        link_dir = sample['link_dir']
        if 'csv' not in self.raw_data_dir or Type != 'Train':
            SAMPLE_Traj = sample['traj']
            SAMPLE_Order = sample['order_info']
        else:
            SAMPLE_Traj = eval(sample['traj'])
            SAMPLE_Order = eval(sample['order_info'])
        while j < len(SAMPLE_Traj):
            traj_sample = SAMPLE_Traj[j]
            if len(traj_sample[1]) <= 2:
                order_type = 0
            else:
                order_type = 1
            order_id = traj_sample[0]
            order_ids.append(order_id)
            if order_id == SAMPLE_Order[i][2]:
                order_times.append(SAMPLE_Order[i][1])
                order_types.append(order_type)
                j += 1
            i += 1
        sorted_id = np.argsort(order_times)
        for i in range(self.his_length):
            if i < len(sorted_id):
                sub_graph = graph_combined[sorted_id[i]]
                seq_padding_index += 1

                raw_traj = SAMPLE_Traj[sorted_id[i]]
                timestamp = order_times[sorted_id[i]]

                if order_types[sorted_id[i]] == 1:
                    if len(sub_graph[4][0][1]) <= 2:
                        order_target = 0
                    else:
                        order_target = 1
                else:
                    order_target = 0
            else:
                sub_graph = graph_combined[sorted_id[-1]]

                raw_traj = SAMPLE_Traj[sorted_id[-1]]
                timestamp = order_times[sorted_id[-1]]
                
                if order_types[sorted_id[-1]] == 1:
                    if len(sub_graph[4][0][1]) <= 2:
                        order_target = 0
                    else:
                        order_target = 1
                else:
                    order_target = 0

            if i== 0:
                seq_span_list.append(0)
                pre_time = timestamp
            else:
                time_span = float((timestamp - pre_time)/60)
                seq_span_list.append(time_span)
                pre_time = timestamp
            edges = np.array([sub_graph[2], sub_graph[3]])
            edge_weight = np.array(sub_graph[-1])
            edge_index_list.append(edges)
            edge_weight_list.append(edge_weight)

            uv_cat_f, traj_f, padding = self.get_graph_feature(sub_graph, all_node, link_dir, Type)
            traj_aux_task_sample = self.get_traj_aux_task_sample(raw_traj, all_node, Type)

            uv_cat_f_list.append(uv_cat_f)
            traj_f_list.append(traj_f)
            traj_padding.append(padding)

            target = np.array([int(sample['status'])])
            target_list.append(np.array([target]*len(uv_cat_f)))
            order_target_list.append(order_target)

            traj_aux_task_sample_list.append(traj_aux_task_sample)

        uv_cat_f_array = np.array(uv_cat_f_list)
        traj_f_array = np.array(traj_f_list)
        traj_padding_array = np.array(traj_padding)
        target_array = np.array(target_list)
        order_target_array = np.array(order_target_list)

        graph = DynamicGraphTemporalSignal(edge_indices = edge_index_list, edge_weights = edge_weight_list, features = uv_cat_f_array, targets = target_array, traj_f = traj_f_array, traj_padding = traj_padding_array)
        return [graph, seq_padding_index, traj_aux_task_sample_list, seq_span_list, order_target_array]

    def get_train_set(self):
        train_set = []
        train_ids = []
        for index in tqdm(range(len(self.train_raw))):
            row = self.train_raw.loc[index]
            graph_list = self.get_traj_graphs(row, 'Train')
            train_set.append(graph_list)
            train_ids.append(row['info_id'])
        return train_set, train_ids
    
    def get_test_set(self):
        test_set = []
        test_ids = []
        for index in tqdm(range(len(self.test_raw))):
            row = self.test_raw.loc[index]
            if '101415' in self.test_dir or "415" in self.test_dir:
                if row['status'] == 0:
                    order_check = self.order_check(row)
                    if order_check == 1:
                        continue
            
            graph_list = self.get_traj_graphs(row, 'Test')
            test_set.append(graph_list)
            test_ids.append(row['info_id'])
        return test_set, test_ids

    def np_move_avg(self, a,n=3,mode="same"):
        return(np.convolve(a+0.1, np.ones((n,))/n, mode=mode))

    def uv_padding(self, now_uv, now_rp, day_uv, week_uv):
        '''
        rp_ratio = np.mean(now_uv[-4:-2])/(np.mean(now_rp[-4:-2])+1)
        day_ratio = np.mean(day_uv[-2:])/(np.mean(day_uv[-4:-2])+1)
        week_ratio = np.mean(week_uv[-2:])/(np.mean(week_uv[-4:-2])+1)

        rp_result = rp_ratio*np.array(now_rp[-2:])
        day_result = day_ratio*np.array(now_uv[-4:-2])
        week_result = week_ratio*np.array(now_uv[-4:-2])

        result = (rp_result+day_result+week_result)//3
        '''
        
        now_uv_smooth = self.np_move_avg(now_uv[:-2])
        gradient_list = np.diff(now_uv_smooth)/now_uv_smooth[:-1]
        gradient = np.mean(gradient_list[-2:])
        result_1 = abs(now_uv_smooth[-1]*gradient+now_uv_smooth[-1])
        result_2 = abs(result_1*gradient+result_1)
        

        return [result_1, result_2]



    def order_check(self, row):
        orders = np.array(row['order_info'])
        last_uv = np.array(row['online_nei'])[0][0][0][-2:]
        exam_et = np.int64(row['exam_t'])
        order_spans = exam_et - np.array(orders[:,1])
        order_types = np.array(orders[:,0])
        count = float(np.sum(order_types == 0) /len(order_types))
        
        if count <= 0.3 or np.mean(last_uv)>=1.5 or np.mean(order_types[-2:]) == 1:
            return 0
        return 1