from torch_geometric.data import Data, DataLoader
import pandas as pd
import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler
import torch

import dgl
import os
from dgl.data import DGLDataset


def create_base_graph():
    # connection GRAPH
    edges_data = pd.read_csv('../STGCN-WVAN/data-su/edge_features2.csv')
    
    edge_fe_name = ['area_ratio','centre distance', 'aspect angle differ', 'elevation differ']
    edges_src = torch.from_numpy(edges_data['src'].to_numpy())
    edges_dst = torch.from_numpy(edges_data['des'].to_numpy())

    g = dgl.graph((edges_src, edges_dst))
    #edge_features = torch.from_numpy(edges_data[edge_fe_name].to_numpy().astype('float16'))
    #g.edata['weight'] = edge_features
    return g

from datetime import datetime, timedelta
def generate_dt_points(st_dt,en_dt):
    start_date = datetime.strptime(st_dt, '%Y%m%d').date()
    end_date = datetime.strptime(en_dt, '%Y%m%d').date()

    # date increase 12day
    date_increment = timedelta(days=12)

    date_list = []
    current_date = start_date

    while current_date <= end_date:
        date_list.append(current_date.strftime('%Y%m%d'))
        current_date += date_increment
        
    return date_list



def transform_time_series(data_list, window_size):
    """
    parameters:
        data_list (list of pd.DataFrame): DataFrame。
        window_size (int): history length。
        
    return:
        np.array: shape of (time_steps, dynamic_fes, history length, samples)。
    """
    n_features = data_list[0].shape[1]  
    n_samples = data_list[0].shape[0]   
    n_times = len(data_list)            

    effective_time_points = n_times - window_size + 1

    transformed_data = np.zeros((effective_time_points, n_features, window_size, n_samples))

    for t in range(effective_time_points):
        for w in range(window_size):
            idx = t + w
            transformed_data[t, :, w, :] = data_list[idx].values.T

    return transformed_data


def get_model_data(device):
    static_node_data = pd.read_csv('../STGCN-WVAN/data-su/node data_ERA5.csv')
    #node_label_data = pd.read_csv('./data-su/mean_abs_deformation_17-22.csv')
    node_label_data = pd.read_csv('../STGCN-WVAN/data-su/smoothed_Savitzky-Golay_deformation_17-22.csv')
    
    edge_data = pd.read_csv('../STGCN-WVAN/data-su/edge_features2.csv')
    node_label_data.isna().sum()
    node_label_data = node_label_data.fillna(0)
    #dy_fe_name = ['max_prec','mean_prec','max_GWS','mean_GWS','min_GWS','differ_GWS','max_SM','mean_SM','max_temp','mean_temp','min_temp']
    dy_fe_name = ['max_prec','mean_prec','min_GWS','differ_GWS','mean_SM','mean_temp','min_temp']
    h5_columns = ['su_id','max_prec','mean_prec','max_GWS','mean_GWS','min_GWS',
              'differ_GWS','su_id1','max_SM','mean_SM','max_temp','mean_temp','min_temp']

    dt_points = node_label_data.columns
    date_list = generate_dt_points('20170416','20221222')
    graph_num = len(date_list)
    dy_node_h5 = h5py.File('../STGCN-WVAN/data-su/dynamic_features2.h5', 'r')
    print(date_list[0])
    window_size = 30

    dy_node_dfs = []
    for t in range(graph_num):
        dy_dt = str(date_list[t])+'/features'
        tmp_dy_arr = dy_node_h5[dy_dt][:]
        tmp_dy_df = pd.DataFrame(tmp_dy_arr,columns = h5_columns)
        dy_node_dfs.append(tmp_dy_df)

    node_label_df = pd.DataFrame(columns = date_list)  #label always from 20170416
    for t in range(graph_num):
        dy_dt = 'D'+str(date_list[t])
        node_label_df[date_list[t]] = node_label_data[dy_dt]
        
    node_label_arr1 = np.array(node_label_df).transpose()   #(23140,174)
    data_array = np.array([df[dy_fe_name].values for df in dy_node_dfs])

    #factors[1:], labels[:-1]
    dy_arr = data_array[1:]
    label_in = node_label_arr1[:-1]
    label_in_ex = np.expand_dims(label_in,axis = -1)
    
    data_array = np.concatenate((dy_arr, label_in_ex), axis=-1)
    print(data_array.shape)
    dtc_arr = data_array[0,:,:]
    print(dtc_arr.shape)
    for i in range(1,data_array.shape[0]):
        dtc_arr = np.vstack((dtc_arr, data_array[i,:,:]))
        
    ss_x = StandardScaler()
    nor_dy_fes_arr = ss_x.fit_transform(dtc_arr)  #181*23140,13
    times_nor_fe = nor_dy_fes_arr.reshape(data_array.shape[0],23140,len(dy_fe_name)+1)

    list_nor_fe_dfs = []
    for i in range(times_nor_fe.shape[0]):
        t_fe = pd.DataFrame(times_nor_fe[i,:,:])
        list_nor_fe_dfs.append(t_fe)
    dy_fe_series = transform_time_series(list_nor_fe_dfs, window_size)
    print(dy_fe_series.shape)  #timepoints,num_fes,look_backs, num_nodes
    ##################################################################
    
    node_label_df = pd.DataFrame(columns = date_list[30:])  #label always from 20170416
    for t in range(30,graph_num):
        dy_dt = 'D'+str(date_list[t])
        #print(dy_dt)
        node_label_df[date_list[t]] = node_label_data[dy_dt]
        
    node_label_arr1 = np.array(node_label_df) #(23140,144)
    node_label_arr1 = np.where(node_label_arr1<0, 0,node_label_arr1)
    node_label_arr2 = np.log(node_label_arr1+1)/4
    #node_label_arr2 = node_label_arr1/10
    nor_label = node_label_arr2.transpose()  # ()

    ss_y = StandardScaler()
    print(nor_label.shape)
    ########################################################
    
    len_val = round(nor_label.shape[0] * 0.08)
    print(len_val)
    len_train = round(nor_label.shape[0] * 0.75)

    '''
    device = (
    torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )'''
    nor_label = torch.Tensor(nor_label).to(device)
    dy_fe_series = torch.Tensor(dy_fe_series).to(device)
    #nor_label = torch.from_numpy(nor_label.astype('float16')).to(device)
    #dy_fe_series = torch.from_numpy(dy_fe_series.astype('float16')).to(device)
    G = create_base_graph()
    G = G.to(device)

    x_train = dy_fe_series[:len_train]
    x_val = dy_fe_series[len_train : len_train + len_val]
    x_test = dy_fe_series[len_train + len_val :]

    print(x_train.shape,x_val.shape,x_test.shape)

    y_train = nor_label[:len_train]
    y_val = nor_label[len_train : len_train + len_val]
    y_test = nor_label[len_train + len_val :]

    print(y_train.shape,y_val.shape,y_test.shape)
    
    #Stastic node features
    static_node_data = pd.read_csv('../STGCN-WVAN/data-su/node data_ERA5.csv')
    static_node_data.head()
    #12 statics features
    static_fe_names = ['std_dem', 'avg_slope', 'std_slope','avg_aspect','std_aspect',
                       'avg_plan_c','std_plan_c', 'avg_prof_c','std_prof_c',  'dis2river',
                       'land cover', 'dis2faults']
    ss_x2 = StandardScaler()
    nor_sta_fes = ss_x2.fit_transform(static_node_data[static_fe_names])
    nor_sta_fes = torch.Tensor(nor_sta_fes).to(device)
    #nor_sta_fes = torch.from_numpy(nor_sta_fes.astype('float16')).to(device)
    print(nor_sta_fes.shape)
    
    #13 edge_features
    edge_fe_name = ['area_ratio','centre distance', 'aspect angle differ', 'elevation differ']
    ss_ex = StandardScaler()
    nor_edge_fes = ss_ex.fit_transform(edge_data[edge_fe_name])
    nor_edge_fes = torch.Tensor(nor_edge_fes).to(device)
    
    
    return ss_x,ss_y,x_train,y_train,x_val,y_val,x_test,y_test,nor_sta_fes,nor_edge_fes,G
