'''
    author: Peijie Sun
    e-mail: sun.hfut@gmail.com 
    released date: 04/18/2019
'''

from collections import defaultdict
import numpy as np
from time import time
import random
import pickle, os

class DataModule():
    def __init__(self, conf, filename):
        self.conf = conf
        self.data_dict = {}
        self.terminal_flag = 1
        self.filename = filename
        self.index = 0 # slicing index for batch
        # added by JWU
        self.item_idx_encode = pickle.load(open(os.path.join(os.getcwd(), 'data/%s'%self.conf.data_name, 'item_idx_encode.p'), 'rb'))
        self.full_eigenvector = pickle.load(open(os.path.join(os.getcwd(), 'data/%s'%self.conf.data_name, 'eigen_vector.p'), 'rb'))
        self.edge_score_min = (np.abs(self.full_eigenvector).min())**2
        self.edge_score_max = (np.abs(self.full_eigenvector).max())**2

###########################################  Initalize Procedures ############################################
    def prepareModelSupplement(self, model):
        data_dict = {}
        if 'CONSUMED_ITEMS_SPARSE_MATRIX' in model.supply_set: # True
            self.generateConsumedItemsSparseMatrix()
            data_dict['CONSUMED_ITEMS_INDICES_INPUT'] = self.consumed_items_indices_list
            data_dict['CONSUMED_ITEMS_VALUES_INPUT'] = self.consumed_items_values_list
        if 'SOCIAL_NEIGHBORS_SPARSE_MATRIX' in model.supply_set: # True
            self.readSocialNeighbors()
            self.generateSocialNeighborsSparseMatrix()
            data_dict['SOCIAL_NEIGHBORS_INDICES_INPUT'] = self.social_neighbors_indices_list
            data_dict['SOCIAL_NEIGHBORS_VALUES_INPUT'] = self.social_neighbors_values_list
        return data_dict

    def initializeRankingTrain(self):
        self.readData()
        self.arrangePositiveData()
        self.generateTrainNegative()

    def initializeRankingVT(self):
        self.readData()
        self.arrangePositiveData()
        self.generateTrainNegative()

    def initalizeRankingEva(self):
        self.readData()
        self.getEvaPositiveBatch()
        self.generateEvaNegative()

    def linkedMap(self):
        self.data_dict['USER_LIST'] = self.user_list
        self.data_dict['ITEM_LIST'] = self.item_list
        self.data_dict['LABEL_LIST'] = self.labels_list
        # add by JWU
        self.data_dict['EDGE_SCORE'] = self.edge_score
    
    def linkedRankingEvaMap(self):
        self.data_dict['EVA_USER_LIST'] = self.eva_user_list
        self.data_dict['EVA_ITEM_LIST'] = self.eva_item_list    

###########################################  Ranking ############################################
    def readData(self):
        f = open(self.filename) # e.g. yelp.train.rating
        total_user_list = set()
        hash_data = defaultdict(int)
        for _, line in enumerate(f):
            arr = line.split("\t") # user, item, rating(1)
            hash_data[(int(arr[0]), self.item_idx_encode[int(arr[1])])] = 1 # shift item index
            total_user_list.add(int(arr[0]))
        self.total_user_list = list(total_user_list) # [user_1, ... ,user_n]
        self.hash_data = hash_data # {(user, item): 1}        
    
    def arrangePositiveData(self):
        positive_data = defaultdict(set)
        total_data = set()
        hash_data = self.hash_data # connection between user and item {(user, item): 1}
        for (u, i) in hash_data:
            total_data.add((u, i))
            positive_data[u].add(i)
        self.positive_data = positive_data # true connection as {user:{item_1, ... ,item_n}
        self.total_data = len(total_data) # total number of edges
    
    '''
        This function designes for the train/val/test negative generating section
    '''
    def generateTrainNegative(self):
        num_items = self.conf.num_items
        num_negatives = self.conf.num_negatives
        negative_data = defaultdict(set)
        total_data = set()
        hash_data = self.hash_data
        for (u, i) in hash_data:
            total_data.add((u, i))
            for _ in range(num_negatives):
                # sample an item j until it's not trully connected to user u
                j = np.random.randint(num_items)+self.conf.num_users
                while (u, j) in hash_data:
                    j = np.random.randint(num_items)+self.conf.num_users
                negative_data[u].add(j) # add the sampled fake connection
                total_data.add((u, j))
        self.negative_data = negative_data
        self.terminal_flag = 1
    
    '''
        This function designes for the positive data in rating evaluate section
    '''
    def getEvaPositiveBatch(self):
        hash_data = self.hash_data
        user_list = []
        item_list = []
        index_dict = defaultdict(list)
        index = 0
        for (u, i) in hash_data:
            user_list.append(u)
            item_list.append(i)
            index_dict[u].append(index) # index for the (user, item) tuple
            index = index + 1
        self.eva_user_list = np.reshape(user_list, [-1, 1])
        self.eva_item_list = np.reshape(item_list, [-1, 1])
        self.eva_index_dict = index_dict
    
    '''
        This function designes for the negative data generation process in rating evaluate section
    '''
    def generateEvaNegative(self):
        hash_data = self.hash_data
        total_user_list = self.total_user_list
        num_evaluate = self.conf.num_evaluate
        num_items = self.conf.num_items
        eva_negative_data = defaultdict(list)
        for u in total_user_list:
            for _ in range(num_evaluate):
                j = np.random.randint(num_items)+self.conf.num_users
                while (u, j) in hash_data:
                    j = np.random.randint(num_items)+self.conf.num_users
                eva_negative_data[u].append(j)
        self.eva_negative_data = eva_negative_data
    
    ####### Followings are for training
    '''
        This function designes for the training process
    '''
    def getTrainRankingBatch(self):
        positive_data = self.positive_data
        negative_data = self.negative_data
        total_user_list = self.total_user_list
        index = self.index # slicing index for batch
        batch_size = self.conf.training_batch_size

        user_list, item_list, labels_list = [], [], []
        # add by JWU
        edge_score = []
        
        if index + batch_size < len(total_user_list):
            target_user_list = total_user_list[index:index+batch_size]
            self.index = index + batch_size
        else:
            target_user_list = total_user_list[index:len(total_user_list)]
            self.index = 0
            self.terminal_flag = 0

        for u in target_user_list:
            user_list.extend([u] * len(positive_data[u]))
            item_list.extend(list(positive_data[u]))
            labels_list.extend([1] * len(positive_data[u]))
            # add by JWU
            user_item_edge_score = self.full_eigenvector[u]*self.full_eigenvector[list(positive_data[u])]
            user_item_edge_score = (user_item_edge_score- self.edge_score_min)/(self.edge_score_max - self.edge_score_min)
            edge_score.extend(user_item_edge_score)
            
            user_list.extend([u] * len(negative_data[u]))
            item_list.extend(list(negative_data[u]))
            labels_list.extend([0] * len(negative_data[u]))
            # add by JWU
            user_item_edge_score = self.full_eigenvector[u]*self.full_eigenvector[list(negative_data[u])]
            user_item_edge_score = (user_item_edge_score- self.edge_score_min)/(self.edge_score_max - self.edge_score_min)
            edge_score.extend(user_item_edge_score)
        
        self.user_list = np.reshape(user_list, [-1, 1])
        self.item_list = np.reshape(item_list, [-1, 1])
        self.labels_list = np.reshape(labels_list, [-1, 1])
        # add by JWU
        self.edge_score = np.reshape(edge_score, [-1,1])
    
    '''
        This function designes for the val/test section, compute loss
    '''
    def getVTRankingOneBatch(self):
        positive_data = self.positive_data
        negative_data = self.negative_data
        total_user_list = self.total_user_list
        user_list = []
        item_list = []
        labels_list = []
        # add by JWU
        edge_score = []
        for u in total_user_list:
            user_list.extend([u] * len(positive_data[u]))
            item_list.extend(positive_data[u])
            labels_list.extend([1] * len(positive_data[u]))
            # add by JWU
            user_item_edge_score = self.full_eigenvector[u]*self.full_eigenvector[list(positive_data[u])]
            user_item_edge_score = (user_item_edge_score- self.edge_score_min)/(self.edge_score_max - self.edge_score_min)
            edge_score.extend(user_item_edge_score)
            
            user_list.extend([u] * len(negative_data[u]))
            item_list.extend(negative_data[u])
            labels_list.extend([0] * len(negative_data[u]))
            # add by JWU
            user_item_edge_score = self.full_eigenvector[u]*self.full_eigenvector[list(negative_data[u])]
            user_item_edge_score = (user_item_edge_score- self.edge_score_min)/(self.edge_score_max - self.edge_score_min)
            edge_score.extend(user_item_edge_score)
        
        self.user_list = np.reshape(user_list, [-1, 1])
        self.item_list = np.reshape(item_list, [-1, 1])
        self.labels_list = np.reshape(labels_list, [-1, 1])
        # add by JWU
        self.edge_score = np.reshape(edge_score, [-1,1])

    '''
        This function designs for the rating evaluate section, generate negative batch
    '''
    def getEvaRankingBatch(self):
        batch_size = self.conf.evaluate_batch_size
        num_evaluate = self.conf.num_evaluate
        eva_negative_data = self.eva_negative_data
        total_user_list = self.total_user_list
        index = self.index
        terminal_flag = 1
        total_users = len(total_user_list)
        user_list = []
        item_list = []
        if index + batch_size < total_users:
            batch_user_list = total_user_list[index:index+batch_size]
            self.index = index + batch_size
        else:
            terminal_flag = 0
            batch_user_list = total_user_list[index:total_users]
            self.index = 0
        for u in batch_user_list:
            user_list.extend([u]*num_evaluate)
            item_list.extend(eva_negative_data[u])
        self.eva_user_list = np.reshape(user_list, [-1, 1])
        self.eva_item_list = np.reshape(item_list, [-1, 1])
        return batch_user_list, terminal_flag

################################################ Supplement for Sparse Computation ##################################
    def readSocialNeighbors(self, friends_flag=1):
        social_neighbors = defaultdict(set)
        links_file = open(self.conf.links_filename)
        for _, line in enumerate(links_file):
            tmp = line.split('\t')
            u1, u2 = int(tmp[0]), int(tmp[1])
            social_neighbors[u1].add(u2)
            if friends_flag == 1:            # undirected or not
                social_neighbors[u2].add(u1)
        self.social_neighbors = social_neighbors

    '''
        Generate Social Neighbors Sparse Matrix Indices and Values
    '''
    def generateSocialNeighborsSparseMatrix(self):
        social_neighbors = self.social_neighbors # social_neighbors: {user_i:{user_m, ..., user_n}} 
        social_neighbors_indices_list = []
        social_neighbors_values_list = []
        social_neighbors_dict = defaultdict(list)
        for u in social_neighbors:
            social_neighbors_dict[u] = sorted(social_neighbors[u]) # sort the neighbors of user u
            
        user_list = sorted(list(social_neighbors.keys()))
        for user in user_list:
            for friend in social_neighbors_dict[user]:
                social_neighbors_indices_list.append([user, friend])
                social_neighbors_values_list.append(1.0/len(social_neighbors_dict[user])) # 1/number of neighbors
        self.social_neighbors_indices_list = np.array(social_neighbors_indices_list).astype(np.int64) # a K*2 matrix
        self.social_neighbors_values_list = np.array(social_neighbors_values_list).astype(np.float32) # a K*1 vector
    
    '''
        Generate Consumed Items Sparse Matrix Indices and Values
    '''
    def generateConsumedItemsSparseMatrix(self):
        positive_data = self.positive_data     # positive_data: {user_i:{item_m, ..., item_n}}, where item_j has shifted index
        consumed_items_indices_list = []
        consumed_items_values_list = []
        consumed_items_dict = defaultdict(list)
        for u in positive_data:
            consumed_items_dict[u] = sorted(positive_data[u])
        user_list = sorted(list(positive_data.keys()))
        for u in user_list:
            for i in consumed_items_dict[u]:
                consumed_items_indices_list.append([u, i])
                consumed_items_values_list.append(1.0/len(consumed_items_dict[u])) # 1/total number of visited items
        self.consumed_items_indices_list = np.array(consumed_items_indices_list).astype(np.int64) # a K*2 matrix
        self.consumed_items_values_list = np.array(consumed_items_values_list).astype(np.float32) # a K*1 vector
