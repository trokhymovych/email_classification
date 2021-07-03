import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from tqdm import tqdm
# from implicit.recommender_base  import RecommenderBase
# from implicit.bpr import BayesianPersonalizedRanking
# from implicit.approximate_als import NMSLibAlternatingLeastSquares
# from implicit.approximate_als import AnnoyAlternatingLeastSquares

import copy
import pickle
import sys

import warnings
def warn(*args, **kwargs): pass
warnings.warn = warn

def make_implicit(x, min_rating):
        if x >= min_rating:
            return 1
        else:
            return 0


def fit_coder(dataset, user_var_name, item_var_name, rating_var_name):
    """
        Function for fitting encoder based on dataset`s users and items, to transform them in numeric form
        :param dataset: pandas dataframe
        :param user_var_name: str - name of column of users,
        :param item_var_name: str - name of column of items,
        :param rating_var_name: str - name of column of ratings,
        :return:
        mapping_dict - used by code function to encode dataset in appropriate form,
        #         inv_mapping_dict - used by code function to decode dataset to row format
        """
    user_var_name = str(user_var_name)
    item_var_name = str(item_var_name)
    rating_var_name = str(rating_var_name)
    users = dataset[user_var_name].unique()
    items = dataset[item_var_name].unique()
    items_dict = {key: value for key, value in zip(items,range(0,len(items)))}
    users_dict = {key: value for key, value in zip(users,range(0,len(users)))}
    mapping_dict = {user_var_name:users_dict, item_var_name:items_dict}
    inv_map_user = {v: k for k, v in users_dict.items()}
    inv_map_item = {v: k for k, v in items_dict.items()}
    inv_mapping_dict = {user_var_name:inv_map_user, item_var_name:inv_map_item}
    return mapping_dict, inv_mapping_dict

def code(test, user_var_name, item_var_name, rating_var_name, mapping_dict):
    """
        Function for encoding/decoding dataset based mapping_dict
        It transforms dataset to friendly, appropriate for Recommender_surprise.py form or to the raw form
        :param test: pandas dataframe,
        :param user_var_name: str - name of column of users,
        :param item_var_name: str - name of column of items,
        :param rating_var_name: str - name of column of ratings,
        :param mapping_dict: - pretrained by fit_coder function dictionary that either encode(mapping dict) or decode(inv_mapping_dict) given dataset.
        :return: dataset - pandas dataframe in either raw or encoded format.
        """
    user_var_name = str(user_var_name)
    item_var_name = str(item_var_name)
    rating_var_name = str(rating_var_name)
    test[user_var_name] = test[user_var_name].map(mapping_dict[user_var_name])
    test[item_var_name] = test[item_var_name].map(mapping_dict[item_var_name])
    return test

class Implicit(object):

    def __init__(self):

        self.model = None
        self.trainset = None
        self.testset = None
        self.mapping_dict = None
        self.inv_mapping_dict = None
        self.table_csr = None
        self.user_items = None
        self.k = 5
        self.item_value_counts = None
        self.max_index_of_item = None
        self.max_index_of_user = None
        self.param = None
        self.default_param = {'factors': 100, 'regularization':0.01, 'iterations':15,'use_native':True, 'use_cg':True, 'use_gpu':False,'calculate_training_loss':False, 'num_threads':0}
           

    def load_dataset(self, path_to_dataset, sep=',', names = ['user', 'item', 'rating', 'timestamp']):
        """
        Upload dataframe from .csv file in such a format:
            1) No header
            2) ',' separator
            3) columns in order user, item, rating
        :param path_to_dataset:
        :return: nothing
        """
        try:
            return pd.read_csv(path_to_dataset, sep=sep, names=names)
        except:
            print("Unable to upload dataset")
            print("Unexpected error:", sys.exc_info()[0])
            raise
        


    def fit_trainset(self, raw_train_dataset):
        self.trainset = copy.deepcopy(raw_train_dataset)
        #self.trainset['rating'] = self.trainset['rating'].apply(make_implicit, min_rating=1) #
        
        self.mapping_dict, self.inv_mapping_dict = fit_coder(self.trainset, 'user', 'item', 'rating')
        self.trainset = code(copy.deepcopy(self.trainset), 'user','item','rating',self.mapping_dict)
        
        self.max_index_of_item = len(self.trainset.item.unique())
        self.max_index_of_user = len(self.trainset.user.unique())
        
        #table = self.trainset.pivot(index='item', columns='user', values='rating').fillna(0)
        
        row = self.trainset.item.values
        col = self.trainset.user.values
        data = self.trainset.rating.values
                
        self.table_csr = csr_matrix( (data,(row,col)), shape=(self.max_index_of_item,self.max_index_of_user) )
        
        self.user_items = self.table_csr.T.tocsr()
        self.item_value_counts = self.trainset.item.value_counts()
        
        
        
    def fit_testset(self, raw_test_dataset):
        if self.trainset is None:
            print('Firstly fit train')
        else:
            self.testset = copy.deepcopy(raw_test_dataset)

            #Handling unknown items and users
            new_test = code(copy.deepcopy(self.testset), 'user','item','rating',self.mapping_dict)

            ind_item = new_test[new_test.item.isnull()].index
            ind_user = new_test[new_test.user.isnull()].index

            unknown_items = self.testset.loc[ind_item,'item'].unique()
            unknown_users = self.testset.loc[ind_user,'user'].unique()

            new_item_dic = {key: value for key, value in zip(unknown_items,range(self.max_index_of_item, self.max_index_of_item+len(unknown_items)))}
            new_user_dic = {key: value for key, value in zip(unknown_users,range(self.max_index_of_user, self.max_index_of_user+len(unknown_users)))}

            inv_new_item_dic = {value: key for key, value in zip(unknown_items,range(self.max_index_of_item, self.max_index_of_item+len(unknown_items)))}
            inv_new_user_dic = {value: key for key, value in zip(unknown_users,range(self.max_index_of_user, self.max_index_of_user+len(unknown_users)))}

            self.mapping_dict['item'].update(new_item_dic)
            self.mapping_dict['user'].update(new_user_dic)

            self.inv_mapping_dict['item'].update(inv_new_item_dic)
            self.inv_mapping_dict['user'].update(inv_new_user_dic)

            self.testset = code(copy.deepcopy(self.testset), 'user','item','rating',self.mapping_dict)
    
    def set_k(self, k):
        self.k = int(k)
        
    def fit_model(self, dic_param = {}):
        if self.table_csr is None:
            print('Firstly fit trainset')
        else:
            d = copy.deepcopy(self.default_param)
            d.update(dic_param)
            self.param = d
            self.model = AlternatingLeastSquares(factors=d['factors'],regularization=d['regularization'],iterations=d['iterations'], use_native=d['use_native'],use_cg=d['use_cg'],use_gpu=d['use_gpu'],calculate_training_loss =d['calculate_training_loss'], num_threads=d['num_threads']) #dic_param
            self.model.fit(self.table_csr)
        
    def recommend_for_user(self, user, filter_already_liked_items):
        if self.max_index_of_user is None:
            print('Firstly fit_testset')
            return None
        
        if user < self.max_index_of_user:
            rec = self.model.recommend(user, self.user_items, self.k, filter_already_liked_items=filter_already_liked_items)
            return rec
        else:
            return None
    

    def recommend_json(self, filter_already_liked_items=True):
        if self.max_index_of_user is None:
            print('Firstly fit_testset')
            return None
        
        most_popular = self.item_value_counts[:self.k].index
        default_rec = [(most_popular[i], (int(self.k) -i)*0.01) for i in range(len(most_popular))]
        
        result = {}
        users = list(self.testset.user.unique())
        for i in tqdm(range(len(users))):
            user = users[i]
            rec = self.recommend_for_user(user, filter_already_liked_items)
            if rec is None: #new user in test
                rec = default_rec
            
            result[str(self.inv_mapping_dict['user'][user])] = [self.inv_mapping_dict['item'][item] for item, _ in rec] #make int
            
        return result
    
    def recommend(self, filter_already_liked_items=True):
        if self.max_index_of_user is None:
            print('Firstly fit_testset')
            return None
        
        most_popular = self.item_value_counts[:self.k].index
        default_rec = [(most_popular[i], (int(self.k) -i)*0.01) for i in range(len(most_popular))]
        
      
        result = []
        users = list(self.testset.user.unique())
        for user in tqdm(users):
            rec = self.recommend_for_user(user, filter_already_liked_items)
            if rec is None: #new user in test
                rec = default_rec
            
            items = [self.inv_mapping_dict['item'][item] for item, _ in rec]
            ratings = [score for _, score in rec]
            real_user = self.inv_mapping_dict['user'][user]
            users_list = [real_user]*int(self.k)
            
            res = list(zip(users_list, items, ratings))
            result.extend(res)
            
        return pd.DataFrame(result, columns = ['user', 'item', 'rating'])


#     def recommend(self, filter_already_liked_items=True):
#         if self.max_index_of_user is None:
#             print('Firstly fit_testset')
#             return None
        
#         most_popular = self.item_value_counts[:self.k].index
#         default_rec = [(most_popular[i], (int(self.k) -i)*0.01) for i in range(len(most_popular))]
        
#         result = pd.DataFrame(columns=['item','rating','user'])
#         users = list(self.testset.user.unique())
#         for i in tqdm(range(len(users))):
#             user = users[i]
#             rec = self.recommend_for_user(user, filter_already_liked_items)
#             if rec is None: #new user in test
#                 rec = default_rec
            
#             df = pd.DataFrame(rec, columns=['item','rating'])
#             df['user'] = [user]*len(df)
            
#             result = pd.concat([result, df])
            
#         result = result[['user','item','rating']]
#         output = code(copy.deepcopy(result), 'user','item','rating',self.inv_mapping_dict)
#         output.index = range(len(output))
        
#         return output
    
    def rank_for_user(self, user):
        if self.max_index_of_user is None:
            print('Firstly fit_testset')
            return None
        
        list_items = self.testset[self.testset.user == user].item
        items_to_rank = list_items[list_items < self.max_index_of_item].values
        items_to_end = list_items[list_items >= self.max_index_of_item].values

        res = []

        if user >= self.max_index_of_user:
            list_to_sort = []
            for item in items_to_rank:
                list_to_sort.append((round(self.item_value_counts[item]*0.001,3),item))

            for item in items_to_end:
                list_to_sort.append((0,item))

            list_to_sort.sort(reverse=True)
            res = [(t[1], t[0]) for t in list_to_sort]
        else:
            res = self.model.rank_items(user, self.user_items,selected_items=items_to_rank)
            for item in items_to_end:
                res.append((item, 0))
        return res
        
        
    
    def rank(self):
        if self.max_index_of_user is None:
            print('Firstly fit_testset')
            return None
        
        result = pd.DataFrame(columns=['item','rating','user'])
        
        users = list(self.testset.user.unique())
        for i in tqdm(range(len(users))):
            user = users[i]
            res = self.rank_for_user(user)
            df = pd.DataFrame(res, columns=['item','rating'])
            df['user'] = [user]*len(df)
            
            result = pd.concat([result, df])
            
        result = result[['user','item','rating']]
        output = code(copy.deepcopy(result), 'user','item','rating',self.inv_mapping_dict)
        output.index = range(len(output))
        
        return output
    
    def dump_model(self, filename = 'dumped_file'):
        """
        Saving the model for further using.
        :param filename: str - path and name of file to save.
        :return:
        """
        if (self.model is None) | (self.trainset is None):
            print('Unable to dump model')
            print('Please firstly fit train dataset and train model')
        else:
            dump_obj = {'model': self.model,
                        'trainset': self.trainset,
                        'mapping_dict': self.mapping_dict,
                        'inv_mapping_dict': self.inv_mapping_dict,
                        'table_csr': self.table_csr,
                        'user_items': self.user_items,
                        'k': self.k,
                        'item_value_counts': self.item_value_counts,
                        'max_index_of_item': self.max_index_of_item, 
                        'max_index_of_user': self.max_index_of_user,
                        'param': self.param
                        }
            pickle.dump(dump_obj, open(filename, 'wb'),
                        protocol=pickle.HIGHEST_PROTOCOL)
            print('Model has succesfuly been dumped!')
            
    

    def load_model(self,filename = 'dumped_file'):
        """
        Function to load ready to use, pre trained model from file.
        :param filename: str - path to the file with model
        :return: nothing
        """
        dump_obj = pickle.load(open(filename, 'rb'))
        self.model = dump_obj['model']
        self.trainset = dump_obj['trainset']
        self.mapping_dict = dump_obj['mapping_dict']
        self.inv_mapping_dict = dump_obj['inv_mapping_dict']
        self.table_csr = dump_obj['table_csr']
        self.user_items = dump_obj['user_items']
        self.k = dump_obj['k']
        self.item_value_counts = dump_obj['item_value_counts']
        self.max_index_of_item = dump_obj['max_index_of_item']
        self.max_index_of_user = dump_obj['max_index_of_user']
        self.param = dump_obj['param']
    
