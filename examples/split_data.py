#!/usr/bin/env python
# coding: utf-8

# ## Import libraries

# In[1]:


import os
import json
import glob
from tqdm import tqdm as tqdm
import sklearn


# Note: Rerun this notebook if we want to generate split data with all Retweets. Just change `remove_RT=False` before rerun.
# 
# **Warning**: When rerun, it'll overwrite the existing 5-fold training/testing data

# In[2]:


remove_RT = True
use_majing_split = True
data_mode = "hierarchical" # either "hierarchical" or "linear"
get_full_data_path = lambda year: f"../data/processed_data/{data_mode}_structure/twitter{year}/full_data/compiled_data_removeRT_keepUsername.json" if remove_RT                     else f"../data/processed_data/{data_mode}_structure/twitter{year}/full_data/compiled_data.json"
get_split_train_path = lambda year, split: f"../data/processed_data/{data_mode}_structure/twitter{year}/split_data/split_{split}/train.json"
get_split_test_path = lambda year, split: f"../data/processed_data/{data_mode}_structure/twitter{year}/split_data/split_{split}/test.json"

get_nfold_majing = lambda mode, year, split: f"../data/nfold/RNN{mode}Set_Twitter{year}{split}_tree.txt"


# ### Helper functions

# In[3]:


import pathlib
def write_to_file(content, output_file_location):
    try:
        with open(output_file_location, 'w', errors='ignore', encoding="utf-8") as f:
            f.write(content)
    except:
        path_folder = pathlib.Path(output_file_location).parent
        pathlib.Path(path_folder).mkdir(parents=True, exist_ok=True)
        with open(output_file_location, 'w', errors='ignore') as f:
            f.write(content)


# In[4]:


def concat_string(strings):
    return ' '.join(strings)


# ### Split data

# In[5]:


from sklearn.model_selection import KFold
import random
    
def split_data(year, split_method):
    compiled_data_path = get_full_data_path(year)
    # Read json file
    with open(compiled_data_path, encoding="utf-8") as f:
        lines = f.readlines()
        full_data = [json.loads(line) for line in lines]
    
        for split_num, train_data, test_data in split_method(full_data, year):
            # Randomize order for test data 
#             random.shuffle(test_data) #shuffle method
            
            # Output file paths
            train_data_output_path = get_split_train_path(year, split_num)
            test_data_output_path = get_split_test_path(year, split_num)

            # Output file paths (small)
            train_data_output_small_path = train_data_output_path.replace("train.json", "train_small.json") 
            test_data_output_small_path = test_data_output_path.replace("test.json", "test_small.json")
        
            # Write to json
            write_to_file('\n'.join([json.dumps(data) for data in train_data]), train_data_output_path)
            write_to_file('\n'.join([json.dumps(data) for data in test_data]), test_data_output_path)

            write_to_file('\n'.join([json.dumps(data) for data in train_data[:32]]), train_data_output_small_path)
            write_to_file('\n'.join([json.dumps(data) for data in test_data[:32]]), test_data_output_small_path)
            
            # Write to tsv
            dev_data_output_path = test_data_output_path.replace("test", "dev")
            json_files = [train_data_output_path, test_data_output_path, dev_data_output_path, train_data_output_small_path, test_data_output_small_path]
#             contents = [train_data, test_data[:len(test_data)//2], test_data[len(test_data)//2:], train_data[:32], test_data[:32]]
            contents = [train_data, test_data, test_data, train_data[:32], test_data[:32]]
            tsv_files = [x.replace('json', 'tsv') for x in json_files]
            
            for filepath, content in zip(tsv_files, contents):
                if data_mode=="linear":
                    write_to_file('\n'.join([f'{data["tweets"]}\t{data["label"]}' for data in content]), filepath)
                elif data_mode == "hierarchical":
                    header = "index\t#1 Label\t#2 String\t#2 String\n"
                    content = '\n'.join([f'{i}\t{data["label"]}\t{data["tweets"][0]}\t{concat_string(data["tweets"][1:])}' for i, data in enumerate(content)])
                    write_to_file(header+content, filepath)
                else:
                    print("Unrecognizable mode. Please make sure it's either 'linear' or 'hierarchical'")
                
             
'''Approach 1: Split randomly'''
def random_split(full_data, year):
    k_fold=5
    kf = KFold(n_splits=k_fold, shuffle=True)
    
    split_num=-1
    for train_index, test_index in tqdm(kf.split(full_data)):
        split_num += 1
        
        train_data = [full_data[i] for i in train_index]
        test_data = [full_data[i] for i in test_index]
                
        yield split_num, train_data, test_data
        

"""Approach 2: Split according to MaJing's split"""
def maJing_split(full_data, year):
    hashmap = create_hashmap_for_full_data(full_data)
    
    k_fold = 5
    
    for split_num in tqdm(range(k_fold)):
        # get training data
        train_data = get_data("train", year, split_num, hashmap)
        test_data = get_data("test", year, split_num, hashmap)
        
        yield split_num, train_data, test_data

def create_hashmap_for_full_data(full_data):
    hashmap = {x['source_id']:x for x in full_data}
    return hashmap

def get_data(mode, year, split, hashmap):
    data_input_path = get_nfold_majing(mode, year, split)
    with open(data_input_path) as f:
        ids = [x.strip() for x in f.readlines()]
    data = [hashmap[id] for id in ids if id in hashmap]
    return data
        
        


# In[6]:


years = ['15', '16']
split_method = maJing_split if use_majing_split else random_split
# split_method = random_split
for year in years:
    split_data(year, split_method)

    
