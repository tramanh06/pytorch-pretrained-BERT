#!/usr/bin/env python
# coding: utf-8

'''
This file will process Ma Jing's Twitter1516 dataset into tsv files to be used by `run_classifier.py`

The files produced here are used by the LinearBERT model. 
In this model, all tweets are sorted according to timestamp, 
then concatenated all contents into 1 document. 
'''

# In[27]:


import json
import re
from tqdm import tqdm as tqdm
import preprocessor as p 
import spacy
# Run `python -m spacy download en` on first run
nlp = spacy.load("en_core_web_sm", disable = ["parser", "tagger", "ner"])


# In[32]:


remove_RT = True
get_tree_path = lambda year: "../data/raw_data/twitter{0}/tree/".format(year)
get_label_file_path = lambda year: f"../data/raw_data/twitter{year}/label.txt"
get_json_out_file_path = lambda year: f"../data/processed_data/linear_structure/twitter{year}/full_data/compiled_data_removeRT_keepUsername.json" if remove_RT \
    else f"../data/processed_data/linear_structure/twitter{year}/full_data/compiled_data.json"

tweet_details_path = '../data/raw_data/tweet_details.json'


# In[33]:


label_mapping = {"false" : 0, 
           "true" : 1, 
           "unverified" : 2, 
           "non-rumor" : 3
          }


# In[29]:


def tokenize_text(text):
    return ' '.join([token.text for token in nlp.tokenizer(text.lower())])
text2 = "EMOJI$ We need real Democracy; which we don't have now! We need to get rid of Capitalism![ Friend Of Russia] Also follow new MENTION$"
tokenize_text(text2)


# In[30]:


def preprocess(text):
    text = text.replace("'", "")
    
    text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "URL", text) # Replace urls with special token
    text = text.replace("@", "")
    text = text.replace("#", "")
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = text.replace("&amp;", "")
    text = text.replace("&gt;", "")
    text = text.replace("\"", "")
    text = text.replace(".", "")
    text = text.replace(",", "")
    text = text.replace("(", "")
    text = text.replace(")", "")
    
    text = ' '.join([token.text for token in nlp.tokenizer(text)])
    text = p.tokenize(text)
    
    text = text.lower()
    
    return text
    

# In[34]:


def furnish_tweet(tweet_id, user_id, lookup_file, timedelay):
    tweet = lookup_file[tweet_id] if tweet_id in lookup_file else None
    if not tweet:
        return 
    return {'id': tweet['id_str'],
           'text': preprocess(tweet['text']),
           'user_id': user_id,
           'timedelay': timedelay}


# In[35]:


def find_index_of_root_line(lines_with_numbers_only):
    index = next((i for i in range(len(lines_with_numbers_only)) if 'ROOT' in lines_with_numbers_only[i][0]))
    return index


# In[36]:


def process_line(line):
    try:
        pattern = r"\['(\w+)', '(\w+)', '(-?\d+\.\d+)'\]->\['(\d+)', '(\d+)', '(-?\d+\.\d+)'\]"
    #    e.g. line = "['972651', '80080680482123777', '0.0']->['12735762', '80080680482123777', '1.8']"
        m = re.match(pattern, line)
        numbers = m.groups()
        return numbers
    except:
        print("Exception occured. ")
        print(f"Line={line}")
    


# In[37]:


def process_tree_file(lines):
    numbers_only = [process_line(line) for line in lines]
    sort_by_timestamp = sorted(numbers_only, key=lambda tup: float(tup[-1]))  # Sort by timestamp which is the last item in numbers_only
    root_line = find_index_of_root_line(sort_by_timestamp)
    sort_by_timestamp = sort_by_timestamp[root_line:]  # Remove lines that come before root, chronologically
    if remove_RT:
        res, seen_so_far = [], set()
        for line in sort_by_timestamp:
            tweet_id = line[4]
            if tweet_id not in seen_so_far:
                res.append(line)
                seen_so_far.add(tweet_id)
        assert len(sort_by_timestamp) >= len(res)
        sort_by_timestamp = res
        
    tweet_sequence = [furnish_tweet(x[4], x[3], lookup_file, x[-1]) for x in sort_by_timestamp]
    return ' '.join([x['text'] for x in tweet_sequence if x])


# In[38]:


with open(tweet_details_path, "r", encoding='utf-8') as f:
    lines = f.readlines()
    tweet_objects = [json.loads(x) for x in lines]
    lookup_file = {x['id_str']:x for x in tweet_objects}


# In[39]:


def get_tweet_sequence(source_id, year):
    tree_path = get_tree_path(year)
    tree_file = tree_path+source_id+".txt"
    with open(tree_file) as f:
        lines = f.readlines()
    tweet_sequence = process_tree_file(lines)
    return tweet_sequence

# source_id = "80080680482123777"
# year = '15'
# tweets = get_tweet_sequence(source_id, year)


# In[40]:


def write_to_file(content, output_file_location):
    try:
        with open(output_file_location, 'w', errors='ignore', encoding="utf-8") as f:
            f.write(content)
    except:
        path_folder = pathlib.Path(output_file_location).parent
        pathlib.Path(path_folder).mkdir(parents=True, exist_ok=True)
        with open(output_file_location, 'w', errors='ignore', encoding="utf-8") as f:
            f.write(content)


# In[41]:


import pathlib
years = ['15', '16']

for year in years:
    # Read labels file
    label_file_path = get_label_file_path(year)
    with open(label_file_path) as f:
        lines = f.readlines()
        labels = [x.strip().split(":") for x in lines]
        
    # Read tree file
    compiled_data = []
    for label, source_id in tqdm(labels):
        tweets = get_tweet_sequence(source_id, year)
        if tweets:
            compiled_data.append({'source_id': source_id, 
                              'label': label_mapping[label],
                              'tweets': tweets
                              })

    
    # Write to Json file
    json_outfile = get_json_out_file_path(year)
    pathlib.Path(json_outfile).parent.mkdir(parents=True, exist_ok=True) # Create dir if folder not exists
    write_to_file('\n'.join([json.dumps(x) for x in compiled_data]), json_outfile)
    


# In[ ]:




