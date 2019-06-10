'''This file is to preprocess data using ekphrasis. 
Author: Jianfei
'''

# coding: utf-8

# In[1]:


import json
import re
from tqdm import tqdm
import pathlib
import os

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

text_processor = TextPreProcessor(
	# terms that will be normalized
	normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
			   'time', 'url', 'date', 'number'],
	# terms that will be annotated
	annotate={"hashtag", "allcaps", "elongated", "repeated",
			  'emphasis', 'censored'},
	fix_html=True,  # fix HTML tokens

	# corpus from which the word statistics are going to be used
	# for word segmentation
	segmenter="twitter",

	# corpus from which the word statistics are going to be used
	# for spell correction
	corrector="twitter",

	unpack_hashtags=True,  # perform word segmentation on hashtags
	unpack_contractions=True,  # Unpack contractions (can't -> can not)
	spell_correct_elong=False,  # spell correction for elongated words

	# select a tokenizer. You can use SocialTokenizer, or pass your own
	# the tokenizer, should take as input a string and return a list of tokens
	tokenizer=SocialTokenizer(lowercase=True).tokenize,

	# list of dictionaries, for replacing tokens extracted from the text,
	# with other expressions. You can pass more than one dictionaries.
	dicts=[emoticons]
)

def furnish_tweet(tweet_id, user_id, lookup_file, timedelay):
    tweet = lookup_file[tweet_id] if tweet_id in lookup_file else None
    if not tweet:
        return {'id': '00000000',
                'text': 'tweet not found !',
                'user_id': user_id,
                'timedelay': timedelay}

    tokenization_flag = True
    if tokenization_flag:
        return {'id': tweet['id_str'],
           'text': ' '.join(text_processor.pre_process_doc(tweet['text'])),
           'user_id': user_id,
           'timedelay': timedelay}
    else:
        return {'id': tweet['id_str'],
           'text': tweet['text'],
           'user_id': user_id,
           'timedelay': timedelay}


# In[6]:


def find_index_of_root_line(lines_with_numbers_only):
    index = next((i for i in range(len(lines_with_numbers_only)) if 'ROOT' in lines_with_numbers_only[i][0]))
    return index


# In[7]:


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
    


# In[8]:


def process_tree_file(lines, lookup_file):
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

    tweet_content_sequence = [x['text'] for x in tweet_sequence if x]
    tweet_time_delay = [x['timedelay'] for x in tweet_sequence if x]
    return tweet_content_sequence, tweet_time_delay


# In[10]:


def get_tweet_sequence(source_id, year, lookup_file):
    tree_path = get_tree_path(year)
    tree_file = tree_path+source_id+".txt"
    with open(tree_file) as f:
        lines = f.readlines()
    tweet_sequence, timedelay_sequence = process_tree_file(lines, lookup_file)
    return tweet_sequence, timedelay_sequence

# source_id = "80080680482123777"
# year = '15'
# tweets = get_tweet_sequence(source_id, year)


# In[11]:


def write_to_file(content, output_file_location):
    path_folder = pathlib.Path(output_file_location).parent
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)

    with open(output_file_location, 'w', errors='ignore') as f:
        f.write(content)

if __name__ == '__main__':

    # In[2]:

    text = "🌟 We need real Democracy; which we do not have now! We need to get rid of Capitalism![ Friend Of Russia]\nAlso follow new @WinderDeyan"
    # p.tokenize(text)

    # In[3]:

    remove_RT = True
    get_tree_path = lambda year: "C:/git/rumor-lstm/data/raw_data/twitter{0}/tree/".format(year)
    get_label_file_path = lambda year: f"C:/git/rumor-lstm/data/raw_data/twitter{year}/label.txt"
    get_out_file_path = lambda \
        year: f"C:/git/rumor-lstm/data/processed_data/hierarchical_structure/twitter{year}/full_data/compiled_data_removeRT_jianfei.json" if remove_RT else f"C:/git/rumor-lstm/data/processed_data/hierarchical_structure/twitter{year}/full_data/compiled_data.json"

    tweet_details_path = 'C:/git/rumor-lstm/data/raw_data/tweet_details.json'

    # In[4]:

    label_mapping = {"false": 0,
                     "true": 1,
                     "unverified": 2,
                     "non-rumor": 3
                     }

    # In[5]:
    with open(tweet_details_path, "r", encoding='utf-8') as f:
        lines = f.readlines()
        tweet_objects = [json.loads(x) for x in lines]
        lookup_file = {x['id_str']: x for x in tweet_objects}


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
            tweets, timedelays = get_tweet_sequence(source_id, year, lookup_file)
            if tweets:
                compiled_data.append({'source_id': source_id,
                                      'label': label_mapping[label],
                                      'tweets': tweets,
                                      'time_delay': timedelays})
        outfile = get_out_file_path(year)
        write_to_file('\n'.join([json.dumps(x) for x in compiled_data]), outfile)

