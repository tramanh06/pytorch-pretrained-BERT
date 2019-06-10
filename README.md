# PyTorch Pretrained BERT: Extended with Heatmap of attention weights

This is a fork from the HuggingFace's Pytorch implementation of BERT. Please see the original README at https://github.com/huggingface/pytorch-pretrained-BERT 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

Install prerequisites as specified in the [original README](https://github.com/huggingface/pytorch-pretrained-BERT).

### Dataset location

Copy the Twitter15/16 dataset under the project folder `./data/raw_data`

Below is the content under data folder:

```
--data/
----raw_data/
------twitter15/
------twitter16/
------tweet_details.json
------user_details.json
```

## Preprocessing

1. Navigate to `examples/` folder

### For Linear model

2. Run `preprocess_rumdect_data_concat_tweets.py` to preprocess files for LinearBERT model.

```
python preprocess_rumdect_data_concat_tweets.py
```

3. In `split_data.py`, change variable `data_mode = linear`

4. Run `split_data.py`. 

```
python split_data.py
```

This will produce data for 5 splits, under `data/processed_data/linear_structure/twitter15/split_data/`

### For Hierarchical model

2. Run `preprocess_rumdect_data_hierarchical.py` to preprocess files for HierarchicalBERT model.

```
python preprocess_rumdect_data_hierarchical.py
```

3. In `split_data.py`, change variable `data_mode = hierarchical`

4. Run `split_data.py`

```
python split_data.py
``` 

This will produce data for 5 splits, under `data/processed_data/hierarchical_structure/twitter15/split_data/`


## Training BERT Classifier

This section will list commands to train the classifier for Linear and Hierarchical model of the Twitter15/16 dataset.

### Training LinearBERT 

Run the following to train linear model, first fold. 

Change `$DATA_DIR` to the correct location.

```
export DATA_DIR=/opt/src/rumor_lstm/data/processed_data/linear_structure/twitter15/split_data

python run_classifier.py \
  --task_name twitter-1516-linear \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $DATA_DIR/split_0/ \
  --bert_model bert-base-uncased \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 8.0 \
  --output_dir ../logs/twitter15_split_0/
```

To train subsequence folds (fold 1-4), at the python command, change parameter `data_dir` and `output_dir` accordingly.

### Training HierarchicalBERT

Change `$DATA_DIR` to the correct location.

```
export DATA_DIR=/opt/src/rumor_lstm/data/processed_data/hierarchical_structure/twitter15/split_data

python run_classifier.py \
  --task_name twitter-1516-2segments \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $DATA_DIR/split_0/ \
  --bert_model bert-base-uncased \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 8.0 \
  --output_dir ../logs/twitter15_split_0/
```

To train subsequence folds (fold 1-4), at the python command, change parameter `data_dir` and `output_dir` accordingly.


## Interpreting Attention weights at the last layer

The following code will save attention weights of the last layer of BERT, while summing up all attention heads of the last layer.

1. In `Interpret_BERT.py`, specify the trained model location for `model_fn`.

```
model_fn = '../logs/hierarchical_models/twitter15_split_0/pytorch_model.bin'   # Example. The model location is to be changed accordingly
```

2. In `Interpret_BERT.py`, specify the directory of the test data. Note that if in step 1, the model chosen is coming from **split_0**, it is 
important to use the corresponding split for the data.

```
# Example. The data location is to be changed accordingly
data_dir = 'C:/git/rumor-lstm/data/processed_data/hierarchical_structure/twitter15/split_data/split_0/'   
```

3. In `Interpret_BERT.py`, change variable `task_name`'s value to `"twitter-1516-2segments"` if running hierarchical model.
Change to `"twitter-1516-linear"` to run linear model.

```
task_name = "twitter-1516-2segments"   # value can be "twitter-1516-2segments" or "twitter-1516-linear"
```

4. Run `Interpret_BERT.py`

```
python Interpret_BERT.py
```

The heatmaps are saved as `.png` images, under folder `./heatmap_output/`. 
File naming convention is of the form `{test-Index}_{ground-truth}_{actual-output}.png`.
For example, `6_non-rumor_unverified.png` means the test sample # 6 has "Non-rumor" as label, but is predicted as "Unverified".
