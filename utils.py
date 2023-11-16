import os
import numpy as np
import pickle as pkl
from tqdm import tqdm
import copy

import torch
from torch.utils.data import DataLoader, Dataset

from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer


class CustomData(Dataset):
  '''
    Process raw data
  '''
  def __init__(self, file_dir):
      self.file = file_dir
      self.article_name = []
      self.questions = []
      self.answers = []
      self.q_diffi = []
      self.a_diffi = []
      self.article_path = []
      self.context = {} # only fill when load the dataset

      # get question answer pairs
      for div in ['S08', 'S09', 'S10']:
        skip = True
        qa_path = os.path.join(self.file, div, "question_answer_pairs.txt")
        with open(qa_path, 'rb') as f:
          for line in f:
            if skip:
              skip = False
              continue
            try:
              row = line.decode().split('\t')
            except:
              continue
            # print(row)
            if "NULL" in row:
              continue # if any feature does not exist -> skip

            context_file = self.file + "/" + div + "/"+ row[5][:-1] + ".txt" # path to the context file
            if not (os.path.exists(context_file) and os.path.isfile(context_file)): # otherwise context doesn't exist: invalid
              continue
            # check if context could be extracted
            try:
              with open(context_file, 'rb') as f:
                curr_context = f.read().decode().replace('\n',' ')
            except:
              continue

            self.article_name.append(row[0])
            self.questions.append(row[1])
            self.answers.append(row[2])
            self.q_diffi.append(row[3])
            self.a_diffi.append(row[4])
            self.article_path.append(div + "/"+ row[5][:-1]) # get rid of '\n
            self.context[row[0]] = curr_context # article_name: context

      print("length of dataset: ", len(self.questions))

  def __len__(self):
      return len(self.questions)

  def __getitem__(self, idx):
      # retrieve context here -> less mem storage overhead
      return self.questions[idx], self.answers[idx], self.context[self.article_name[idx]]


# tokenize the text features
def prepare_features(data, cache_path, tokenizer, max_len_inp=100000,max_len_out=96):
  '''
    tokenize the text features and embed them by the inputed tokenizer
    the features are dump to the cache_path via pickle to avoid re-computation next imte
  '''

  inputs = []
  targets = []
  for q, a, c in tqdm(data):
    input_ = f"context: {c}  answer: {a}" # T5 Input format for QA tasks
    target = f"question: {str(q)}" # Output format we require

    # tokenize inputs
    tokenized_inputs = tokenizer.batch_encode_plus(
      [input_], max_length=max_len_inp,padding='max_length',
      return_tensors="pt" #pytorch tensors
    )
    # tokenize targets
    tokenized_targets = tokenizer.batch_encode_plus(
      [target], max_length=max_len_out,
      padding='max_length',return_tensors="pt"
    )

    inputs.append(tokenized_inputs)
    targets.append(tokenized_targets)

  all_features = {}
  all_features['input'] = inputs
  all_features['question'] = targets
  pkl.dump(all_features, open(cache_path, 'wb')) # dump the features somewhere


class FeatureData(Dataset):
    '''
    Dataset for the preprocessed features
    '''
    def __init__(self, feat_path, split, split_point):
        self.feat_path = feat_path
        # load features
        feats = pkl.load(open(self.feat_path, 'rb' )) # load the features and extract
        if split == 'train':
            self.inputs = feats['input'][:-split_point]
            self.questions = feats['question'][:-split_point]
        elif split == 'test':
            self.inputs = feats['input'][-split_point:]
            self.questions = feats['question'][-split_point:]

        print(f"length of feature {split} set: ", len(self.questions))

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        # retrieve context here -> less mem storage overhead
        input_ids = self.inputs[index]['input_ids'].squeeze()
        target_ids = self.questions[index]['input_ids'].squeeze()
        input_mask = self.inputs[index]['attention_mask'].squeeze()
        target_mask = self.questions[index]['attention_mask'].squeeze()
        labels = copy.deepcopy(target_ids)
        labels[labels == 0] = -100
        return {'input_ids': input_ids, 'input_mask': input_mask, 
                'target_ids': target_ids, 'target_mask': target_mask, 
                'labels': labels}



def get_dataloaders(feats_train, feats_test, batch_size):
  # split here
  dataloader_train = DataLoader(feats_train, batch_size=batch_size)
  dataloader_test = DataLoader(feats_test, batch_size=batch_size)
  print(f"Loaded train feature data with {len(dataloader_train)} batches")
  print(f"Loaded test feature data with {len(dataloader_test)} batches")
  return dataloader_train, dataloader_test


# load T5 tokenizer
def get_tokenizer(checkpoint: str) -> T5Tokenizer:
    tokenizer = T5Tokenizer.from_pretrained(checkpoint)
    return tokenizer


# load T5
def get_model(checkpoint: str, device: str, tokenizer: T5Tokenizer) -> T5ForConditionalGeneration:
    config = T5Config(decoder_start_token_id=tokenizer.pad_token_id)
    model = T5ForConditionalGeneration(config).from_pretrained(checkpoint)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    return model

