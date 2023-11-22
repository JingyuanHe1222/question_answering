import os
import numpy as np
import pickle as pkl
from tqdm import tqdm
import copy

import nltk
nltk.download('punkt')
from nltk import tokenize

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer, T5Model


# prepare data -> retrieve the top k context sentences (join with space)
class CustomData(Dataset):
    '''
    Process raw data
    '''
    def __init__(self, file_dir, model, k=1):
        self.file = file_dir
        self.article_name = []
        self.questions = []
        self.answers = []
        self.q_diffi = []
        self.a_diffi = []
        self.article_path = []
        self.context_nn = {}
        self.context = {} # only fill when load the dataset
        self.context_embed = {}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device)
        model.eval()
        
        # get question answer pairs
        for div in ['S08', 'S09', 'S10']:
            skip = True
            qa_path = os.path.join(self.file, div, "question_answer_pairs.txt")
            num_lines = sum(1 for line in open(qa_path,'rb'))
            with open(qa_path, 'rb') as f:

                for line in tqdm(f, total=num_lines):

                    if skip:
                        skip = False # skip the first line
                        continue

                    try: # only continue if the decoding is valid for utf-8
                        row = line.decode().split('\t')
                    except:
                        continue

                    if "NULL" in row:
                        continue # if any feature does not exist -> skip

                    context_file = self.file + "/" + div + "/"+ row[5][:-1] + ".txt" # path to the context file
                    if not (os.path.exists(context_file) and os.path.isfile(context_file)): # otherwise context doesn't exist: invalid
                        continue

                    # only process document embedding when needed (article first found)
                    if row[0] not in self.context_embed.keys():
                        # check if context could be extracted
                        try:
                            with open(context_file, 'rb') as f:
                                curr_context = f.read().decode() # could be decoded, otherwise skip
                        except:
                            continue

                        curr_context = curr_context.split('Related Wikipedia Articles')[0] # ignore everything after Related articles
                        curr_context = curr_context.replace('\n',' ')
                        self.context[row[0]] = tokenize.sent_tokenize(curr_context)
                        # encode context and add to corresponding files
                        c_embed = []
                        for context in self.context[row[0]]:
                            output = self.model.encode(context)
                            c_embed.append(output)
                            self.context_embed[row[0]] = np.vstack(c_embed)

                    # get top-1 similar context
                    qa_input = row[0] + " " + row[1]
                    # qa embedding 
                    encoded_qa = self.model.encode(qa_input)# detach to save gpu mem
                    c_embed = self.context_embed[row[0]] # load the context embeddings
                    # compute knn score: dot product
                    
#                     print("c_embed size: ", c_embed.shape)
#                     print("qa_embed size: ", encoded_qa.shape)
                    
                    scores = c_embed.dot(encoded_qa)
                    k_nn = scores.argsort()[-k:]
                    k_nn = list(k_nn)
                    # the text of the closest neighbor
                    nn_context = " ".join([top_context for top_context in np.array(self.context[row[0]])[k_nn]])
                    self.context_nn[row[0]] = nn_context

                    # other info
                    self.article_name.append(row[0])
                    self.questions.append(row[1])
                    self.answers.append(row[2])
                    self.q_diffi.append(row[3]) # difficulty
                    self.a_diffi.append(row[4])
                    self.article_path.append(div + "/"+ row[5][:-1]) # get rid of '\n

        print("length of dataset: ", len(self.questions))

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return self.questions[idx], self.answers[idx], self.context_nn[self.article_name[idx]]
    
    
# class CustomData(Dataset):
#   '''
#     Process raw data
#   '''
#   def __init__(self, file_dir, model, tokenzier, k=1):
#       self.file = file_dir
#       self.article_name = []
#       self.questions = []
#       self.answers = []
#       self.q_diffi = []
#       self.a_diffi = []
#       self.article_path = []
#       self.context_nn = {}
#       self.context = {} # only fill when load the dataset
#       self.context_embed = {}

#       device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#       self.model = model.to(device)
#       self.tok = tokenzier
#       model.eval()
        
#       # get question answer pairs
#       for div in ['S08', 'S09', 'S10']:
#         skip = True
#         qa_path = os.path.join(self.file, div, "question_answer_pairs.txt")
#         num_lines = sum(1 for line in open(qa_path,'rb'))
#         with open(qa_path, 'rb') as f:
#           for line in tqdm(f, total=num_lines):
#             if skip:
#               skip = False # skip the first line
#               continue
#             try: # only continue if the decoding is valid for utf-8
#               row = line.decode().split('\t')
#             except:
#               continue
#             # print(row)
#             if "NULL" in row:
#               continue # if any feature does not exist -> skip

#             context_file = self.file + "/" + div + "/"+ row[5][:-1] + ".txt" # path to the context file
#             if not (os.path.exists(context_file) and os.path.isfile(context_file)): # otherwise context doesn't exist: invalid
#               continue

#             # only process document embedding when needed (article first found)
#             if row[0] not in self.context_embed.keys():
#               # check if context could be extracted
#               try:
#                 with open(context_file, 'rb') as f:
#                   curr_context = f.read().decode() # could be decoded, otherwise skip
#               except:
#                 continue

#               curr_context = curr_context.split('Related Wikipedia Articles')[0] # ignore everything after Related articles
#               curr_context = curr_context.replace('\n',' ')
#               self.context[row[0]] = tokenize.sent_tokenize(curr_context)
#               # encode context and add to corresponding files
#               c_embed = []
#               for context in self.context[row[0]]:
#                 enc = self.tok(context, max_length=512, padding='max_length', return_tensors="pt")
#                 output = self.model.encoder(
#                   input_ids=enc["input_ids"].to(device),
#                   attention_mask=enc["attention_mask"].to(device),
#                   return_dict=True
#                 ).last_hidden_state.cpu().detach() # detach to save gpu mem
#                 c_embed.append(output)
#               self.context_embed[row[0]] = torch.cat(c_embed)

#             # get top-1 similar context
#             qa_input = row[0] + " " + row[1]
#             qa_enc = self.tok(qa_input, max_length=512, padding='max_length', return_tensors="pt")
#             encoded_qa = self.model.encoder(
#               input_ids=qa_enc["input_ids"].to(device),
#               attention_mask=qa_enc["attention_mask"].to(device),
#               return_dict=True
#             ).last_hidden_state.cpu().detach() # detach to save gpu mem
#             c_embed = self.context_embed[row[0]] # load the context embeddings
#             # compute knn score: dot product
#             scores = c_embed.matmul(encoded_qa)
#             nn = torch.argmax(scores)
#             # the text of the closest neighbor
#             self.context_nn[row[0]] = self.context[row[0]][nn]

#             # other info
#             self.article_name.append(row[0])
#             self.questions.append(row[1])
#             self.answers.append(row[2])
#             self.q_diffi.append(row[3]) # difficulty
#             self.a_diffi.append(row[4])
#             self.article_path.append(div + "/"+ row[5][:-1]) # get rid of '\n

#       print("length of dataset: ", len(self.questions))

#   def __len__(self):
#       return len(self.questions)

#   def __getitem__(self, idx):
#       return self.questions[idx], self.answers[idx], self.context[self.article_name[idx]]
    
    
    
# dump the raw dataset to json file
def dump_set(data, output_file):
  l = []
  for q, a, c in data:
    l.add({"question": q, "answer": a, "context": context})
  with open(output_file, 'w') as fp:
      json.dump(l, fp)


        
# load the raw dataset from json file as a dataset object
def load_json_data(json_file):
  test_dataset = load_dataset('json', data_files={'all':json_file})
  return test_dataset['all']



# tokenize the text features
def prepare_features_q(data, cache_path, tokenizer, max_len_inp=512,max_len_out=512):
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
  all_features['target'] = targets
  pkl.dump(all_features, open(cache_path, 'wb')) # dump the features somewhere
    
    
# tokenize the text features
def prepare_features_a(data, cache_path, tokenizer, max_len_inp=512,max_len_out=512):

  inputs = []
  targets = []

  for q, a, c in tqdm(data):

    input_ = f"question: {q}  context: {c}" # T5 Input format for QA tasks
    target = f"answer: {str(a)}" # Output format we require

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
  all_features['target'] = targets
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
            self.questions = feats['target'][:-split_point]
        elif split == 'test':
            self.inputs = feats['input'][-split_point:]
            self.questions = feats['target'][-split_point:]

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


