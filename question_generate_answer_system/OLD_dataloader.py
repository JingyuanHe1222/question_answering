## OLD DATALOADER

import torch
from torch.utils.data import Dataset
import os
import numpy as np


class CustomDataOld(Dataset):

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
              self.article_name.append(row[0])
              self.questions.append(row[1])
              self.answers.append(row[2])
              self.q_diffi.append(row[3])
              self.a_diffi.append(row[4])
              self.article_path.append(div + "/"+ row[5][:-1]) # get rid of '\n

        print("length of dataset: ", len(self.questions))


    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):

        # retrieve context here -> less mem storage overhead
        try:
          curr_context = self.context[self.article_name[idx]]
        except KeyError:
          context_file = self.file + "/" + self.article_path[idx] + ".txt"
          # read all content, including the related items
          with open(context_file, 'rb') as f:
            curr_context = f.read().decode().replace('\n',' ')
          self.context[self.article_name[idx]] = curr_context

        #return self.questions[idx], self.answers[idx], curr_context
        return (self.article_name[idx],
                self.questions[idx],
                self.answers[idx],
                self.q_diffi[idx],
                self.a_diffi[idx],
                self.article_path[idx],
                curr_context
                )