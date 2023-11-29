import os
import numpy as np
import pickle as pkl
from tqdm import tqdm
import argparse
import yaml

# pytorch
import torch
from torch.utils.data import DataLoader, Dataset

# model config
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline, RobertaModel, T5Config, T5ForConditionalGeneration, T5Tokenizer, T5Model, AutoModelWithLMHead, pipeline
from sentence_transformers import SentenceTransformer

# model optim
from torch.optim import AdamW, SGD

# lr schedulers
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

# custom libraries and tools
from utils import *


# specify device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def __save_model(model_dir, model, model_type='latest'):

    if model_type == "latest":
        saved_name = 'latest_model.pt'
    else:
        saved_name = 'best_model.pt'

    root_model_path = os.path.join(model_dir, saved_name)
    state_dict = {'weights': model.state_dict(), 
                  'optimizer': model.optimizer.state_dict(), 
                  'scheduler': model.scheduler.state_dict()}
    torch.save(state_dict, root_model_path)
        
        

# Loads the experiment data if exists to resume training from last saved checkpoint.
def __load_experiment(model_dir, model, model_type='latest'):
    
    if model_type == "latest":
        saved_name = 'latest_model.pt'
    else:
        saved_name = 'best_model.pt'

    if os.path.exists(os.path.join(model_dir, 'train.log')):
        # get current epoch
        current_epoch = 0
        with open(os.path.join(model_dir, 'train.log')) as f:
            for line in f:
                current_epoch += 1
        # get the latest model
        state_dict = torch.load(os.path.join(model_dir, saved_name), map_location=device.type)
        model.load_state_dict(state_dict['weights'])
        model.optimizer.load_state_dict(state_dict['optimizer'])
        model.scheduler.load_state_dict(state_dict['scheduler'])
    else:
        current_epoch = 0

    return model, current_epoch



def log(output_dir, log_str, file_name=None):
    if file_name is None:
        file_name = "all.log"
    output_file = os.path.join(output_dir, file_name)
    with open(output_file, 'a') as f:
        f.write(log_str + '\n')


def get_optimizer(model, opt_name, lr, eps): 
    if opt_name == 'Adam':
        return AdamW(model.parameters(), lr=lr, eps=eps)
    elif opt_name == 'SGD':
        return SGD(model.parameters(), lr=lr, eps=eps)
    
    
def get_scheduler(model, scheduler, n_batches, n_epochs, warmup_portion=0.1):
    train_steps = n_epochs*n_batches
    warm_step = int(train_steps*warmup_portion)
    if scheduler == "linear": 
        return get_linear_schedule_with_warmup(model.optimizer, num_warmup_steps=warm_step,num_training_steps=train_steps)
    elif scheduler == "cosine":
        return get_cosine_schedule_with_warmup(model.optimizer, num_warmup_steps=warm_step,num_training_steps=train_steps)


# training loop
def train(model, dataloader_train, n_epochs, model_dir, log_file):

    model.train() # put to train mode
    
    # load current model if exist
    model, current_epoch = __load_experiment(model_dir, model)
    
    all_losses = []
    
    for e in range(current_epoch, n_epochs):

        losses = 0
        for step, batch in tqdm(enumerate(dataloader_train), total=len(dataloader_train)):
            
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['input_mask'].to(device),
#                 decoder_input_ids=batch['target_ids'].to(device),
#                 decoder_attention_mask=batch['target_mask'].to(device), 
                start_positions = batch['start'].to(device), 
                end_positions = batch['end'].to(device)
            )

            loss = outputs[0]
#             print(loss)

            model.optimizer.zero_grad() # clear loss
            loss.backward()
            model.optimizer.step()  # backprop to update the weights

            if model.scheduler is not None:
                model.scheduler.step()  # update learning rate schedule 

            # log losses
            loss /= len(dataloader_train)
            losses += loss.item()
            
        # output stats
        print(f"Epoch {e}; loss {losses}")
        log(model_dir, "Epoch " + str(e+1) + "; loss " + str(losses), log_file)
        all_losses.append(losses)
        # save model
        __save_model(model_dir, model) # save latest
        if (e > current_epoch and losses < all_losses[-1]):
            __save_model(model_dir, model, model_type='best') # save best model        
        
        

def test(model, dataloader_test, model_dir, log_file):
    
    model, e = __load_experiment(model_dir, model, model_type='latest')
    
    model.eval()
    
    losses = 0
    for step, batch in tqdm(enumerate(dataloader_test), total=len(dataloader_test)):

        outputs = model(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['input_mask'].to(device),
#             decoder_input_ids=batch['target_ids'].to(device),
#             decoder_attention_mask=batch['target_mask'].to(device),
            start_positions = batch['start'].to(device), 
            end_positions = batch['end'].to(device)
        )

        loss = outputs[0]

        # log losses
        loss /= len(dataloader_test)
        losses += loss.item()
        
    # output stats
    print(f"Validation loss {losses}")
    log(model_dir, "Validation loss " + str(losses), log_file)
    
    
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help='path to config file')
    args = parser.parse_args()

    with open(args.config_file) as cf_file:
        config = yaml.safe_load(cf_file.read())
        
    # ---------------- Hyper-Parameters --------------------
    print("---------------- Hyper-Parameters --------------------")
    # models 
    base_model = config['base_model']
    encoder = config['encoder'] 
    # dataset split
    test_points = config['test_points'] # number of samples leave for testing   
    # hyperparameters
    n_epochs = config['n_epochs']
    lr = config['lr']
    weight_decay = config['eps']
    batch_size = config['batch_size']
    optim = config['optim']
    scheduler = config['scheduler']
    log_file = config['log_file'] # log_file
    
    # -------------------- Model -----------------------
    print("---------------- Model --------------------")
    # load tokenizer and model
    processer = get_tokenizer(base_model)
    model = get_model(base_model, device, processer)
    
    # IR encoder -> T-5 sentence dense embeddings
    encoder_model = SentenceTransformer(encoder)

    # -------------------- Dataset -----------------------
    print("---------------- Dataset --------------------")
    # define data_path for raw input and feature_path for feature input
    data_path = 'Question_Answer_Dataset_v1.2'
    feature_cache_path = 'Question_Answer_Dataset_v1.2/features_answers'
    
    # prepare feature data if not yet exist 
    if not (os.path.exists(feature_cache_path) and os.path.isfile(feature_cache_path)):
        # use the encoder to get the raw dataset (context are extracted by IR with the K-NN sentence to the QA pair)
        print("processing raw dataset... ")
        raw_dataset = CustomData(data_path, encoder_model, k=1)
        print("computing features...")
        # tokenize
        prepare_features_a(raw_dataset, feature_cache_path, processer, max_len_inp=512,max_len_out=512)
    else:
        print("features exists")
        
    # feature dataset
    train_dataset = FeatureData_A(feature_cache_path, 'train', test_points)
    test_dataset = FeatureData_A(feature_cache_path, 'test', test_points) 
    
    # ---------------- Setup --------------------
    print("---------------- Setup --------------------")
    # dataloaders
    dataloader_train, dataloader_test = get_dataloaders(train_dataset, test_dataset, batch_size=batch_size)

    # model optimizer
    model.optimizer = get_optimizer(model, optim, lr, weight_decay)

    # learning rate scheduler
    model.scheduler = get_scheduler(model, scheduler, len(dataloader_train), n_epochs)
    
    name = base_model.split('/')[-1] # only the last name

    # model state_dict
    model_dir = f"{name}_e{n_epochs}_lr{lr}_eps{weight_decay}_{optim}_{scheduler}_batch{batch_size}"
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
        
    # -----------------Train and Test-----------------
    print("---------------- Train and Test --------------------")
    train(model, dataloader_train, n_epochs, model_dir, log_file)
    
    test(model, dataloader_test, model_dir, log_file)
    
    
    
    