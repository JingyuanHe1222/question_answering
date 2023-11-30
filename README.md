# question_answering
Question Generation and Answering System for 11-611 CMU 

## install dependencies
- in EC2 with Pytorch deep Learning AMI (Otherwise create your python or conda environment): 

		source activate pytorch
  
- install dependencies and unzip dataset
  
		pip install -r requirements.txt
  		unzip Question_Answer_Dataset_v1.2.zip
        
## usage
- fine-tune generation model
    - put the hyperparameters in a config file and run the training process for type "question" or type "answer"
    
			python fine_tune_{type}_generation.py path_to_your_config/config.yaml  
            
    - model and log status will be saved automatically
    - the last checkpoint model will be reloaded to resume training upon interruption
    

## file structure
- utils.py
  - process raw dataset and tokenization
    - do information retreival on the paragraphs to get the top k context sentences for a q-a pair
      - algo: KNN
      - metric: inner product
      - embedding: encoded by sentence-t5-base
  - dataloaders
    - load the tokenized inputs 
  - get T5 family models and tokenizer
 
- fine_tune_question_generation.py, fine_tune_answer_generation.py
  - pipeline for fine-tuning model in command line or shell script 
    
- Question_Generation_Playground.ipynb 
  - compute the raw data and its question generation feature if it doesn't exist yet and save the features 
  - training pipeline stage for question generation
    
- Answer_G_Playground.ipynb
  - compute the raw data and its answer generation feature if it doesn't exist yet and save the features 
  - training pipeline stage for answer generation
