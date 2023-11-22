# question_answering
Question Generation and Answering System for 11-611 CMU 

## install dependencies
- In EC2 with Pytorch eep Learning AMI (Otherwise create your python or conda environment): 

		source activate pytorch
  
- install dependencies and unzip dataset
  
		pip install -r requirements.txt
  		unzip Question_Answer_Dataset_v1.2.zip
  

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
  - get T5 family models and tokenizer
- Question_Generation_Playground.ipynb 
  - compute the raw data and its question generation feature if it doesn't exist yet and save the features 
  - full training pipeline for question generation
- Answer_Q_Playground.ipynb
  - compute the raw data and its answer generation feature if it doesn't exist yet and save the features 
  - training pipeline for answer generation
