#!/bin/bash


echo "entering cache folder..."
cd .model-cache

echo "getting model weights from gdown..."
gdown --id 1_OF-lFEnlxQH6Fyfy3Bgg3cOqwDJoIxo

echo "unzipping the model checkpoint"
unzip t5-base_e12_lr1e-05_eps5e-05_Adam_cosine_batch1.zip

echo "wrapping up..."
mv t5-base_e12_lr1e-05_eps5e-05_Adam_cosine_batch1/latest_model.pt ~/question_answering/question_generate_answer_system/.model-cache

