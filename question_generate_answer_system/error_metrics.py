#!pip install rouge
#!pip install scikit-learn

################################################
### SEMANTIC

#BLEU SCORE: evaluating whole sentences or phrases, focusing on the correct order and choice of words
import nltk
from nltk.translate.bleu_score import sentence_bleu

def calculate_bleu_score(reference, candidate, debug=False):
    # Tokenize the reference and candidate sentences
    reference_tokens = reference.split()
    candidate_tokens = candidate.split()
    
    result=sentence_bleu([reference_tokens], candidate_tokens)

    # NLTK expects the reference as a list of lists of tokens, and candidate as a list of tokens
    if debug==True:
        print('BLEU:', result)

    return result


#F1-Character Score: measure of how many characters are correctly predicted, regardless of their order.
#import scikit-learn
#from sklearn.metrics import f1_score

def calculate_f1_character_score(true_answer, predicted_answer, debug=False):
    true_chars = set(true_answer)
    predicted_chars = set(predicted_answer)
    
    common_chars = true_chars.intersection(predicted_chars)
    precision = len(common_chars) / len(predicted_chars) if predicted_chars else 0
    recall = len(common_chars) / len(true_chars) if true_chars else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    if debug==True:
        print('f1 character score:', f1)

    return f1

'''
#ROUGE (Recall-Oriented Understudy for Gisting Evaluation) Score measures the overlap between the predicted text and the reference text.

#!pip install rouge
from rouge import Rouge

def calculate_rouge_score(reference, candidate, debug):
    rouge = Rouge()
    scores = rouge.get_scores(candidate, reference)
    
    if debug==True:
        print('Rouge:', scores)
    
    return scores  # Returns ROUGE-1 (unigram), ROUGE-2 (bigram), and ROUGE-L (largest common sequence) scores. 
'''

from nltk.metrics import jaccard_distance
from nltk.util import ngrams
from collections import Counter

def tokenize(sentence):
    # Tokenizes the sentence into words and removes punctuation
    return set(sentence.lower().translate(str.maketrans('', '', '?.,!')).split())

def get_jaccard_similarity(sentence1, sentence2):
    # Convert sentences to sets of words
    words_sentence1 = tokenize(sentence1)
    words_sentence2 = tokenize(sentence2)
    
    # Check if the union of sets is empty
    if len(words_sentence1.union(words_sentence2)) == 0:
        # If both are empty, they are identical, hence similarity is 1
        # If one is non-empty, then similarity is 0
        return float(words_sentence1 == words_sentence2)
    
    # Calculate Jaccard distance and convert to Jaccard similarity
    jaccard_dist = jaccard_distance(words_sentence1, words_sentence2)
    jaccard_similarity = 1 - jaccard_dist
    return jaccard_similarity

def get_jaccard_similarity_bigrams(sentence1, sentence2):
    # Generate bigrams for both sentences
    bigrams_sentence1 = set(ngrams(sentence1.split(), n=2))
    bigrams_sentence2 = set(ngrams(sentence2.split(), n=2))
    
    # Check if the union of bigram sets is empty
    if len(bigrams_sentence1.union(bigrams_sentence2)) == 0:
        # If both are empty, they are identical, hence similarity is 1
        # If one is non-empty, then similarity is 0
        return float(bigrams_sentence1 == bigrams_sentence2)
    
    # Calculate Jaccard distance for bigrams and convert to similarity
    jaccard_dist_bigrams = jaccard_distance(bigrams_sentence1, bigrams_sentence2)
    jaccard_similarity_bigrams = 1 - jaccard_dist_bigrams
    return jaccard_similarity_bigrams

def get_overlap_coefficient(sentence1, sentence2):
    # Convert sentences to sets of words
    words_sentence1 = tokenize(sentence1)
    words_sentence2 = tokenize(sentence2)
    
    # Calculate overlap coefficient
    overlap_coefficient = len(words_sentence1 & words_sentence2) / np.maximum(1,min(len(words_sentence1), len(words_sentence2)))
    return overlap_coefficient



################################################
### SYNTACTIC

# More general question type classifitcation
import re

def classify_question(question, debug=False):
    question = question.lower().strip()

    find='Open-Ended' #by default

    # Yes/No Questions
    if question.startswith(('is ', 'are ', 'do ', 'does ', 'did ', 'was ', 'were ', 'will ', 'can ', 'could ', 'should ', 'have ', 'has ', 'had ')):
        find='Yes/No'

    # Choice Questions
    if ' or ' in question:
        find= 'Choice'

    # WH-Questions
    if question.startswith(('what ', 'where ', 'who ', 'when ', 'why ', 'how ')):
        if question.startswith(('what', 'where', 'who')):
            find='What/Where/Who'
        elif question.startswith(('when', 'why')):
            find='When/Why'
        elif question.startswith('how'):
            find='How'

    if debug==True:
        print('Question type:', find)

    return find


# IS YES/NO QUESTION TYPE
def is_yes_no_question(question, debug=False):
    question = question.lower().strip()
    #purposely has space after words since it should be a word in itself as part of a sentence
    find=question.startswith(('is ', 'are ', 'do ', 'does ', 'did ', 'was ', 'were ', 'will ', 'can ', 'could ', 'should ', 'have ', 'has ', 'had '))

    if debug==True:
        print('Is Polar Question:',find)

    return find

# IS YES/NO ANSWER TYPE
def is_yes_no_answer(answer, debug=False):
    answer = answer.lower().strip()
    find=answer.startswith(('yes','no'))
    
    if debug==True:
        print('Is Polar Answer:', find)
    
    return find

# COHERENCY: QUESTION AND ANSWER TYPE MATCH
def evaluate_type_coherency(question,answer, debug=False):
    eval=is_yes_no_question(question)==is_yes_no_answer(answer)
    if debug==True:
        print('Answer is coherent with Question type:', eval)
    return eval




# CONCISENESS
import spacy
from nltk.tokenize import word_tokenize
import numpy as np

# Load the spaCy model for linguistic features
nlp = spacy.load("en_core_web_sm")

def evaluate_conciseness(answer, debug=False, detail=False):
    # Tokenize the answer and analyze with spaCy
    if len(answer)==0: 
        return 0

    doc = nlp(answer)
    word_count = len(word_tokenize(answer))
    punctuation_count = sum(token.pos == spacy.symbols.PUNCT for token in doc)
    relevant_words=np.maximum(1,word_count-punctuation_count)

    # content words NOUN and VERB
    content_words_count = sum(token.pos in [spacy.symbols.NOUN, spacy.symbols.VERB] for token in doc)
    
    # Consider yes/no as content if it is the first token
    first_token_is_yes_no = doc[0].text.lower() in ["yes", "no"]
    if first_token_is_yes_no:
        content_words_count += 1
    
    conciseness_score=content_words_count/relevant_words

    if debug==True:
        print('Answer:', answer)
        print('Word count:',word_count, 
              'Punctuation count', punctuation_count, 
              'Content words count:', content_words_count, 
              'Relevant words:', relevant_words, 
              '|Conciseness score:', conciseness_score)
    
    if detail==True:
        return word_count, content_words_count, relevant_words, conciseness_score

    return conciseness_score


# FLUENCY
import spacy

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

def evaluate_syntax_fluency(sentence, debug=False, detail=False):
    doc = nlp(sentence)

    if len(doc)==0:
        return 0
    
    # Heuristics for scoring
    score = 0
    max_score = 5  # Total number of checks

    # Check 1: Sentence has a verb  
    has_verb = any(token.pos_ == "VERB" for token in doc)
    score += 1 if has_verb else 0

    # Check 2: Sentence has a nominal subject  
    has_subject = any(token.dep_ == "nsubj" or token.dep_ == "nsubjpass" for token in doc)
    score += 1 if has_subject else 0

    # Check 3: Reasonable sentence length (not too long or too short)
    # Assuming a sentence should typically be between 3 and 20 words for simplicity
    has_reasonable_length = 1 if 3 <= len(doc) <= 20 else 0
    score += has_reasonable_length

    # Check 4: Variety in sentence structure (using different parts of speech)
    unique_pos = len(set(token.pos_ for token in doc))
    if unique_pos >= 4:  # Arbitrary threshold for variety
        score += 1

    # Check 5: Coherence in dependency structure (root should be a verb or an auxiliary verb)
    has_coherent_structure = doc[0].dep_ == "ROOT" and doc[0].pos_ in ["VERB", "AUX"]
    score += 1 if has_coherent_structure else 0

    # Calculate final score
    fluency_score = score / max_score

    if debug==True:
        print(
        'Sentence', sentence, '\n',
        'Has verb:',has_verb, 
        '|Has subject:', has_subject, 
        '|Reasonable length:', has_reasonable_length, 
        '|Unique POS:', unique_pos, 
        '|Coherent structure:',has_coherent_structure,
        '||Fluency score:', fluency_score
        )

    if detail==True:
        return has_verb, has_subject, has_reasonable_length, unique_pos, has_coherent_structure, fluency_score

    return fluency_score