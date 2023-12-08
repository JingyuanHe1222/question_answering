#!pip install rouge
#!pip install scikit-learn

################################################
### SEMANTIC

#BLEU SCORE: evaluating whole sentences or phrases, focusing on the correct order and choice of words
import nltk
from nltk.translate.bleu_score import sentence_bleu

'''
def calculate_bleu_score(reference, candidate, debug=False):
    # Tokenize the reference and candidate sentences
    reference_tokens = reference.split()
    candidate_tokens = candidate.split()
    
    result=sentence_bleu([reference_tokens], candidate_tokens)

    # NLTK expects the reference as a list of lists of tokens, and candidate as a list of tokens
    if debug==True:
        print('BLEU:', result)

    return result
'''

import nltk
from nltk.translate.bleu_score import sentence_bleu

def calculate_bleu_score(reference, candidate, debug=False):

    # Tokenizing the sentences into words
    reference_tokens = nltk.word_tokenize(reference)
    candidate_tokens = nltk.word_tokenize(candidate)

    # Calculating BLEU score
    # Note that the reference sentence must be in a list of lists format
    bleu_score = sentence_bleu([reference_tokens], candidate_tokens)

    return bleu_score




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


def syntactic_score_for_answers(question, answer, weights_IS_yes_no=[3,2,1], weights_NOT_yes_no=[1,2,3], debug=False, extra_outputs='False'):
    
    #only question and answer are strictly required. Others are set to default values of equal weights, no print out and no additional outputs
    #There are 2 sets of weights since answers to YES/NO type questions are not comparable to non Yes/No type
    #weights are of [0] coherency, [1] conciseness and [2] fluency are proportions which are scaled to 0 and 1 within the function. 
    
    # Default values have the following logic:
    # if IS a Yes/No type, then fluency is less relevant since short answers are generally not fluent. Better answers should be coherent first and concise second, thus weights_IS_yes_no=[3,2,1]
    # if NOT a Yes/No type, then fluency is deemed more important than conciseness. Coherency in this case represents answering in the "wrong format", thus weights_NOT_yes_no=[1,2,3]
    

    is_yes_no_type=is_yes_no_question(question)
    coherency = evaluate_type_coherency(question,answer)
    conciseness=evaluate_conciseness(answer)
    fluency=evaluate_syntax_fluency(answer)
    
    #choose adequate weights for the case
    if is_yes_no_type==1:
        weights=weights_IS_yes_no   
    else:
        weights=weights_NOT_yes_no   

    #calulate score as weighted average
    syntactic_score=(coherency*weights[0]+conciseness*weights[1]+fluency*weights[2])/np.sum(weights)

    #print out for debug
    if debug==True:
        print('Q:',question,
              '|YES/NO TYPE:',is_yes_no_type, 
              '||A:',answer,
              '|COHERENCY(',weights[0],'):',coherency, 
              '|CONCISENESS(',weights[1],'):',conciseness,
              '|FLUENCY(',weights[2],'):', fluency, 
              '||SYNTACTIC SCORE:',syntactic_score
               ) 
    
    #return outputs
    if extra_outputs==True:
        return syntactic_score, is_yes_no_type, coherency, fluency, conciseness
    else:
        return syntactic_score

def syntactic_score_for_questions(question, weights=[1,1]):
    if question=='':
        return 0
    #only question is strictly required.  

    conciseness=evaluate_conciseness(question)
    fluency=evaluate_syntax_fluency(question)
    
    #calulate score as weighted average
    syntactic_score=(conciseness*weights[0]+fluency*weights[1])/np.sum(weights)

    return syntactic_score

##################################

def load_QA(file):
    human_questions=[]
    with open(file+'_human_questions.txt', 'r') as read:
        # Read and print each line
        for line in read:
            human_questions.append(line.strip())
    print('Human Q:',len(human_questions),human_questions)

    #Human answers
    human_answers=[]
    with open(file+'_human_answers.txt', 'r') as read:
        # Read and print each line
        for line in read:
            human_answers.append(line.strip())
    print('Human A:',len(human_answers),human_answers)

    #Machine questions
    machine_questions=[]
    with open(file+'_machine_questions.txt', 'r') as read:
        # Read and print each line
        for line in read:
            line=line[3:] #take out first characters
            machine_questions.append(line.strip())
    print('Machine Q:',len(machine_questions),machine_questions)

    machine_answers=[]
    with open(file+'_machine_question_answer_pairs.txt', 'r') as read:
        # Read and print each line
        for line in read:
            if line[0]=='A': #only upload answers
                line=line[3:] #take out first characters
                machine_answers.append(line.strip())
    print('Machine A:',len(machine_answers),machine_answers)

    return human_questions, human_answers, machine_questions, machine_answers


def question_syntactic_analysis(article_name, human_questions, machine_questions, output_file):
    #Human Question Analysis

    with open(output_file, 'a') as file: 

        header=('article_name',
                'human/machine',
                'question',
                'is_yes_no_type_question',
                'question_type',
                'word_count', 'n_content_words', 'n_relevant_words', 'conciseness_score',
                'has_verb', 'has_subject', 'has_reasonable_length', 'n_unique_POS','has_coherent_structure', 'fluency_score',
                'syntactic_score_for_questions'
                )

        file.write('|'.join(header) + "\n")

        for i in range(len(human_questions)):

                line=(article_name, #article name
                    'human', #human/machine,
                    human_questions[i], #question,
                    is_yes_no_question(human_questions[i]), #is yes/no type
                    classify_question(human_questions[i]), #question_type
                    *evaluate_conciseness(human_questions[i], detail=True), #word count, n content words, n relevant words, conciseness score
                    *evaluate_syntax_fluency(human_questions[i],detail=True), #'has_verb', 'has_subject', 'has_reasonable_length', 'has_unique_POS','has_coherent_structure', 'fluency_score'
                    syntactic_score_for_questions(human_questions[i])
                    )

                # Convert all elements to strings, converting True/False to 1/0, join them with a comma, and write to the file
                line = [str(int(item)) if isinstance(item, bool) else str(item) for item in line]
                file.write('|'.join(line) + "\n")

    #Machine Question Analysis

    with open(output_file, 'a') as file:

        for i in range(len(machine_questions)):

                line=(article_name, #article name
                    'machine', #human/machine,
                    machine_questions[i], #question,
                    is_yes_no_question(machine_questions[i]), #is yes/no type
                    classify_question(machine_questions[i]), #question_type
                    *evaluate_conciseness(machine_questions[i], detail=True), #word count, n content words, n relevant words, conciseness score
                    *evaluate_syntax_fluency(machine_questions[i],detail=True), #'has_verb', 'has_subject', 'has_reasonable_length', 'has_unique_POS','has_coherent_structure', 'fluency_score'
                    syntactic_score_for_questions(machine_questions[i])
                    )

                # Convert all elements to strings, converting True/False to 1/0, join them with a comma, and write to the file
                line = [str(int(item)) if isinstance(item, bool) else str(item) for item in line]
                file.write('|'.join(line) + "\n")


    
def question_semantic_analysis(article_name, human_questions, machine_questions, output_file):
    with open(output_file, 'a') as file:

        header=('article_name',
                'machine_question',
                'human_question',
                'is_yes_no_machine_question',
                'type_machine_question',
                'is_yes_no_human_question',
                'type_human_question',
                'match_yes_no_type_question',
                'match_type_question',

                'f1_char_score',
                'bleu_score',
                'jaccard_similarity',
                'jaccard_similarity_bigrams',
                'overlap_coefficient'
                )

        file.write('|'.join(header) + "\n")

        for machine in machine_questions:
                for human in human_questions:
                    line=(article_name, #article name
                            machine,      #machine questions
                            human,         #human question
                            is_yes_no_question(machine), #is yes/no type machine question
                            classify_question(machine), #type machine question
                            is_yes_no_question(human), #is yes/no human question
                            classify_question(human), #type human question,
                            is_yes_no_question(machine)==is_yes_no_question(human), #match yes/no type question
                            classify_question(machine)==classify_question(human), #match type question

                            calculate_f1_character_score(human,machine),
                            calculate_bleu_score(human, machine),
                            get_jaccard_similarity(human,machine),
                            get_jaccard_similarity_bigrams(human,machine),
                            get_overlap_coefficient(human,machine)
                            )
                                # Convert all elements to strings, converting True/False to 1/0, join them with a comma, and write to the file
                    line = [str(int(item)) if isinstance(item, bool) else str(item) for item in line]
                    file.write('|'.join(line) + "\n")


def answer_syntactic_semantic_analysis(article_name, human_questions, human_answers, machine_answers, output_file):
## Answer Analysis - Semantic and Syntactic

      with open(output_file, 'a') as file:
            header=('article_name',
            'human_question',
            'is_yes_no_type_question',
            'question_type',
            'human_answer',
            'machine_answer',
            'is_yes_no_type_human_answer',
            'is_yes_no_type_machine_answer',
            'is_coherent_type_human_answer',
            'is_coherent_type_machine_answer',

            #Semantic
            'f1_char_score',
            'bleu_score',
            'jaccard_similarity',
            'jaccard_similarity_bigrams',
            'overlap_coefficient',

            #Syntactic human
            'word_count_human', 'n_content_words_human', 'n_relevant_words_human', 'conciseness_score_human',
            'has_verb_human', 'has_subject_human', 'has_reasonable_length_human', 'n_unique_POS_human','has_coherent_structure_human', 'fluency_score_human',
            'syntactic_score_for_answers_human', 

            #Syntactic machine
            'word_count_machine', 'n_content_words_machine', 'n_relevant_words_machine', 'conciseness_score_machine',
            'has_verb_machine', 'has_subject_machine', 'has_reasonable_length_machine', 'n_unique_POS_machine','has_coherent_structure_machine', 'fluency_score_machine',
            'syntactic_score_for_answers_machine'

            )

            file.write('|'.join(header) + "\n")

            for i in range(len(human_questions)):

                line=(article_name, #article name
                    human_questions[i], #human question,
                    is_yes_no_question(human_questions[i]), #is yes/no type
                    classify_question(human_questions[i]), #question_type
                    human_answers[i], #human answer,
                    machine_answers[i], #machine answer,
                    is_yes_no_answer(human_answers[i]), #is yes/no type human
                    is_yes_no_answer(machine_answers[i]), #is yes/no type machine
                    is_yes_no_question(human_questions[i])==is_yes_no_answer(human_answers[i]), #'is_coherent_type_human_answer',
                    is_yes_no_question(human_questions[i])==is_yes_no_answer(machine_answers[i]), #'is_coherent_type_machine_answer',

                    #Semantic
                    calculate_f1_character_score(human_answers[i],machine_answers[i]),
                    calculate_bleu_score(human_answers[i],machine_answers[i]),
                    get_jaccard_similarity(human_answers[i],machine_answers[i]),
                    get_jaccard_similarity_bigrams(human_answers[i],machine_answers[i]),
                    get_overlap_coefficient(human_answers[i],machine_answers[i]),

                    #Syntactic human
                    *evaluate_conciseness(human_answers[i], detail=True), #word c,ount, n content words, n relevant words, conciseness score
                    *evaluate_syntax_fluency(human_answers[i],detail=True), #'has_verb', 'has_subject', 'has_reasonable_length', 'has_unique_POS','has_coherent_structure', 'fluency_score'
                    syntactic_score_for_answers(human_questions[i], human_answers[i]),

                    #Syntactic machine
                    *evaluate_conciseness(machine_answers[i], detail=True), #word count, n content words, n relevant words, conciseness score
                    *evaluate_syntax_fluency(machine_answers[i],detail=True), #'has_verb', 'has_subject', 'has_reasonable_length', 'has_unique_POS','has_coherent_structure', 'fluency_score'
                    syntactic_score_for_answers(human_questions[i], machine_answers[i])
                
                    )

                # Convert all elements to strings, converting True/False to 1/0, join them with a comma, and write to the file
                line = [str(int(item)) if isinstance(item, bool) else str(item) for item in line]
                file.write('|'.join(line) + "\n")