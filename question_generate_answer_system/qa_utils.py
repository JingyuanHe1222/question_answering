import os
import wikipediaapi
from functools import lru_cache
import nltk
from nltk.tokenize import sent_tokenize
import re
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

nltk.download("punkt")


@lru_cache(maxsize=None)
def split_text_into_sentences(text):
    """
    Splits the given text into sentences using NLTK's sentence tokenizer.

    :param text: String containing the text to be split into sentences.
    :return: A list of sentences.
    """
    sentences = sent_tokenize(text)
    return sentences


@lru_cache(maxsize=None)
def split_into_paragraphs(text, tokenizer, max_length=400):
    """
    Splits the text into paragraphs and returns a list of paragraphs.
    Each paragraph's token length is limited to max_length.
    """
    paragraphs = [p for p in text.split("\n") if p]
    short_paragraphs = []
    for paragraph in paragraphs:
        tokens = tokenizer.tokenize(paragraph)
        start_token = 0
        while start_token < len(tokens):
            end_token = min(start_token + max_length, len(tokens))
            short_paragraph = tokens[start_token:end_token]
            short_paragraphs.append(tokenizer.convert_tokens_to_string(short_paragraph))
            start_token = end_token
    return short_paragraphs


@lru_cache(maxsize=None)
def download_article_from_wikipedia(article_name, save_locally=True):
    wiki_wiki = wikipediaapi.Wikipedia(
        language="en",
        extract_format=wikipediaapi.ExtractFormat.WIKI,
        user_agent="11411_nlp_team",
    )
    article_content = wiki_wiki.page(article_name).text
    if save_locally:
        if not article_content:
            raise RuntimeError("Article not found or empty.")
        # Save the text to a file
        file_name = f"{article_name.replace(' ', '_')}.txt"
        with open(file_name, "w", encoding="utf-8") as file:
            file.write(article_content)
    return article_content


def is_yes_no_question(question):
    question = question.lower().strip()
    # purposely has space after words since it should be a word in itself as part of a sentence
    return question.startswith(
        (
            "is ",
            "are ",
            "do ",
            "does ",
            "did ",
            "was ",
            "were ",
            "will ",
            "can ",
            "could ",
            "should ",
            "would " "have ",
            "has ",
            "had ",
            "must ",
            "am ",
        )
    )


def is_yes_no_question_info_retrieval_text(question):
    if is_yes_no_question(question) == 1:  # it is a yes/no question
        words = question.split()  # split words
        words = words[1:]  # exclude first word
        if words[-1].endswith("?"):  # exclude question mark if it exists
            words[-1] = words[-1][:-1]
        return " ".join(words)  # join the string back
    return (
        None  # do not return info for information retrieval if it is NOT a yes/no type
    )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder_model = SentenceTransformer("sentence-transformers/sentence-t5-base").to(device)


def extract_context_information_retrival(article, question, n):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_model.to(device)
    all_context = article

    # split the context into sentences
    list_context = split_article_to_sentences_nltk(all_context)

    # embed everything and compute the 1-nn IR result for the question
    c_embed = []
    for context in list_context:
        output = encoder_model.encode(context)
        c_embed.append(output)
    context_embeds = np.vstack(c_embed)

    q_embed = encoder_model.encode(question)

    scores = context_embeds.dot(q_embed)
    k_nn = scores.argsort()[-n:][::-1]
    k_nn = list(k_nn)
    # closest context sentence in text
    contexts = [
        top_context for top_context in np.array(list_context)[k_nn]
    ]  # top n context sentences
    return " ".join(contexts)
    # top_scores = sorted(scores, reverse=True)[:n]

    # q_text = is_yes_no_question_info_retrieval_text(question)
    # if q_text == None:
    #     ### pipeline thing
    #     #         print("---TODO---") ### TODO
    #     return None  ### TODO
    # else:
    #     first_context = contexts[0]
    #     score = top_scores[0]

    #     return score  ### TODO


def read_article_file(article_filename):
    """
    I choose to not implement any exception handling here. Failure to read the
    input article should stop the execution of the program.
    """
    # Check if the file name contains a path
    if not os.path.isabs(article_filename):
        # If not, assume the file is in the current directory
        article_filename = os.path.join(os.getcwd(), article_filename)
    with open(article_filename, "r") as file:
        return file.read()


@lru_cache(maxsize=None)
def split_article_to_sentences_nltk(article):
    """
    Split the article into sentences
    """
    # split the context into sentences
    curr_context = article.split("Related Wikipedia Articles")[0]
    curr_context = curr_context.replace("\n", " ")
    list_context = nltk.tokenize.sent_tokenize(curr_context)
    return list_context


def read_questions_from_file(question_filename):
    questions = []
    if not os.path.isabs(question_filename):
        # If not, assume the file is in the current directory
        question_filename = os.path.join(os.getcwd(), question_filename)
    with open(question_filename, "r") as file:
        for line in file:
            # Strip leading/trailing whitespace and check if line is not empty
            stripped_line = line.strip()
            if stripped_line:
                questions.append(stripped_line)
    return questions


YES_NO_PATTERN = re.compile(
    r"^(is|are|can|do|does|did|will|would|should|has|have|had|am|were|was)\b"
)


def is_boolean_question(question):
    return bool(YES_NO_PATTERN.match(question.strip().lower()))


class QuestionAnswerWriter:
    QUESTION = 0
    QUESTION_AND_ANSWER = 1

    def __init__(
        self,
        qa_pair_filename="generated_question_answer_pairs.txt",
        question_filename="generated_questions.txt",
    ):
        self.qa_pair_filename = qa_pair_filename
        self.question_filename = question_filename

    def set_qa_pair_filename(self, qa_pair_filename):
        self.qa_pair_filename = qa_pair_filename

    def start_new_session(self, mode):
        file_to_remove = self.qa_pair_filename
        if QuestionAnswerWriter.QUESTION == mode:
            file_to_remove = self.question_filename
        if os.path.exists(file_to_remove):
            os.remove(file_to_remove)

    def write_questions_to_file(self, questions):
        for question in questions:
            question = question.strip()
            with open(self.question_filename, "a") as file:
                file.write(f"Q: {question}\n")

    def write_question_answer_pair_to_file(self, question, answer):
        question = question.strip()
        answer = answer.strip()
        with open(self.qa_pair_filename, "a") as file:
            file.write(
                f"Q: {question}\nA: {answer}\n"
            )  # append the question-answer pair followed by a newline
