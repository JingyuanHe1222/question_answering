import os
import wikipediaapi
from functools import lru_cache
import nltk
from nltk.tokenize import sent_tokenize
import re

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
def download_article_from_wikipedia(article_name):
    wiki_wiki = wikipediaapi.Wikipedia(
        language="en",
        extract_format=wikipediaapi.ExtractFormat.WIKI,
        user_agent="11411_nlp_team",
    )
    return wiki_wiki.page(article_name).text


def split_article_to_sentences_nltk(article):
    """
    Split the article into sentences
    """
    # split the context into sentences
    curr_context = article.split("Related Wikipedia Articles")[0]
    curr_context = curr_context.replace("\n", " ")
    list_context = nltk.tokenize.sent_tokenize(curr_context)
    print("context is: ")
    print(list_context)
    return list_context


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
        question_filename="questions.txt",
    ):
        self.qa_pair_filename = qa_pair_filename
        self.question_filename = question_filename

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
