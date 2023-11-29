from qa_generator import QAGeneratorWithCache
import qa_utils
import re


class SimpleYesQuestionGenerator(QAGeneratorWithCache):
    def __init__(
        self,
        article_filename,
        model_name="deepset/bert-base-cased-squad2",
        questions_to_generate=10,
    ):
        super().__init__(article_filename, model_name)
        self.questions_to_generate = questions_to_generate
        self.regex_pattern = re.compile(r"(\w+) (\w+) ([^;:,]+)")
        self.excluded_subjects = {
            "it",
            "this",
            "that",
            "he",
            "him",
            "these",
            "those",
            "there",
            "here",
            "they",
            "she",
            "her",
        }

    def generate_questions(self):
        article = self._load_article()
        sentences = qa_utils.split_text_into_sentences(article)
        questions = []
        for sentence in sentences:
            question = self._transform_sentence_to_question(sentence)
            if self.filter_question(question):
                questions.append(question)
                if len(questions) >= self.questions_to_generate:
                    break
        return questions

    def filter_question(self, question):
        return question is not None  # and self.article_name.lower() in question.lower()

    def _transform_sentence_to_question(self, sentence):
        sentence = sentence.strip()
        if sentence and sentence[-1] in ",.;":
            sentence = sentence[:-1]
        if not sentence.lower().startswith(self.article_name.lower()):
            return None
        words = sentence.split()

        # Find the position where the article name ends
        article_name_words = self.article_name.split()
        subject_end_index = len(article_name_words)

        # Reconstruct the subject from the sentence
        subject = " ".join(words[:subject_end_index])

        # The rest of the sentence after the subject
        rest_of_sentence = " ".join(words[subject_end_index:])

        # Try to find the verb and the rest of the sentence
        # Assuming the sentence format is 'Subject Verb Rest'
        verb_and_rest = rest_of_sentence.split(" ", 1)

        if len(verb_and_rest) >= 2:
            verb, rest = verb_and_rest
            # Check if verb is one of the specified verbs
            if verb.lower() in ["is", "are", "was", "were", "has", "had"]:
                verb = verb.capitalize()
                return f"{verb} {subject} {rest}?"
        return None
        # match = self.regex_pattern.match(sentence)
        # if match:
        #     subject, verb, rest = match.groups()
        #     subject = subject.lower()
        #     if (
        #         verb.lower() in ["is", "are", "was", "were"]
        #         and subject not in self.excluded_subjects
        #     ):
        #         verb = verb.capitalize()
        #         return f"{verb} {subject} {rest}?"
        # return None
