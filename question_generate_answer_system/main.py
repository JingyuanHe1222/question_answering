import argparse
from answer_generators.wiki_answer_generator import WikiAnswerGenerator
from answer_generators.llm_answer_generator import LLMAnswerGenerator

# from question_generators.simple_yes_question_generator import SimpleYesQuestionGenerator
import qa_utils
from qa_utils import QuestionAnswerWriter
from question_generators.t5_question_generator import T5QuestionGenerator


def main():
    parser = argparse.ArgumentParser(
        description="This script generates questions from a given article file and writes them along with their answers to a specified output file.",
    )
    parser.add_argument("--article", type=str, help="Local article filename")

    # Optional arguments
    parser.add_argument(
        "--questions",
        type=str,
        help="Output file for generated questions",
        default=None,
    )
    parser.add_argument(
        "--questions_to_generate",
        type=int,
        help="Number of questions to generate",
        default=10,
    )
    parser.add_argument(
        "--answers",
        type=str,
        help="Output file for generated questions and answer pairs",
        default=None,
    )
    parser.add_argument(
        "--questions_only",
        action="store_true",
        help="Generate only questions without answers",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Generate only questions without answers",
    )
    args = parser.parse_args()
    article_filename = args.article
    questions_filename = args.questions
    questions_to_generate = args.questions_to_generate
    answer_output_file = args.answers
    enable_llm = args.llm

    qa_writer = QuestionAnswerWriter()
    if answer_output_file:
        qa_writer.set_qa_pair_filename(answer_output_file)

    questions_to_answer = []
    # no question filename is supplied, generate questions from article
    if questions_filename is None:
        question_generator = T5QuestionGenerator(
            article_filename, questions_to_generate=questions_to_generate
        )
        questions_to_answer = question_generator.generate_questions()

        qa_writer.start_new_session(mode=QuestionAnswerWriter.QUESTION)
        qa_writer.write_questions_to_file(questions_to_answer)
    # Otherwise read pre-written questions from file
    else:
        questions_to_answer = qa_utils.read_questions_from_file(questions_filename)

    # Stop execution if --questions_only flag is used
    if args.questions_only:
        return

    answer_generator_llm = None
    if enable_llm:
        answer_generator_llm = LLMAnswerGenerator(article_filename)

    answer_generator = WikiAnswerGenerator(article_filename)
    qa_writer.start_new_session(mode=QuestionAnswerWriter.QUESTION_AND_ANSWER)
    for question in questions_to_answer:
        if enable_llm and qa_utils.is_yes_no_question(question):
            answer = answer_generator_llm.generate_answer(question)
        else:
            answer = answer_generator.generate_answer(question)
        qa_writer.write_question_answer_pair_to_file(question, answer)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
