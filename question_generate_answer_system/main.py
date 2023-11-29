import sys
from answer_generator import WikiAnswerGenerator
from question_generator import SimpleYesQuestionGenerator
from qa_utils import QuestionAnswerWriter


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 main.py <article_filename> <questions_filename>")
        print("<article_filename> is the name of the article file")
        print(
            "<questions_filename> is the name of the file containing all questions generated from the article"
        )
        sys.exit(1)

    article_filename = sys.argv[1]
    questions_filename = sys.argv[2]

    qa_writer = QuestionAnswerWriter(question_filename=questions_filename)

    question_generator = SimpleYesQuestionGenerator(
        article_filename, questions_to_generate=20
    )
    questions = question_generator.generate_questions()
    for question in questions:
        print(question)

    with open(questions_filename, "r") as file:
        questions = file.readlines()

    answer_generator = WikiAnswerGenerator(article_filename, use_backup_model=True)

    qa_writer.start_new_session(mode=QuestionAnswerWriter.QUESTION_AND_ANSWER)
    for question in questions:
        answer = answer_generator.generate_answer(question)
        qa_writer.write_question_answer_pair_to_file(question, answer)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
