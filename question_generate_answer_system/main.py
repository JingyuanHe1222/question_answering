import sys
from wiki_answer_generator import WikiAnswerGenerator


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
    # with open(article_filename, 'r') as file:
    #     article_content = file.read()
    with open(questions_filename, "r") as file:
        questions = file.readlines()

    answer_generator = WikiAnswerGenerator(article_filename, use_backup_model=True)
    for question in questions:
        answer = answer_generator.generate_answer(question)
        print(f"Q: {question}")
        print(f"A: {answer}\n\n")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
