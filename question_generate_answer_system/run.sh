#!/bin/bash

# Function to display usage
display_usage() {
    echo "Usage: $0 --article article_filename [--questions questions_filename] [--questions_to_generate number_of_questions] [--questions_only] [--llm] [--answers answers_filename]"
    echo "   --article                         - Required: The filename of the article."
    echo "   --questions                       - Optional: Specifies a .txt file containing pre-written questions. The program reads these"
    echo "                                                 questions and generates corresponding answers. If this flag is not provided, the program"
    echo "                                                 will automatically generate questions based on the article content."
    echo "   --questions_to_generate           - Optional: The number of questions to generate. Default is 10. This flag will be ignored if the --questions"
    echo "                                                 flag is supplied."
    echo "   --questions_only                  - Optional: Supply this flag if you want the system to only generate questions and skip answer generation. "
    echo "   --answers                         - Optional: Specifies the filename for saving the generated question and answer pairs. If this flag is not used, "
    echo "                                                 the default filename 'generated_question_answer_pairs.txt' will be applied."    
    echo "   --llm                             - Optional: Enable Mistral 7B LLM for answer generation. Be aware that Mistral 7B can only be ran on an A100 GPU or any better GPUs thereof."
}

# Display usage if no arguments or -h is the first argument
if [ "$#" -eq 0 ] || [ "$1" == "-h" ]; then
    display_usage
    exit 0
fi

if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install it before running this script."
    exit 1
fi

VENV_NAME=".venv"
if [ ! -d "$VENV_NAME" ]; then
    python3 -m venv $VENV_NAME
fi

source $VENV_NAME/bin/activate

echo "Installing required Python packages..."
pip3 install -q -r requirements.txt

echo "Running main.py with provided arguments..."
python3 main.py "$@"

deactivate
