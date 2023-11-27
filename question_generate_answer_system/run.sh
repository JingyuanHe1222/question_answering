#!/bin/bash

VENV_NAME=".venv"
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <article_filename> <questions_filename>"
    exit 1
fi

ARTICLE_FILENAME=$1
QUESTIONS_FILENAME=$2

if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install it before running this script."
    exit 1
fi

if [ ! -d "$VENV_NAME" ]; then
    python3 -m venv $VENV_NAME
fi

source $VENV_NAME/bin/activate

echo "Installing required Python packages..."
pip3 install -r requirements.txt

echo "Running main.py..."
python3 main.py "$ARTICLE_FILENAME" "$QUESTIONS_FILENAME"

deactivate
