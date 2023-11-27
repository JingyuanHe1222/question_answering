import torch
import os
from pathlib import Path
from transformers import BertTokenizer, BertForQuestionAnswering
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import re
