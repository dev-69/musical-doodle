# tokenizer_utils.py

import string
import nltk
from nltk.tokenize import word_tokenize

# Ensure necessary data is downloaded
nltk.download('punkt', quiet=True)

def tokenizer_better(text):
    punc_list = string.punctuation + '0123456789'
    t = str.maketrans(dict.fromkeys(punc_list, " "))
    text = text.lower().translate(t)
    tokens = word_tokenize(text)
    return tokens

my_stop_words = [
    'the', 'and', 'to', 'of', 'was', 'with', 'a', 'on', 'in', 'for', 'name',
    'is', 'patient', 's', 'he', 'at', 'as', 'or', 'one', 'she', 'his', 'her', 'am',
    'were', 'you', 'pt', 'pm', 'by', 'be', 'had', 'your', 'this', 'date',
    'from', 'there', 'an', 'that', 'p', 'are', 'have', 'has', 'h', 'but', 'o',
    'namepattern', 'which', 'every', 'also'
]
