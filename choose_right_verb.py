import time

import spacy
import pandas as pd
import random

import texts
from utils_choose_verb import custom_tokenizer, test_choose_form


nlp = spacy.load("en_core_web_trf")

nlp.tokenizer = custom_tokenizer(nlp)

df_frequency = pd.read_csv('wordFrequency.csv')

clean_texts = texts.TEXTS

vocab_num = 5000

# test_choose_form() random texts
for i in range(3):
    text = clean_texts[random.randint(0, len(clean_texts)) - 1]
    sentence, wrong_sentence, test_options = test_choose_form(nlp, text, df_frequency, vocab_num)
    print(sentence)
    print(wrong_sentence)
    print(test_options)
    print()
