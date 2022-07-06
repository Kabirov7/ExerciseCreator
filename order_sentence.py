import time

import spacy
import pandas as pd
import random

import texts
from utils_choose_verb import custom_tokenizer, test_order


nlp = spacy.load("en_core_web_trf")

nlp.tokenizer = custom_tokenizer(nlp)

df_frequency = pd.read_csv('wordFrequency.csv')

clean_texts = texts.TEXTS

vocab_num = 5000

for i in range(3):
    text = clean_texts[random.randint(0, len(clean_texts)) - 1]
    start = time.time()
    print(len(text))
    ex, sent = test_order(nlp, text, df_frequency, vocab_num)
    end = time.time()
    print(end - start)
    print(ex)
    print(sent)
    print()