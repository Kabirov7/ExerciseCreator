import time
import random

import texts
from ExerciseCreator import ExerciseCreator, custom_tokenizer


creator = ExerciseCreator('en_core_web_trf', 'wordFrequency.csv', custom_tokenizer, texts.TEXTS)

for i in range(3):
    text = texts.TEXTS[random.randint(0, len(texts.TEXTS)) - 1]
    start = time.time()
    print(len(text))
    ex, sent = creator.shuffle_sentence_exercise(text, 5000)
    print(time.time() - start)
    print(ex)
    print(sent)
    print()

for i in range(3):
    text = texts.TEXTS[random.randint(0, len(texts.TEXTS)) - 1]
    start = time.time()
    print(len(text))
    sentence, wrong_sentence, test_options = creator.verb_form_exercise(text, 5000)
    print(time.time() - start)
    print(sentence)
    print(wrong_sentence)
    print(test_options)
    print()