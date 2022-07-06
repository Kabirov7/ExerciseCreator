import spacy
import pandas as pd
import random
import json
from urllib.request import urlopen
from bs4 import BeautifulSoup, SoupStrainer
import httplib2
import re

from spacy.tokenizer import Tokenizer
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER, CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex
from spacy.matcher import Matcher


import lemminflect
from lemminflect import getInflection, getAllInflections


def custom_tokenizer(nlp):
    infixes = (
            LIST_ELLIPSES
            + LIST_ICONS
            + [
                r"(?<=[0-9])[+\-\*^](?=[0-9-])",
                r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
                    al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
                ),
                r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
                # r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
                r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
            ]
    )

    infix_re = compile_infix_regex(infixes)

    return Tokenizer(nlp.vocab,
                     prefix_search=nlp.tokenizer.prefix_search,
                     suffix_search=nlp.tokenizer.suffix_search,
                     infix_finditer=infix_re.finditer,
                     token_match=nlp.tokenizer.token_match,
                     rules=nlp.Defaults.tokenizer_exceptions)


def is_good(list_of_lemmas, df_word_frequency, vocab):
    '''Возвращает True если предложение подходит под уровень студента исходя из словарного запаса'''
    for i, row in df_word_frequency.iterrows():
        if row['lemma'] not in list_of_lemmas or row['rank'] > vocab:
            return False
        else:
            return True


def word_forms(lemma, original_word):
    '''Принимает слово и возвращает список из трёх вариантов, один из которых - верный'''
    # global options
    if original_word.text.lower() in ['am', 'is', 'are']:
        options = ['am', 'is', 'are']
    elif original_word.text.lower() in ['was', 'were']:
        options = ['was', 'were']
    elif original_word.text.lower() in ['have', 'had', 'has']:
        options = ['have', 'had', 'has']
    else:
        options = []
        inflections = getAllInflections(lemma, upos='VERB')

        for tag in inflections:
            if inflections[tag][0] != original_word.text.lower():
                options.append(inflections[tag][0])

        options = list(set(options))
        if len(options) > 2:
            for i in range(len(options) - 2):
                del options[random.randint(0, 2)]

        options.append(original_word.text.lower())
        options = random.sample(options, len(options))

    return options


def without_contractions(sentence):
    '''Возвращает предложение без сокращений'''
    contractions = ["don't", "doesn't", "didn't", "can't", "couldn't", "I've", "haven't", "hasn't", "hadn't", "I'll",
                    "he'll", "she'll", "it'll", "won't", "wouldn't", "I'm", "aren't"]
    full = ["do not", "does not", "did not", "cannot", "could not", "I have", "have not", "has not", "had not",
            "I will", "he will", "she will", "it will", "will not", "would not", "I am", "are not"]
    # abbreviation = {"I'll": 'I will', "I'm": 'I am', "I've": 'I have', "aren't": 'are not', "can't": 'cannot', "couldn't": 'could not', "didn't": 'did not', "doesn't": 'does not', "don't": 'do not', "hadn't": 'had not', "hasn't": 'has not', "haven't": 'have not', "he'll": 'he will', "it'll": 'it will', "she'll": 'she will', "won't": 'will not', "wouldn't": 'would not' }

    for i, cont in enumerate(contractions):
        if cont.lower() in sentence.lower():
            sentence = sentence.replace(cont, full[i])
    return sentence


def sent_w_brackets(verb, doc_sent):
    'Принимает токен глагол и док предложения и возвращает предложение с леммой глагола в скобках'
    for token in doc_sent:
        if verb == token:
            res = doc_sent[:token.i].text + ' (' + token.lemma_ + ') ' + doc_sent[token.i + 1:].text
    return res


def allowed_sentences(nlp, text, df_word_frequency, vocab_num):
    '''Принимает текст, таблицу лемм и число из этой таблицы (rank), а возвращает список из предложений, соответствующих уровню студента'''
    doc = nlp(text)  # todo it is stopped here
    right_sents = []
    for i, sent in enumerate(doc.sents):
        sent_lemmas = [token.lemma_ for token in sent if
                       token.pos_ != 'PROPN' and token.pos_ != 'PUNCT' and token.text != '’s']
        if is_good(sent_lemmas, df_word_frequency, vocab_num):
            sent = " ".join(sent.text.split())
            right_sents.append(sent)
    return right_sents


def token_verbs(nlp, splitted_sentence, doc):
    'Возвращает список глаголов, которые можно использовать для задания'
    matcher = Matcher(nlp.vocab)
    matcher.add('passive infinitive', [[{'LOWER': 'be'}, {'TAG': 'VBN'}]])
    matcher.add('h_been', [[{'LOWER': {'IN': ['have', 'has', 'had']}}, {'LOWER': 'been'}]])
    token_verbs = []
    verb_to_del = []
    for token in doc:
        span = doc[token.i:token.i + 2]
        if matcher(span):
            verb_to_del.append(doc[token.i + 1])
            if nlp.vocab.strings[matcher(doc)[0][0]] == 'h_been':
                token_verbs.append(token)
            continue
        else:
            to_be = ['be', 'was', 'were', 'been', 'being', 'am', 'is', 'are']
            if token.pos_ == 'VERB' or token.text.lower() in to_be:
                token_verbs.append(token)

    for i in verb_to_del:
        if i in token_verbs:
            token_verbs.remove(i)

    return token_verbs


def test_order(nlp, text, df_word_frequency, vocab_num):
    '''Задание: поставить слова в нужном порядке'''
    allowed = allowed_sentences(nlp, text, df_word_frequency, vocab_num)
    sentence = random.choice(allowed)
    splitted_sentence = sentence.split()
    shuffled_sentence = random.sample(splitted_sentence, len(splitted_sentence))

    for i, word in enumerate(shuffled_sentence):
        doc = nlp(word)
        for token in doc:
            if token.is_punct:
                shuffled_sentence[i] = word.replace(token.text, '')

    str_json = '''{
                  "title": "Упорядочить предложение",
                  "description": null,
                  "type": "object",
                  "required": [
                    "answer"
                  ],
                  "properties": {
                    "answer": {
                      "type": "string",
                      "title": "Ответ",
                      "default": "Ваш ответ"
                    },
                    "right_answer": {
                      "type": "string",
                      "title": "Правильный ответ:",
                      "default": null
                    }
                  }
                }'''

    ui_json = '''{
                "right_answer": {
                  "ui:disabled": true
                  }
                }'''

    exercise = ' / '.join(shuffled_sentence)

    data = json.loads(str_json)
    ui = json.loads(ui_json)
    data['description'] = exercise
    data['properties']['right_answer']['default'] = sentence
    new_json = json.dumps(data)
    new_ui = json.dumps(ui)

    # return new_json, new_ui

    return exercise, sentence


def test_choose_form(nlp, text, df_word_frequency, vocab_num):
    '''Задание: выбрать нужную форму слова'''
    allowed = allowed_sentences(nlp, text, df_word_frequency, vocab_num)
    flag = False
    while flag == False:
        sentence = random.choice(allowed)
        sentence = without_contractions(sentence)
        doc = nlp(sentence)

        splitted_sentence = [token for token in doc if not token.is_punct]
        verbs_token = token_verbs(nlp, splitted_sentence, doc)
        if not verbs_token:
            continue
        else:
            verb = verbs_token[random.randint(0, len(verbs_token) - 1)]
            lemma = verb.lemma_
            wrong_sentence = sent_w_brackets(verb, doc)
            test_options = word_forms(lemma, verb)
            flag = True

    str_json = '''{
                "type": "object",
                "title": "Выберите правильную форму слова в скобке",
                "properties": {
                  "numberEnumRadio": {
                    "type": "number",
                    "title": "sentence",
                    "enum": []
                  },
                  "right_answer": {
                    "type": "string",
                    "title": "Правильный ответ:",
                    "default": null
                  }
                }
              }'''

    ui_json = '''{
                  "numberEnumRadio": {
                    "ui:widget": "radio"
                  },
                 "right_answer": {
                    "ui:disabled": true
                  }
                }'''

    data = json.loads(str_json)
    ui = json.loads(ui_json)
    data['properties']['numberEnumRadio']['title'] = wrong_sentence
    data['properties']['numberEnumRadio']['enum'] = test_options
    data['properties']['right_answer']['default'] = sentence
    new_json = json.dumps(data)
    new_ui = json.dumps(ui)

    # return new_json, new_ui

    return sentence, wrong_sentence, test_options