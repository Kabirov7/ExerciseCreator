import spacy
import pandas as pd
import random
from lemminflect import getAllInflections
from spacy.matcher import Matcher
from spacy.tokenizer import Tokenizer
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER, CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex


class ExerciseCreator:
    def __init__(self, model_name, frequency_file_csv, tokenizer, texts):
        self.nlp = spacy.load(model_name)
        self.nlp.tokenizer = tokenizer(self.nlp)
        self.df_frequency = pd.read_csv(frequency_file_csv)
        self.texts = texts

    def is_user_lvl(self, list_of_lemmas, vocab):
        '''Возвращает True если предложение подходит под уровень студента исходя из словарного запаса'''
        for i, row in self.df_frequency.iterrows():
            if row['lemma'] not in list_of_lemmas or row['rank'] > vocab:
                return False
            else:
                return True

    def word_forms(self, lemma, original_word):
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

    def without_contractions(self, sentence):
        '''Возвращает предложение без сокращений'''
        contractions = ["don't", "doesn't", "didn't", "can't", "couldn't", "I've", "haven't", "hasn't", "hadn't",
                        "I'll",
                        "he'll", "she'll", "it'll", "won't", "wouldn't", "I'm", "aren't"]
        full = ["do not", "does not", "did not", "cannot", "could not", "I have", "have not", "has not", "had not",
                "I will", "he will", "she will", "it will", "will not", "would not", "I am", "are not"]

        for i, cont in enumerate(contractions):
            if cont.lower() in sentence.lower():
                sentence = sentence.replace(cont, full[i])
        return sentence

    def sent_w_brackets(self, verb, doc_sent):
        'Принимает токен глагол и док предложения и возвращает предложение с леммой глагола в скобках'
        for token in doc_sent:
            if verb == token:
                res = doc_sent[:token.i].text + ' (' + token.lemma_ + ') ' + doc_sent[token.i + 1:].text
                return res

    def allowed_sentences(self, text, vocab_num):
        '''Принимает текст, таблицу лемм и число из этой таблицы (rank), а возвращает список из предложений, соответствующих уровню студента'''
        doc = self.nlp(text)  # todo it is stopped here
        right_sents = []
        for i, sent in enumerate(doc.sents):
            sent_lemmas = [token.lemma_ for token in sent if
                           token.pos_ != 'PROPN' and token.pos_ != 'PUNCT' and token.text != '’s']
            if self.is_user_lvl(sent_lemmas, vocab_num):
                sent = " ".join(sent.text.split())
                right_sents.append(sent)
        return right_sents

    def token_verbs(self, doc):
        'Возвращает список глаголов, которые можно использовать для задания'
        matcher = Matcher(self.nlp.vocab)
        matcher.add('passive infinitive', [[{'LOWER': 'be'}, {'TAG': 'VBN'}]])
        matcher.add('h_been', [[{'LOWER': {'IN': ['have', 'has', 'had']}}, {'LOWER': 'been'}]])
        token_verbs = []
        verb_to_del = []
        for token in doc:
            span = doc[token.i:token.i + 2]
            if matcher(span):
                verb_to_del.append(doc[token.i + 1])
                if self.nlp.vocab.strings[matcher(doc)[0][0]] == 'h_been':
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

    def shuffle_sentence_exercise(self, text, vocab_num):
        '''Задание: поставить слова в нужном порядке'''
        allowed = self.allowed_sentences(text, vocab_num)
        sentence = random.choice(allowed)
        splitted_sentence = sentence.split()
        shuffled_sentence = random.sample(splitted_sentence, len(splitted_sentence))

        for i, word in enumerate(shuffled_sentence):
            doc = self.nlp(word)
            for token in doc:
                if token.is_punct:
                    shuffled_sentence[i] = word.replace(token.text, '')

        exercise = ' / '.join(shuffled_sentence)

        return exercise, sentence

    def verb_form_exercise(self, text, vocab_num):
        '''Задание: выбрать нужную форму слова'''
        allowed = self.allowed_sentences(text, vocab_num)
        flag = False

        while flag == False:
            sentence = random.choice(allowed)
            sentence = self.without_contractions(sentence)
            doc = self.nlp(sentence)

            splitted_sentence = [token for token in doc if not token.is_punct]
            verbs_token = self.token_verbs(doc)
            if not verbs_token:
                continue
            else:
                verb = verbs_token[random.randint(0, len(verbs_token) - 1)]
                lemma = verb.lemma_
                wrong_sentence = self.sent_w_brackets(verb, doc)
                test_options = self.word_forms(lemma, verb)
                flag = True

        return sentence, wrong_sentence, test_options


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