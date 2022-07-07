def hello_world(request):
    import random
    import texts
    from ExerciseCreator import ExerciseCreator, custom_tokenizer
    
    creator = ExerciseCreator('en_core_web_trf', 'https://storage.googleapis.com/media_journal_bucket/wordFrequency.csv', custom_tokenizer, texts.TEXTS)
    text = texts.TEXTS[random.randint(0, len(texts.TEXTS)) - 1]
    ex, sent = creator.shuffle_sentence_exercise(text, 5000)

    return {"ex":ex, "sent":sent}