import json
import random
import nltk
import pandas as pd
from pymorphy2 import MorphAnalyzer


def preprocess_text(text: str, morph_analyzer: MorphAnalyzer):
    """
    Предобработка текста: токенизация, удаление стоп-слов, нормализация.
    """
    tokens = nltk.word_tokenize(text)
    processed_tokens = []

    stop_words = set(nltk.corpus.stopwords.words('russian'))

    for token in tokens:
        token = token.lower()
        if token not in stop_words:
            token = morph_analyzer.parse(token)[0].normal_form
            processed_tokens.append(token)

    return " ".join(processed_tokens)


def preprocess_emotions(csv_data: pd.DataFrame) -> pd.DataFrame:
    """
    Предобработка csv данных: предобработка текста, разделение строк с множественным выбором эмоций.
    """
    additional_rows = []

    for i in csv_data.index:
        if 'choices' in csv_data.at[i, 'emotion']:
            choices = json.loads(csv_data.at[i, 'emotion'])['choices']
            for choice in choices:
                new_row = {
                    'alcohol': csv_data.at[i, 'alcohol'],
                    'annotation_id': csv_data.at[i, 'annotation_id'],
                    'annotator': csv_data.at[i, 'annotator'],
                    'created_at': csv_data.at[i, 'created_at'],
                    'emotion': choice,
                    'id': csv_data.at[i, 'id'],  # NOTE id будет дублироваться
                    'lead_time': csv_data.at[i, 'lead_time'],
                    'multiple_people': csv_data.at[i, 'multiple_people'],
                    'text': csv_data.at[i, 'text'],
                    'updated_at': csv_data.at[i, 'updated_at']
                }
                additional_rows.append(new_row)
        else:
            new_row = {
                'alcohol': csv_data.at[i, 'alcohol'],
                'annotation_id': csv_data.at[i, 'annotation_id'],
                'annotator': csv_data.at[i, 'annotator'],
                'created_at': csv_data.at[i, 'created_at'],
                'emotion': csv_data.at[i, 'emotion'],
                'id': csv_data.at[i, 'id'],  # NOTE id будет дублироваться
                'lead_time': csv_data.at[i, 'lead_time'],
                'multiple_people': csv_data.at[i, 'multiple_people'],
                'text': csv_data.at[i, 'text'],
                'updated_at': csv_data.at[i, 'updated_at']
            }
            additional_rows.append(new_row)

    return pd.DataFrame(additional_rows, columns=csv_data.columns)


def preprocess_data_for_rnn(csv_data: pd.DataFrame) -> list[tuple[str, str]]:
    data_for_rnn = []

    for _, row in csv_data.iterrows():
        data_for_rnn.append((row['text'], row['emotion']))

    return data_for_rnn


def balance_data_for_rnn(csv_data: pd.DataFrame) -> list[tuple[str, str]]:
    data_for_rnn = []
    neutral_data = []
    emotions_cnt = {
        'Happiness': 0,
        'Sadness': 0,
        'Anger': 0,
        'Fear': 0,
        'Neutral': 0,
        'Surprise': 0
    }

    for _, row in csv_data.iterrows():
        emotion = row['emotion']
        if emotion == 'Neutral':
            neutral_data.append((row['text'], 'Neutral'))
        elif emotion == 'Disgust':
            emotions_cnt['Fear'] += 1
            data_for_rnn.append((row['text'], 'Fear'))
        else:
            emotions_cnt[emotion] += 1
            data_for_rnn.append((row['text'], emotion))

    # Случайным образом выбираем 1/4 записей из neutral_data
    neutral_sample_size = len(neutral_data) // 4
    emotions_cnt['Neutral'] += neutral_sample_size
    if neutral_sample_size > 0:
        neutral_sample = random.sample(neutral_data, neutral_sample_size)
        data_for_rnn.extend(neutral_sample)

    print(emotions_cnt)

    return data_for_rnn
