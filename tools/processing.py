import json
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


def preprocess_data(csv_data: pd.DataFrame, morph_analyzer: MorphAnalyzer) -> pd.DataFrame:
    """
    Предобработка csv данных: предобработка текста, разделение строк с множественным выбором эмоций.
    """
    additional_rows = []
    for i in csv_data.index:
        csv_data.at[i, 'text'] = preprocess_text(csv_data.at[i, 'text'], morph_analyzer)

        if 'choices' in csv_data.at[i, 'emotion']:
            choices = json.loads(csv_data.at[i, 'emotion'])['choices']
            for choice in choices:
                new_row = [csv_data.at[i, 'alcohol'],
                           csv_data.at[i, 'annotation_id'],
                           csv_data.at[i, 'annotator'],
                           csv_data.at[i, 'created_at'],
                           choice,
                           csv_data.at[i, 'id'],  # NOTE id дублируется
                           csv_data.at[i, 'lead_time'],
                           csv_data.at[i, 'multiple_people'],
                           csv_data.at[i, 'text'],
                           csv_data.at[i, 'updated_at']]
                additional_rows.append(pd.DataFrame([new_row], columns=csv_data.columns))
        csv_data = csv_data.drop(index=i)
    return pd.concat([csv_data] + additional_rows, ignore_index=True)
