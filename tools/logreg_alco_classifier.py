import nltk
import pymorphy2
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


BASE_DIR = os.path.dirname(os.getcwd())

# Загрузка данных (предполагается, что у вас есть файл с данными в формате CSV)
data = pd.read_csv(f'{BASE_DIR}/labeled_data/text_data.csv')

# Предобработка текста
nltk.download('punkt')
nltk.download('stopwords')
morph = pymorphy2.MorphAnalyzer(lang='ru')


def preprocess_text(text: str):
    """
    Предобработка текста: токенизация, удаление стоп-слов, нормализация.
    """
    tokens = text.split()
    processed_tokens = set()

    stop_words = set(nltk.corpus.stopwords.words('russian'))

    for token in tokens:
        token = token.lower()
        if token not in stop_words:
            token = morph.parse(token)[0].normal_form
            processed_tokens.add(token)

    return " ".join(tokens)


data['text'] = data['text'].apply(preprocess_text)

# Разделение данных на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['alcohol'], test_size=0.2, random_state=42)

# Векторизация текста с использованием TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Обучение модели логистической регрессии
model = LogisticRegression(random_state=42)
model.fit(X_train_vectorized, y_train)

# Предсказание на тестовом наборе
y_pred = model.predict(X_test_vectorized)

# Оценка точности модели
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy:.2f}")

# Вывод отчета о классификации
print(classification_report(y_test, y_pred))
