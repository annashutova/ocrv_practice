import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import os
import string
import pandas as pd
from nltk import download
import pymorphy2

from processing import preprocess_emotions, balance_data_for_rnn

BASE_DIR = os.path.dirname(os.getcwd())

# Загрузите ваши данные
csv_data = pd.read_csv(f'{BASE_DIR}/labeled_data/text_data.csv')

# Подготовка данных
download('stopwords')
morph_analyzer = pymorphy2.MorphAnalyzer(lang='ru')
stop_words = set(stopwords.words('russian'))
trans_table = str.maketrans('', '', string.punctuation)
data = balance_data_for_rnn(preprocess_emotions(csv_data))

# Создайте словарь слов
word_counts = Counter()
processed_tokens = []
for text, _ in data:
    text = text.translate(trans_table)
    tokens = word_tokenize(text.lower())
    for token in tokens:
        token = token.lower()
        if token not in stop_words:
            token = morph_analyzer.parse(token)[0].normal_form
            processed_tokens.append(token)
    word_counts.update(processed_tokens)

vocabulary = {word: idx for idx, word in enumerate(word_counts.keys(), 1)}  # Добавляем 1, чтобы 0 было для padding

# Создание словаря для кодирования эмоций
emotion_to_idx = {
    "Neutral": 0,
    "Sadness": 1,
    "Anger": 2,
    "Fear": 3,
    # "Disgust": 2,
    "Surprise": 4,
    "Happiness": 5,
}


# Преобразование текста и эмоций в последовательности чисел
def encode_text(text):
    encoded_text = []
    text = text.translate(trans_table)

    tokens = word_tokenize(text.lower())
    for token in tokens:
        token = token.lower()
        if token not in stop_words:
            token = morph_analyzer.parse(token)[0].normal_form
            encoded_text.append(vocabulary.get(token, 0))

    return encoded_text


def encode_emotion(emotion):
    return emotion_to_idx[emotion]


# Создание датасета
class EmotionDataset(Dataset):
    def __init__(self, data):
        self.data = [(text, encode_emotion(emotion)) for text, emotion in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, emotion = self.data[idx]
        encoded_text = encode_text(text)
        return encoded_text, emotion


# Разделение данных на тренировочные, тестовые и валидационные
df = pd.DataFrame(data, columns=["text", "emotion"])

# Стратифицированное разделение на обучающие и временные данные
split1 = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_indices, temp_indices = next(split1.split(df, df["emotion"]))

train_data = df.iloc[train_indices]
temp_data = df.iloc[temp_indices]

# Стратифицированное разделение временных данных на валидационные и тестовые данные
split2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
val_indices, test_indices = next(split2.split(temp_data, temp_data["emotion"]))

val_data = temp_data.iloc[val_indices]
test_data = temp_data.iloc[test_indices]

# Преобразование обратно в списки
train_data = train_data.to_records(index=False)
val_data = val_data.to_records(index=False)
test_data = test_data.to_records(index=False)

train_dataset = EmotionDataset(train_data)
test_dataset = EmotionDataset(test_data)
validation_dataset = EmotionDataset(val_data)


# Функция для collate_fn, которая будет использоваться в DataLoader
def collate_fn(batch):
    texts, emotions = zip(*batch)
    filtered_batch = [(text, emotion) for text, emotion in zip(texts, emotions) if len(text) > 0]

    if len(filtered_batch) == 0:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])

    texts, emotions = zip(*filtered_batch)
    lengths = torch.tensor([len(text) for text in texts])
    padded_texts = nn.utils.rnn.pad_sequence([torch.tensor(text) for text in texts], batch_first=True, padding_value=0)
    emotions = torch.tensor(emotions)
    return padded_texts, emotions, lengths


# Создание DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)


# Создание модели RNN
class EmotionClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text, lengths):
        embedded = self.embedding(text)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.rnn(packed_embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        output = self.fc(hidden[-1])
        return output


# Параметры модели
vocab_size = len(vocabulary) + 1  # +1 для padding
embedding_dim = 128
hidden_dim = 256
output_dim = 6  # 6 эмоций

# Инициализация модели и оптимизатора
model = EmotionClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()


# Обучение модели
def train_model(model, train_loader, val_loader, epochs, optimizer, criterion):
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            text, emotion, length = batch
            if text.size(0) == 0:
                continue
            optimizer.zero_grad()
            output = model(text, length)
            loss = criterion(output, emotion)
            loss.backward()
            optimizer.step()

        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
        print(f"Validation loss: {val_loss}, Validation accuracy: {val_accuracy * 100:.2f}%")


# Тест и валидация модели
def evaluate_model(model, loader, criterion):
    model.eval()
    total_correct = 0
    total_examples = 0
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            text, emotion, length = batch
            if text.size(0) == 0:
                continue
            output = model(text, length)
            loss = criterion(output, emotion)
            total_loss += loss.item()
            _, predicted = torch.max(output, dim=1)
            total_correct += (predicted == emotion).sum().item()
            total_examples += len(emotion)
    average_loss = total_loss / len(loader)
    accuracy = total_correct / total_examples
    return average_loss, accuracy


def final_test_model(model, test_loader, criterion):
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion)
    print(f"Loss: {test_loss}, Accuracy: {test_accuracy * 100:.2f}%")


epochs = 10
train_model(model, train_loader, validation_loader, epochs, optimizer, criterion)
final_test_model(model, test_loader, criterion)


# Использование модели для классификации
def predict_emotion(text):
    encoded_text = encode_text(text)

    if len(encoded_text) == 0:
        return "Unknown"  # Обозначает пустую последовательность

    encoded_text = torch.tensor(encoded_text).unsqueeze(0)
    length = torch.tensor([len(encoded_text[0])])
    output = model(encoded_text, length)
    _, predicted = torch.max(output, dim=1)
    emotion_labels = ["Sadness", "Fear", "Anger", "Surprise", "Happiness", "Neutral"]
    return emotion_labels[predicted.item()]


# Пример использования
example_text = "Я так рад, что ты приехал!"
predicted_emotion = predict_emotion(example_text)
print(f"Predicted emotion: {predicted_emotion}")
