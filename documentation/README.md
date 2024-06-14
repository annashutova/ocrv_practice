## Описание:

Данный репозиторий содержит набор размеченных данных, предназначенных для обучения и тестирования алгоритмов определения 
алкогольного опьянения и эмоций. Также в репозитории представлены сопутствующие инструменты, помогающие в обработке и анализе данных.

## Структура:
```
├── labeled_data
│   ├── audio_data.csv
│   └── text_data.csv
├── raw_data
│   ├── audio
│   │   ├── video_title_1
│   │   │   ├── video_title_1_chunk0.wav
│   │   │   └── ...
│   │   ├── video_title_2
│   │   │   ├── video_title_2_chunk0.wav
│   │   │   └── ...
│   │   └── ...
│   ├── text
│   │   ├── video_title_1
│   │   │   ├── video_title_1_chunk0.txt
│   │   │   └── ...
│   │   ├── video_title_2
│   │   │   ├── video_title_2_chunk0.txt
│   │   │   └── ...
│   │   └── ...
│   └── video
│       ├── video_title_1.mp4
│       ├── video_title_2.mp4
│       └── ...
├── tools
│   ├── config.py
│   ├── logreg_alco_classifier.py
│   ├── logreg_emotion_clissifier.py
│   ├── processing.py
│   ├── rnn_emotion_classifier.py
│   ├── split_video.py
│   └── whisper_transcription.ipynb
├── documentation
│   └── README.md
└── requirements.txt
```

## Описание директорий:

labeled_data:
- audio_data.csv: Файл с размеченными данными по аудио файлам.
- text_data.csv: Файл с размеченными данными по текстовым файлам.

raw_data:
- audio: Содержит исходные, необработанные аудио сегменты видео.
  - video_title_1, video_title_2 ...: Папки с сегментами аудио для каждого видео.
    - video_title_1_chunk0.wav, video_title_2_chunk0.wav ...: Аудио файлы формата wav - сегменты видео.
- text: Содержит исходные, необработанные текстовые сегменты видео.
  - video_title_1, video_title_2 ...: Папки с сегментами текста для каждого видео.
    - video_title_1_chunk0.wav, video_title_2_chunk0.wav ...: Текстовые файлы формата txt - сегменты видео.
- video: Содержит исходные, необработанные видео.
  - video_title_1.mp4, video_title_2.mp4 ...: Видео файлы формата mp4.

rejected_data: Содержит данные не использованные в разметке.
- audio: Содержит исходные, необработанные аудио сегменты видео.
  - video_title_1, video_title_2 ...: Папки с сегментами аудио для каждого видео.
    - video_title_1_chunk0.wav, video_title_2_chunk0.wav ...: Аудио файлы формата wav - сегменты видео.
- video: Содержит исходные, необработанные видео.
  - video_title_1.mp4, video_title_2.mp4 ...: Видео файлы формата mp4.

tools:
  - config.py: Конфигурации для разделения видео на сегменты.
  - logreg_alco_classifier.py: Предобработка данных и обучение модели классификации алкогольного опьянения.
  - logreg_emotion_clissifier.py: Предобработка данных и обучение модели классификации эмоций.
  - processing.py: Функции для предобработки данных.
  - rnn_emotion_classifier.py: Скрипт для обучения модели RNN.
  - split_video.py: Скрипт для разделения видео на сегменты по тишине и сохранения их как wav файлов.
  - whisper_transcription.ipynb: Jupiter Notebook скрипт для транскрибирования аудио сегментов.

documentation:
  - README.md: Описание репозитория.

requirements.txt: Файл с зависимостями.

## Использование:

1. Клонируйте репозиторий.
2. Используйте скрипты в директории "tools" для обработки данных, обучения моделей и оценки результатов.
3. Используйте документацию в директории "documentation" для понимания структуры репозитория и описания данных.
