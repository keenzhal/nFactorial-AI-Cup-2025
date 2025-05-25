# 🛡️ AI AGENT - Ваш цифровой телохранитель в Казахстане

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.0%2B-green.svg)](https://flask.palletsprojects.com/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.0%2B-yellow.svg)](https://huggingface.co/transformers/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-orange.svg)](https://opencv.org/)
[![Socket.IO](https://img.shields.io/badge/Socket.IO-4.0%2B-red.svg)](https://socket.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)](https://opensource.org/licenses/MIT)

## 🔍 Краткое описание

**AI AGENT** — это инновационная система, разработанная для защиты пользователей в Казахстане, особенно аудитории 30+, от растущей угрозы цифрового мошенничества. Проект сочетает в себе передовые AI-модели для анализа текстовых сообщений на предмет фишинга и других мошеннических схем, а также для детекции deepfake-видео, обеспечивая комплексную защиту в онлайн-коммуникациях.

## 📌 Проблема

Цифровое мошенничество становится все более изощренным, и Казахстан не является исключением. По неофициальным данным и анализу новостных сводок, значительная часть населения, особенно люди старше 30-35 лет, регулярно сталкиваются с попытками обмана через мессенджеры и социальные сети. Мошенники используют методы социальной инженерии, фишинговые ссылки, а в последнее время и deepfake-технологии для создания поддельных видеообращений. Это приводит к финансовым потерям и подрыву доверия к цифровым коммуникациям. Статистика показывает, что с каждым годом количество таких случаев растет, а целевая аудитория 30+ часто оказывается менее осведомленной о новых видах угроз.

## 💡 Решение

**AI AGENT** предоставляет многоуровневую защиту благодаря интеграции нескольких AI-компонентов:

1.  **Анализ текста на мошенничество:**
    *   **Модель:** Используется предобученная модель `DeepPavlov/rubert-base-cased-conversational` (RuBERT) из библиотеки Hugging Face Transformers.
    *   **Подход:** Текстовые сообщения векторизуются с помощью RuBERT для получения семантических эмбеддингов. На этих эмбеддингах обучается классификатор `LogisticRegression` (из `scikit-learn`) для определения вероятности мошеннического содержания. Данные для обучения (`text_train.json`) содержат примеры мошеннических и обычных текстов.
    *   **Реализация:** Функция `check_text_threat` принимает текст, генерирует его эмбеддинг и передает в обученный классификатор для оценки угрозы.

2.  **Детекция Deepfake в видео:**
    *   **Модель:** Применяется модель `Wvolf/ViT_Deepfake_Detection` (Vision Transformer) из Hugging Face Transformers, специализированная на обнаружении дипфейков.
    *   **Подход:** Видеофайл (MP4) обрабатывается покадрово. Для выбранных кадров (каждый 10-й кадр, максимум 30 кадров) модель ViT предсказывает вероятность того, что изображение является поддельным. Рассчитывается процент "подозрительных" кадров.
    *   **Реализация:** Функция `analyze_video` использует `cv2` (OpenCV) для чтения кадров, `PIL` (Pillow) для их обработки и `ViTForImageClassification` / `ViTImageProcessor` для анализа.

3.  **Интерактивная платформа:**
    *   **Технологии:** Веб-приложение на Flask с использованием Flask-SocketIO для обмена сообщениями в реальном времени.
    *   **Функционал:** Пользователи могут регистрироваться, общаться в чате, обмениваться текстовыми сообщениями и видео. Система автоматически анализирует контент и предупреждает пользователей о потенциальных угрозах.

## 🚀 Как это работает

Система функционирует следующим образом:

1.  **Вход/Регистрация:** Пользователь входит в систему через веб-интерфейс (`index.html`, `login.html`).
2.  **Чат:** Пользователь выбирает собеседника и начинает чат (`chat.html`).
3.  **Отправка сообщения:**
    *   **Текст:** При отправке текстового сообщения оно перехватывается сервером. Функция `check_text_threat` анализирует его. Если вероятность угрозы превышает 50%, и отправителю, и получателю (в случае получателя - от имени "AI Agent") отправляется предупреждение вместе с советами (`generate_ai_response`).
    *   **Видео:** Пользователь может загрузить видеофайл (MP4).
4.  **Анализ видео:**
    *   Видео загружается на сервер в папку `uploads`.
    *   Функция `analyze_video` обрабатывает видео, вычисляя процент кадров, которые модель считает поддельными.
    *   Клиенту, загрузившему видео, возвращается результат анализа (процент подделки, вердикт).
5.  **Отправка видеосообщения:**
    *   Информация о видео (включая URL на сервере, результат анализа) отправляется выбранному получателю через WebSocket (`on_video_message`).
    *   Если видео было распознано как возможно поддельное, получателю также отправляется системное предупреждение от "AI Agent".
6.  **Отображение:**
    *   Получатель видит текстовые сообщения и может просмотреть видео. Предупреждения от AI Agent отображаются особым образом.

**Схема работы (упрощенная):**

```
Пользователь А             Веб-сервер (Flask + SocketIO)                 Пользователь Б
-----------------             -----------------------------                 -----------------
1. Вводит текст/загружает видео
      |                                      |
      |---- Текст/Видео (HTTP POST/Socket) ->|
      |                                      |
      |                             2. Анализ текста (RuBERT + LogReg)
      |                             3. Анализ видео (ViT Deepfake)
      |                                      |
      |<-- Результат анализа видео (JSON) ---|
      |                                      |
      |---- Видео-сообщение (Socket) ------>|
      | (с URL видео и результатом)          |
      |                                      |---- Предупреждение AI (Socket) ->| (если угроза)
      |                                      |
      |                                      |---- Текст/Ссылка на видео ------>| 4. Отображение
```

## 🛠️ Технологический стек

| Технология                      | Назначение                                                                 |
|---------------------------------|----------------------------------------------------------------------------|
| Python 3.9+                     | Основной язык программирования                                               |
| Flask                           | Веб-фреймворк для создания API и обслуживания HTML страниц                 |
| Flask-SocketIO                  | Обеспечение двусторонней связи в реальном времени (чаты)                     |
| Hugging Face Transformers       | Библиотека для работы с state-of-the-art моделями (RuBERT, ViT)            |
| `DeepPavlov/rubert-base-cased-conversational` | Модель для получения эмбеддингов русскоязычного текста                    |
| `Wvolf/ViT_Deepfake_Detection`  | Модель для детекции deepfake в видео                                       |
| Scikit-learn (LogisticRegression) | Обучение классификатора для текстовых угроз                               |
| OpenCV (`cv2`)                  | Обработка видео: чтение кадров                                               |
| Pillow (`PIL`)                  | Обработка изображений (кадров видео) перед подачей в модель ViT             |
| PyTorch                         | Основной фреймворк для работы моделей из Transformers                      |
| NumPy                           | Численные операции, работа с массивами (эмбеддинги, данные для обучения)   |
| HTML5, CSS3, JavaScript       | Клиентская часть: интерфейс пользователя, взаимодействие с сервером        |
| `text_train.json`               | JSON-файл с обучающими данными для текстового классификатора                |

## ⚙️ Установка и запуск

Для участников хакатона:

1.  **Клонируйте репозиторий:**
    ```bash
    git clone <URL_ВАШЕГО_РЕПОЗИТОРИЯ>
    cd <НАЗВАНИЕ_ПАПКИ_ПРОЕКТА>
    ```

2.  **Создайте и активируйте виртуальное окружение:**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

3.  **Установите зависимости:**
    *   Рекомендуется создать файл `requirements.txt`. На основе `app.py` он может выглядеть так:
        ```
        flask
        flask-socketio
        numpy
        torch
        opencv-python
        Pillow
        transformers
        scikit-learn
        werkzeug
        # Для корректной работы SocketIO часто требуется
        python-engineio
        python-socketio
        ```
    *   Запустите установку:
        ```bash
        pip install -r requirements.txt
        ```
        *(Примечание: Если `requirements.txt` нет, установите пакеты вручную: `pip install flask flask-socketio numpy torch opencv-python Pillow transformers scikit-learn werkzeug python-engineio python-socketio`)*

4.  **Убедитесь, что у вас есть файл `text_train.json`** в корневой директории проекта. Этот файл необходим для обучения классификатора текстовых угроз.

5.  **Запустите приложение:**
    ```bash
    python app.py
    ```

6.  Откройте браузер и перейдите по адресу `http://127.0.0.1:5000/`.

## 🎮 Использование

1.  **Начальная страница (`/` или `/index.html`):** Введите имя пользователя для входа в чат.
    *(Скриншот главной страницы или формы входа)*

2.  **Интерфейс чата (`/chat`):**
    *   Слева отображается список доступных пользователей.
    *   Выберите пользователя для начала чата.
    *   Вводите текстовые сообщения или прикрепляйте видеофайлы (MP4).
    *(Скриншот интерфейса чата с примером сообщения и видео)*

3.  **Предупреждения AI Agent:**
    *   Если текстовое сообщение распознано как потенциально мошенническое, вы увидите предупреждение от "AI Agent" и рекомендации.
    *   Если отправленное или полученное видео имеет высокий процент deepfake, "AI Agent" также пришлет предупреждение.
    *(Скриншот чата с предупреждением от AI Agent)*

**Пример API ответа при загрузке видео (эндпоинт `/upload`):**
```json
{
    "message": "File analyzed",
    "video_url": "/uploads/example_video.mp4",
    "filename": "example_video.mp4",
    "result": {
        "percent_fake": 75.2,
        "verdict": "⚠️ ВНИМАНИЕ: Видео возможно поддельное!"
    }
}
```

## ソースコードのハイライト (Основные фрагменты исходного кода)

Ниже представлены ключевые участки кода из `app.py`, отвечающие за работу AI-компонентов.

### 1. Важные импорты для AI

```python
import numpy as np
import torch
import cv2
from PIL import Image
from transformers import (
    AutoTokenizer,
    AutoModel,
    ViTForImageClassification,
    ViTImageProcessor
)
from sklearn.linear_model import LogisticRegression
```

### 2. Загрузка AI Моделей

```python
# Загрузка моделей
text_model_name = "DeepPavlov/rubert-base-cased-conversational"
text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
text_model = AutoModel.from_pretrained(text_model_name)

video_model_name = "Wvolf/ViT_Deepfake_Detection"
video_model = ViTForImageClassification.from_pretrained(video_model_name)
video_processor = ViTImageProcessor.from_pretrained(video_model_name)
```

### 3. Анализ текста (Получение эмбеддингов и Классификация)

```python
def get_text_embedding(text):
    inputs = text_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = text_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()

# ... (пропущено чтение text_train.json) ...

def train_text_classifier():
    X_train = np.array([get_text_embedding(text) for text in texts_train])
    y_train = np.array(labels_train)
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    return clf

text_clf = train_text_classifier()

def check_text_threat(text):
    emb = get_text_embedding(text).reshape(1, -1)
    prob = text_clf.predict_proba(emb)[0][1] * 100
    print(f"AI Agent check: '{text}' -> Threat score: {prob:.2f}%")
    return prob
```

### 4. Анализ видео (Детекция Deepfake)

```python
def analyze_video(video_path, max_frames=30, step=10, threshold=0.6):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    analyzed = 0
    suspicious = 0

    while cap.isOpened() and analyzed < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % step == 0:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = video_processor(images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = video_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze().tolist()

            fake_score = probs[1] # Индекс 1 обычно соответствует "fake" классу
            # print(f"Frame {frame_count}: fake_score={fake_score:.3f}") # Для отладки

            if fake_score > threshold:
                suspicious += 1
            analyzed += 1

        frame_count += 1

    cap.release()
    percent_suspicious = (suspicious / analyzed) * 100 if analyzed else 0
    return percent_suspicious
```

### 5. Генерация ответа от AI Agent

```python
def generate_ai_response(message, sender, threat_score):
    base_msg = f"⚠️ Внимание! Сообщение от '{sender}' содержит потенциальную угрозу ({threat_score:.1f}%). "
    message_lower = message.lower()

    advices = []
    if any(word in message_lower for word in ['карта', 'номер карты', 'cvv', 'pin']):
        advices.append("Не передавайте данные банковской карты.")
    if any(word in message_lower for word in ['пароль', 'логин', 'аккаунт']):
        advices.append("Не сообщайте свои пароли.")
    if not advices:
        advices.append("Будьте осторожны с личной информацией.")

    return base_msg + ' '.join(advices)
```

###  Author

*   **Kalmakhan Dias (yeano9)** 


## 🚀 Будущие улучшения (Дорожная карта)

*   **Расширение датасета для текстовых угроз:** Сбор и разметка большего количества специфичных для Казахстана мошеннических текстов.
*   **Улучшение модели детекции текста:** Использование более продвинутых методов классификации или fine-tuning RuBERT на целевых данных.
*   **Анализ аудиосообщений:** Добавление функционала для детекции мошенничества в голосовых сообщениях (например, подделка голоса).
*   **Более гранулярный анализ видео:** Не просто "fake/real", а попытка определить тип манипуляции.
*   **Поддержка других мессенджеров/платформ:** Разработка API или плагинов для интеграции с популярными в Казахстане мессенджерами.
*   **Локализация на казахский язык:** Перевод интерфейса и адаптация моделей (если возможно) для казахского языка.
*   **База данных известных мошеннических номеров/ссылок:** Интеграция с черными списками.
*   **Персонализированные советы:** Более контекстно-зависимые рекомендации по безопасности.
*   **Мобильное приложение:** Для более удобного доступа.

---

*Создано с ❤️ для хакатона!* 