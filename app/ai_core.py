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
import json

# Загрузка моделей
TEXT_MODEL_NAME = "DeepPavlov/rubert-base-cased-conversational"
TEXT_TOKENIZER = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
TEXT_MODEL = AutoModel.from_pretrained(TEXT_MODEL_NAME)

VIDEO_MODEL_NAME = "Wvolf/ViT_Deepfake_Detection"
VIDEO_MODEL = ViTForImageClassification.from_pretrained(VIDEO_MODEL_NAME)
VIDEO_PROCESSOR = ViTImageProcessor.from_pretrained(VIDEO_MODEL_NAME)

TEXT_TRAIN_DATA_PATH = "data/text_train.json" # Обновленный путь

def get_text_embedding(text):
    inputs = TEXT_TOKENIZER(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = TEXT_MODEL(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()


def train_text_classifier():
    try:
        with open(TEXT_TRAIN_DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {TEXT_TRAIN_DATA_PATH} not found. Text classifier will not be trained.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode {TEXT_TRAIN_DATA_PATH}. Text classifier will not be trained.")
        return None

    texts_train = [item["text"] for item in data]
    labels_train = [item["label"] for item in data]

    if not texts_train or not labels_train:
        print("Error: No training data or labels found. Text classifier will not be trained.")
        return None
        
    X_train = np.array([get_text_embedding(text) for text in texts_train])
    y_train = np.array(labels_train)
    
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    print("Text classifier trained successfully.")
    return clf

TEXT_CLASSIFIER = train_text_classifier()

def check_text_threat(text):
    if TEXT_CLASSIFIER is None:
        print("Text classifier is not available. Skipping threat check.")
        return 0.0 # Возвращаем нейтральное значение, если классификатор не обучен

    emb = get_text_embedding(text).reshape(1, -1)
    prob = TEXT_CLASSIFIER.predict_proba(emb)[0][1] * 100
    print(f"AI AGENT check: '{text}' -> Threat score: {prob:.2f}%")
    return prob

def analyze_video(video_path, max_frames=30, step=10, threshold=0.6):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return 0.0
        
    frame_count = 0
    analyzed = 0
    suspicious = 0

    while cap.isOpened() and analyzed < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % step == 0:
            try:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                inputs = VIDEO_PROCESSOR(images=image, return_tensors="pt")

                with torch.no_grad():
                    outputs = VIDEO_MODEL(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze().tolist()
                
                fake_score = probs[1]
                # print(f"Frame {frame_count}: fake_score={fake_score:.3f}")

                if fake_score > threshold:
                    suspicious += 1
                analyzed += 1
            except Exception as e:
                print(f"Error processing frame {frame_count} from {video_path}: {e}")

        frame_count += 1

    cap.release()
    if analyzed == 0:
        print(f"Warning: No frames were analyzed from video {video_path}. This could be due to a very short video or issues reading frames.")
        return 0.0
        
    percent_suspicious = (suspicious / analyzed) * 100
    return percent_suspicious

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