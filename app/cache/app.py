from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
from flask_socketio import SocketIO, emit, disconnect
import os
import json
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
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4'}
socketio = SocketIO(app, cors_allowed_origins="*")

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Загрузка моделей
text_model_name = "DeepPavlov/rubert-base-cased-conversational"
text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
text_model = AutoModel.from_pretrained(text_model_name)

video_model_name = "Wvolf/ViT_Deepfake_Detection"
video_model = ViTForImageClassification.from_pretrained(video_model_name)
video_processor = ViTImageProcessor.from_pretrained(video_model_name)

def get_text_embedding(text):
    inputs = text_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = text_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()

with open("text_train.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts_train = [item["text"] for item in data]
labels_train = [item["label"] for item in data]

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

            fake_score = probs[1]
            print(f"Frame {frame_count}: fake_score={fake_score:.3f}")

            if fake_score > threshold:
                suspicious += 1
            analyzed += 1

        frame_count += 1

    cap.release()
    percent_suspicious = (suspicious / analyzed) * 100 if analyzed else 0
    return percent_suspicious

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

users_sid = {}
users_name = {}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        username = request.form.get('username')
        if username and username not in users_sid:
            return redirect(url_for('chat', username=username))
        else:
            return render_template('login.html', error="Имя занято или не указано")
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        if username and username not in users_sid:
            return redirect(url_for('chat', username=username))
        else:
            return render_template('login.html', error="Имя занято или не указано")
    return render_template('login.html')

@app.route('/chat')
def chat():
    username = request.args.get('username')
    if not username:
        return redirect(url_for('index'))
    return render_template('chat.html', username=username)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        percent_fake = analyze_video(filepath)
        verdict = "⚠️ ВНИМАНИЕ: Видео возможно поддельное!" if percent_fake > 50 else "✅ Видео скорее настоящее."
        video_url = url_for('uploaded_file', filename=filename)

        return jsonify({
            'message': 'File analyzed',
            'video_url': video_url,
            'filename': filename,
            'result': {
                'percent_fake': round(percent_fake, 1),
                'verdict': verdict
            }
        })

    return jsonify({'error': 'Invalid file type'}), 400

@socketio.on('join')
def on_join(data):
    username = data.get('username')
    sid = request.sid
    if not username:
        disconnect()
        return

    old_sid = users_sid.get(username)
    if old_sid and old_sid != sid:
        emit('system_message', {
            'message': 'Ваша сессия была прервана из-за повторного входа с другим устройством.'
        }, room=old_sid)
        disconnect(old_sid)

    users_sid[username] = sid
    users_name[sid] = username
    emit_user_list()

@socketio.on('disconnect')
def on_disconnect():
    sid = request.sid
    username = users_name.get(sid)
    if username:
        users_sid.pop(username, None)
        users_name.pop(sid, None)
        emit_user_list()

def emit_user_list():
    for sid in users_name.keys():
        username = users_name[sid]
        other_users = [u for u in users_sid.keys() if u != username]
        emit('user_list', {'users': other_users}, room=sid)

@socketio.on('private_message')
def on_private_message(data):
    sender_sid = request.sid
    sender = users_name.get(sender_sid)
    recipient = data.get('to')
    message = data.get('message', '').strip()

    if not sender or not recipient or not message:
        return

    recipient_sid = users_sid.get(recipient)
    if not recipient_sid:
        emit('system_message', {'message': f"Пользователь {recipient} не в сети."}, room=sender_sid)
        return

    if isinstance(message, str):
        threat_score = check_text_threat(message)
        emit('private_message', {
            'from': sender,
            'message': message
        }, room=recipient_sid)

        emit('private_message', {
            'from': sender,
            'message': message,
            'self': True
        }, room=sender_sid)

        if threat_score > 50:
            ai_response = generate_ai_response(message, sender, threat_score)
            emit('private_message', {
                'from': 'AI AGENT',
                'message': ai_response,
                'system': True
            }, room=recipient_sid)

@socketio.on('video_message')
def on_video_message(data):
    sender_sid = request.sid
    sender = users_name.get(sender_sid)
    recipient = data.get('to')
    video_info = {
        'from': sender,
        'video_url': data.get('video_url'),
        'is_fake': data.get('is_fake'),
        'percent_fake': data.get('percent_fake'),
        'verdict': data.get('verdict')
    }

    if not sender or not recipient:
        return

    recipient_sid = users_sid.get(recipient)
    if not recipient_sid:
        emit('system_message', {'message': f"Пользователь {recipient} не в сети."}, room=sender_sid)
        return

    emit('video_message', video_info, room=recipient_sid)

    if data.get('is_fake'):
        warning_msg = f"⚠️ Внимание! Полученное видео от '{sender}' может быть поддельным ({data.get('percent_fake'):.1f}%). Будьте осторожны."
        emit('private_message', {
            'from': 'AI AGENT',
            'message': warning_msg,
            'system': True
        }, room=recipient_sid)

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

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    socketio.run(app, debug=True)