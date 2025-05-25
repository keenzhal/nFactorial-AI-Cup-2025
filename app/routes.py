from flask import render_template, request, redirect, url_for, jsonify, send_from_directory, current_app
from werkzeug.utils import secure_filename
import os

from .utils import allowed_file, get_upload_folder # Используем относительный импорт
from .ai_core import analyze_video # Используем относительный импорт
from . import socketio # Импортируем socketio из __init__.py

# Предполагается, что users_sid и users_name будут управляться в socket_handlers или __init__
# Для простоты пока оставим их здесь, но лучше вынести
users_sid = {}
users_name = {}

def register_routes(app):
    @app.route('/', methods=['GET', 'POST'])
    def index():
        if request.method == 'POST':
            username = request.form.get('username')
            # Проверку users_sid лучше делать через обработчики сокетов или общий стейт
            if username: # Упрощенная проверка для примера
                return redirect(url_for('chat_page', username=username)) # Изменено имя маршрута
            else:
                return render_template('login.html', error="Имя не указано")
        return render_template('index.html')

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            username = request.form.get('username')
            if username: # Упрощенная проверка
                return redirect(url_for('chat_page', username=username))
            else:
                return render_template('login.html', error="Имя не указано")
        return render_template('login.html')

    @app.route('/chat') # Это будет основной маршрут для страницы чата
    def chat_page(): # Переименовано для ясности
        username = request.args.get('username')
        if not username:
            return redirect(url_for('index'))
        return render_template('chat.html', username=username)

    @app.route('/upload', methods=['POST'])
    def upload_file_route(): # Переименовано во избежание конфликтов
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file and allowed_file(file.filename):
            upload_folder_path = get_upload_folder()
            filename = secure_filename(file.filename)
            filepath = os.path.join(upload_folder_path, filename)
            
            try:
                file.save(filepath)
            except Exception as e:
                current_app.logger.error(f"Error saving file: {e}")
                return jsonify({'error': 'Could not save file'}), 500

            percent_fake = analyze_video(filepath)
            verdict = "⚠️ ВНИМАНИЕ: Видео возможно поддельное!" if percent_fake > 50 else "✅ Видео скорее настоящее."
            # Важно: url_for здесь должен ссылаться на маршрут, который отдает файлы из uploads
            video_url = url_for('uploaded_file_route', filename=filename) 

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

    @app.route('/uploads/<filename>') # Этот маршрут для отдачи файлов
    def uploaded_file_route(filename): # Переименовано
        return send_from_directory(get_upload_folder(), filename) 