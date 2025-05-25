from flask_socketio import emit, disconnect
from flask import request, current_app

from . import socketio # Импортируем socketio из __init__.py
from .ai_core import check_text_threat, generate_ai_response # Используем относительный импорт

# Эти словари должны быть доступны глобально для этого модуля или передаваться/управляться иначе
# В идеале, это состояние должно управляться более надежно (например, Redis для многопроцессности)
users_sid = {}
users_name = {}

def register_socket_handlers():

    def emit_user_list():
        for sid_target in users_name.keys():
            username_target = users_name[sid_target]
            other_users = [u for u, s_id in users_sid.items() if u != username_target]
            try:
                socketio.emit('user_list', {'users': other_users}, room=sid_target)
            except Exception as e:
                current_app.logger.error(f"Error emitting user list to {username_target}: {e}")

    @socketio.on('join')
    def on_join(data):
        username = data.get('username')
        sid = request.sid
        current_app.logger.info(f"User {username} trying to join with SID {sid}")

        if not username:
            current_app.logger.warning(f"Join attempt with no username, SID {sid}. Disconnecting.")
            disconnect()
            return

        old_sid = users_sid.get(username)
        if old_sid and old_sid != sid:
            current_app.logger.info(f"User {username} reconnected with new SID {sid}. Old SID {old_sid} will be disconnected.")
            try:
                socketio.emit('system_message', {
                    'message': 'Ваша сессия была прервана из-за повторного входа с другим устройством.'
                }, room=old_sid)
                disconnect(old_sid) # Отключаем старую сессию
            except Exception as e:
                current_app.logger.error(f"Error handling old session for {username}: {e}")

        users_sid[username] = sid
        users_name[sid] = username
        current_app.logger.info(f"User {username} joined with SID {sid}. Current users: {users_sid}")
        emit_user_list()

    @socketio.on('disconnect')
    def on_disconnect():
        sid = request.sid
        username = users_name.pop(sid, None) # Удаляем по sid и получаем имя пользователя
        if username:
            users_sid.pop(username, None) # Также удаляем из users_sid
            current_app.logger.info(f"User {username} (SID: {sid}) disconnected. Current users: {users_sid}")
            emit_user_list()
        else:
            current_app.logger.info(f"SID {sid} disconnected, but was not in users_name list.")

    @socketio.on('private_message')
    def on_private_message(data):
        sender_sid = request.sid
        sender = users_name.get(sender_sid)
        recipient_username = data.get('to')
        message = data.get('message', '').strip()

        if not sender or not recipient_username or not message:
            current_app.logger.warning(f"Private message attempt with missing data: sender={sender}, to={recipient_username}, msg_empty={not message}")
            return

        recipient_sid = users_sid.get(recipient_username)
        if not recipient_sid:
            try:
                socketio.emit('system_message', {'message': f"Пользователь {recipient_username} не в сети."}, room=sender_sid)
            except Exception as e:
                current_app.logger.error(f"Error emitting 'user not online' to {sender}: {e}")
            return

        if isinstance(message, str):
            threat_score = check_text_threat(message)
            try:
                socketio.emit('private_message', {
                    'from': sender,
                    'message': message
                }, room=recipient_sid)

                socketio.emit('private_message', {
                    'from': sender,
                    'message': message,
                    'self': True
                }, room=sender_sid)
            except Exception as e:
                current_app.logger.error(f"Error sending private message from {sender} to {recipient_username}: {e}")

            if threat_score > 50:
                ai_response = generate_ai_response(message, sender, threat_score)
                try:
                    socketio.emit('private_message', {
                        'from': 'AI AGENT',
                        'message': ai_response,
                        'system': True
                    }, room=recipient_sid)
                except Exception as e:
                    current_app.logger.error(f"Error sending AI response to {recipient_username}: {e}")

    @socketio.on('video_message')
    def on_video_message(data):
        sender_sid = request.sid
        sender = users_name.get(sender_sid)
        recipient_username = data.get('to')
        
        video_info = {
            'from': sender,
            'video_url': data.get('video_url'),
            'is_fake': data.get('is_fake'),
            'percent_fake': data.get('percent_fake'),
            'verdict': data.get('verdict')
        }

        if not sender or not recipient_username or not video_info.get('video_url'):
            current_app.logger.warning(f"Video message with missing data: sender={sender}, to={recipient_username}, url_empty={not video_info.get('video_url')}")
            return

        recipient_sid = users_sid.get(recipient_username)
        if not recipient_sid:
            try:
                socketio.emit('system_message', {'message': f"Пользователь {recipient_username} не в сети."}, room=sender_sid)
            except Exception as e:
                current_app.logger.error(f"Error emitting 'user not online' for video to {sender}: {e}")
            return
        try:
            socketio.emit('video_message', video_info, room=recipient_sid)
        except Exception as e:
            current_app.logger.error(f"Error sending video message from {sender} to {recipient_username}: {e}")

        if data.get('is_fake'):
            warning_msg = f"⚠️ Внимание! Полученное видео от '{sender}' может быть поддельным ({data.get('percent_fake'):.1f}%). Будьте осторожны."
            try:
                socketio.emit('private_message', {
                    'from': 'AI AGENT',
                    'message': warning_msg,
                    'system': True
                }, room=recipient_sid)
            except Exception as e:
                current_app.logger.error(f"Error sending AI video warning to {recipient_username}: {e}")

    current_app.logger.info("Socket.IO handlers registered.") 