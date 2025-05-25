from app import create_app, socketio # Импортируем create_app и socketio из пакета app
import os

app = create_app(debug=True) # Включаем debug режим для разработки

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    # Используйте socketio.run для корректной работы с Flask-SocketIO
    # host='0.0.0.0' чтобы приложение было доступно извне (например, в Docker)
    socketio.run(app, port=port, allow_unsafe_werkzeug=True if app.debug else False) 