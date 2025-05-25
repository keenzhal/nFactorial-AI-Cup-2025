from flask import Flask
from flask_socketio import SocketIO
import os
import logging

# Инициализация SocketIO здесь, чтобы его можно было импортировать в другие модули
socketio = SocketIO()

def create_app(debug=False):
    """Создает и конфигурирует экземпляр приложения Flask."""
    app = Flask(__name__, 
                static_folder='static',      # Указываем, что папка static внутри app/
                template_folder='templates'  # Указываем, что папка templates внутри app/
               )
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your_very_secret_key_for_production_!@#$%^')
    app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'uploads') # Путь к uploads внутри app/
    # ALLOWED_EXTENSIONS лучше держать в utils.py или здесь, если они специфичны для app

    # Настройка логирования
    if not app.debug or os.environ.get("FLASK_ENV") == "production":
        # В продакшене логируем в stdout, чтобы Docker/PaaS могли их собирать
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        app.logger.addHandler(stream_handler)
    else:
        # В режиме отладки можно использовать стандартный логгер Flask
        app.logger.setLevel(logging.DEBUG)
    
    app.logger.info('Flask App created')

    # Убедимся, что папка для загрузок существует
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.logger.info(f"Upload folder is {app.config['UPLOAD_FOLDER']}")

    # Инициализация SocketIO с приложением
    socketio.init_app(app, cors_allowed_origins="*") # Добавлен cors_allowed_origins
    app.logger.info('SocketIO initialized with app')

    # Регистрация маршрутов
    with app.app_context(): # Убедимся, что находимся в контексте приложения
        from . import routes
        routes.register_routes(app)
        app.logger.info('Routes registered')

        # Регистрация обработчиков SocketIO
        from . import socket_handlers
        socket_handlers.register_socket_handlers() # Убедитесь, что эта функция существует и вызывается корректно
        app.logger.info('Socket handlers registered')
        
        # Важно: ai_core импортируется для загрузки моделей при старте
        from . import ai_core 
        app.logger.info('AI core loaded, models should be initializing.')

    return app 