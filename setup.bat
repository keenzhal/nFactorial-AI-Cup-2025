@echo off
REM Создание виртуального окружения
py -m venv venv

REM Активация виртуального окружения (для Windows)
call venv\Scripts\activate.bat

REM Обновление pip
python -m pip install --upgrade pip

REM Установка библиотек
pip install flask flask-socketio numpy torch opencv-python Pillow transformers scikit-learn werkzeug python-engineio python-socketio

echo Установка завершена.
pause
