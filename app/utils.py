import os

ALLOWED_EXTENSIONS = {'mp4'}
UPLOAD_FOLDER = 'app/uploads' # Обновленный путь

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_upload_folder():
    # Убедимся, что папка существует
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    return UPLOAD_FOLDER 