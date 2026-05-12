import os

BASE_DIR = os.path.dirname(__file__)

SECRET_KEY = 'dev'

# Runpod 서버 URL
CELLSAM_SERVER_URL = '' 
LLM_SERVER_URL = ''

# 업로드 설정
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'app/static/uploads')
MAX_CONTENT_LENGTH = 16 * 1024 * 1024