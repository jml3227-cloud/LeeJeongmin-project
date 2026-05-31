import os

BASE_DIR = os.path.dirname(__file__)

SECRET_KEY = 'dev'

# Runpod 서버 URL
CELLSAM_SERVER_URL = os.environ.get('CELLSAM_SERVER_URL', 'http://216.81.245.125:17582') 
LLM_SERVER_URL = os.environ.get('LLM_SERVER_URL', 'http://216.81.245.125:17582')
VLM_SERVER_UTL = os.environ.get('VLM_SERVER_URL',  'http://216.81.245.125:17582')

# 업로드 설정
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'app/static/uploads')
MAX_CONTENT_LENGTH = 16 * 1024 * 1024