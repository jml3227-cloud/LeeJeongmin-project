from flask import Blueprint, request, jsonify, render_template
import requests

bp = Blueprint('llm_views', __name__)

LLM_SERVER_URL = "http://localhost:5001/generate"  # LLM 서버 주소 (나중에 변경)

@bp.route('/llm')
def llm_page():
    return render_template('llm.html')

@bp.route('/llm/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')

    try:
        res = requests.post(LLM_SERVER_URL, json={'message': user_message}, timeout=60)
        reply = res.json().get('reply', '응답을 받지 못했습니다.')
    except Exception as e:
        reply = f'서버 연결 오류: {str(e)}'

    return jsonify({'reply': reply})