from flask import Blueprint, request, jsonify, render_template, current_app
import requests

bp = Blueprint('llm', __name__, url_prefix='/llm')

@bp.route('/')
def index():
    return render_template('llm.html', active_page='llm')

@bp.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')

    try:
        server_url = current_app.config['LLM_SERVER_URL']
        res = requests.post(f'{server_url}/llm/generate', json={'question': user_message}, timeout=60)
        reply = res.json().get('answer', '응답을 받지 못했습니다.')
    except Exception as e:
        reply = f'서버 연결 오류: {str(e)}'

    return jsonify({'reply': reply})