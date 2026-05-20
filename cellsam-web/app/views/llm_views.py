from flask import Blueprint, request, jsonify, render_template, current_app, session
import requests

bp = Blueprint('llm', __name__, url_prefix='/llm')

@bp.route('/')
def index():
    session['history'] = []
    return render_template('llm.html', active_page='llm')

@bp.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')

    if 'history' not in session:
        session['history'] = []

    try:
        server_url = current_app.config['LLM_SERVER_URL']
        res = requests.post(
            f'{server_url}/llm/generate', 
            json={
                'question': user_message,
                'history': session['history']
            }, 
                timeout=60
        )
        reply = res.json().get('answer', '응답을 받지 못했습니다.')

        session['history'].append({'role': 'user', 'content': user_message})
        session['history'].append({'role': 'assistant', 'content': reply})
        session['history'] = session['history'][-6:]
        session.modified = True

    except Exception as e:
        reply = f'서버 연결 오류: {str(e)}'

    return jsonify({'reply': reply})