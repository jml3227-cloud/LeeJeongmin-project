from flask import Blueprint, render_template, request, jsonify, current_app, session
import requests

bp = Blueprint('vlm', __name__, url_prefix='/vlm')

@bp.route('/')
def index():
    session['vlm_history'] = []
    return render_template('vlm.html', active_page='vlm')

@bp.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': '이미지가 없습니다'}), 400
    
    image = request.files['image']
    question = request.form.get('question', '이 조직 슬라이드 소견을 말해주세요.')

    session['vlm_history'] = []

    server_url = current_app.config['VLM_SERVER_URL']
    response = requests.post(
        f'{server_url}/vlm/analyze',
        files={'image': (image.filename, image.read(), image.content_type)},
        data={'question': question},
        timeout=120
    )

    data = response.json()
    if 'error' in data:
        return jsonify(data), 500
    
    session['vlm_history'] = [
        {'role': 'user', 'content': question},
        {'role': 'assistant', 'content': data.get('answer', '')}
    ]
    session.modified = True

    return jsonify(data)

@bp.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data.get('question', '')

    if not question:
        return jsonify({'error': '질문이 없습니다.'}), 400
    
    if 'vlm_history' not in session or not session['vlm_history']:
        return jsonify({'error': '먼저 이미지를 분석해주세요'}), 400
    
    server_url = current_app.config['VLM_SERVER_URL']
    response = requests.post(
        f'{server_url}/vlm/chat',
        json={
            'question': question,
            'history': session['vlm_history']
        },
        timeout=120
    )

    result = response.json()
    answer = result.get('answer', '')

    session['vlm_history'].append({'role': 'user', 'content': question})
    session['vlm_history'].append({'role': 'assistant', 'content': answer})
    session['vlm_history'] = session['vlm_history'][-10:]
    session.modified = True

    return jsonify({'answer': answer})