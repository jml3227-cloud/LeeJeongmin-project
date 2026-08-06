from flask import Blueprint, render_template, request, jsonify, current_app
import requests

bp = Blueprint('finebio', __name__, url_prefix='/finebio')

@bp.route('/')
def index():
    return render_template('finebio.html', active_page='finebio')

@bp.route('/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files:
        return jsonify({'error': '비디오가 없습니다'}), 400
    
    video = request.files['video']
    if not video.filename.lower().endswith('.mp4'):
        return jsonify({'error': '지원하지 않는 형식입니다. MP4만 지원합니다.'}), 400
    
    server_url = current_app.config['FINEBIO_SERVER_URL']
    response = requests.post(
        f'{server_url}/finebio/analyze',
        files={'video': (video.filename, video.read(), video.content_type)},
        timeout=120
    )

    data = response.json()
    if 'error' in data:
        return jsonify(data), 500
    
    return jsonify(data)