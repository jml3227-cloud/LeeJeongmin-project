from flask import Blueprint, render_template, request, jsonify, current_app
import requests

bp = Blueprint('cellsam', __name__, url_prefix='/cellsam')

@bp.route('/')
def index():
    return render_template('cellsam.html', active_page='cellsam')

@bp.route('/analyze', methods=['POST'])
def analyze():
    if 'image'not in request.files:
        return jsonify({'error': '이미지가 없습니다'}), 400
    
    image = request.files['image']

    # Runpod CellSAM 서버로 전달
    cellsam_url = current_app.config['CELLSAM_SERVER_URL']
    response = requests.post(
        f'{cellsam_url}/predict',
        files={'image': (image.filename, image.read(), image.content_type)}
    )

    return jsonify(response.json())