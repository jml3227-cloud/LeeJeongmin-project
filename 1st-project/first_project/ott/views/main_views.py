from flask import Blueprint, url_for, render_template, current_app
from ott.models import PredictionModel

bp = Blueprint('main', __name__, url_prefix='/')

@bp.route('/')
def index():
    model = PredictionModel(
        user=current_app.config['USER'],
        password=current_app.config['PASSWORD'],
        dsn=current_app.config['DSN']
    )
    result = model.get_group_counts()

    stats = {}
    stats['low'] = result[0]
    stats['mid'] = result[1]
    stats['high'] = result[2]

    total = stats['low'] + stats['mid'] + stats['high']
    stats['low_pct'] = round((stats['low'] / total) * 100, 1)
    stats['high_pct'] = round((stats['high'] / total) * 100, 1)

    return render_template('index.html', stats=stats, active_page='main')
