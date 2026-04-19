from flask import Blueprint, url_for, current_app, render_template, request, jsonify
from ott.models import PredictionModel
from werkzeug.utils import redirect
from ott.forms import ChurnForm, LookupForm
import numpy as np
import pandas as pd

bp = Blueprint('churn', __name__, url_prefix='/churn')

@bp.route('/', methods=['GET'])
def index():
    form = ChurnForm()

    return render_template('freq_classification/S03_input.html', form=form, active_page='churn')

@bp.route('/predict', methods=['POST'])
def predict():
    form = ChurnForm()

    if request.method == 'POST' and form.validate_on_submit():

        ott_count = len(form.ott_services.data)
        device_count = len(form.devices.data)
        content_diversity = len(form.content_types.data)

        total_min = form.total_min.data
        search_view = form.search_view.data
        recommend_view = form.recommend_view.data
        family_type = form.family_type.data
        monthly_fee_map = {1:1500, 2: 4000, 3:7000, 4:10500, 5:13500, 6:17500, 7:20000}
        monthly_fee = monthly_fee_map[form.monthly_fee_code.data]
        binge_watch = form.binge_watch.data
        used_last_week = 1 if total_min > 0 else 0
        watch_original = 1 if 3 in form.content_types.data else 0
        watch_movie = 1 if 4 in form.content_types.data else 0
        watch_shortform = 1 if 5 in form.content_types.data else 0

        total_min_safe = total_min if total_min > 0 else np.nan
        explore_idx = ((content_diversity / total_min_safe) * search_view) if total_min > 0 else 0
        explore_idx = min(explore_idx, 0.12480769230769248)

        cherry_pick_idx = (ott_count / total_min_safe) if total_min > 0 else 0
        cherry_pick_idx = min(cherry_pick_idx, 0.018030303030303167)

        feature_cols = [
            'RECOMMEND_VIEW', 'TOTAL_MIN', 'EXPLORE_IDX',
            'CHERRY_PICK_IDX', 'MONTHLY_FEE', 'FAMILY_TYPE',
            'BINGE_WATCH', 'OTT_COUNT', 'DEVICE_COUNT', 'USED_LAST_WEEK',
            'WATCH_ORIGINAL', 'WATCH_MOVIE', 'WATCH_SHORTFORM'
        ]

        features = pd.DataFrame([[
            recommend_view, total_min, explore_idx,
            cherry_pick_idx, monthly_fee, family_type,
            binge_watch, ott_count, device_count, used_last_week,
            watch_original, watch_movie, watch_shortform
        ]], columns=feature_cols)

        prediction = current_app.config['MODEL'].predict(features)

        freq_group = int(prediction[0])
        insight_map = {
            2: '안정적인 구독자입니다.',
            1: '이탈 위험 관찰 대상입니다.',
            0: '이탈 위험군입니다.'
        }

        group_label = {
            2: '고빈도(안정 구독자)',
            1: '중빈도(관찰 필요)',
            0: '저빈도(이탈 위험)'
        }



        explainer = current_app.config['EXPLAINER']
        shap_vals = explainer.shap_values(features)

        exclude_features = ['FAMILY_TYPE', 'USED_LAST_WEEK', 'EXPLORE_IDX', 'CHERRY_PICK_IDX']
        shap_importance = [(i, abs(shap_vals[0, i, freq_group])) for i, col in enumerate(feature_cols) if
                           col not in exclude_features]
        shap_importance.sort(key=lambda x: x[1], reverse=True)
        top_features = [i for i, _ in shap_importance[:3]]

        reasons = []
        for i in top_features:
            name = feature_cols[i]
            shap_val = shap_vals[0, i, freq_group]
            pos = shap_val > 0

            if name == 'TOTAL_MIN':
                if pos and total_min >= 540:
                    reasons.append('주간 시청 시간이 많습니다')
                elif pos and total_min < 540:
                    reasons.append('시청 시간이 이용 패턴에 영향을 줬습니다')
                elif not pos and total_min >= 540:
                    reasons.append('시청 시간이 많음에도 이탈 위험이 존재합니다')
                else:
                    reasons.append('주간 시청 시간이 적습니다')

            elif name == 'BINGE_WATCH':
                if pos and binge_watch >= 3:
                    reasons.append('몰아보기를 즐겨 이용에 적극적입니다')
                elif pos and binge_watch < 3:
                    reasons.append('몰아보기 성향이 이용 패턴에 영향을 줬습니다')
                elif not pos and binge_watch >= 3:
                    reasons.append('몰아보기를 즐기나 이탈 위험이 존재합니다')
                else:
                    reasons.append('몰아보기를 즐기지 않습니다')

            elif name == 'OTT_COUNT':
                if pos and ott_count >= 2:
                    reasons.append('여러 OTT를 구독해 콘텐츠 의존도가 높습니다')
                elif pos and ott_count < 2:
                    reasons.append('OTT 구독 패턴이 이용 수준에 영향을 줬습니다')
                elif not pos and ott_count >= 2:
                    reasons.append('여러 OTT를 구독 중이나 이탈 위험이 존재합니다')
                else:
                    reasons.append('단일 OTT만 구독 중입니다')

            elif name == 'DEVICE_COUNT':
                if pos and device_count >= 2:
                    reasons.append('다양한 기기로 시청해 이용 환경이 유연합니다')
                elif pos and device_count < 2:
                    reasons.append('기기 이용 패턴이 이용 수준에 영향을 줬습니다')
                elif not pos and device_count >= 2:
                    reasons.append('다양한 기기를 이용하나 이탈 위험이 존재합니다')
                else:
                    reasons.append('단일 기기로만 시청합니다')

            elif name == 'RECOMMEND_VIEW':
                if pos and recommend_view >= 4:
                    reasons.append('추천 콘텐츠를 적극적으로 시청합니다')
                elif pos and recommend_view < 4:
                    reasons.append('추천 콘텐츠 시청이 이용 패턴에 영향을 줬습니다')
                elif not pos and recommend_view >= 4:
                    reasons.append('추천 콘텐츠를 즐기나 이탈 위험이 존재합니다')
                else:
                    reasons.append('추천 콘텐츠 시청이 적습니다')

            elif name == 'MONTHLY_FEE':
                if pos and monthly_fee >= 13500:
                    reasons.append('높은 요금제를 이용해 서비스 몰입도가 높습니다')
                elif pos and monthly_fee < 13500:
                    reasons.append('현재 요금제가 이용 패턴에 영향을 줬습니다')
                elif not pos and monthly_fee >= 13500:
                    reasons.append('높은 요금제 부담이 이탈 위험 요인입니다')
                else:
                    reasons.append('낮은 요금제로 이용 중입니다')

            elif name == 'WATCH_ORIGINAL':
                if pos and watch_original == 1:
                    reasons.append('OTT 오리지널 콘텐츠를 시청합니다')
                elif not pos and watch_original == 0:
                    reasons.append('OTT 오리지널 콘텐츠를 시청하지 않습니다')
                else:
                    reasons.append('콘텐츠 시청 패턴이 이용 수준에 영향을 줬습니다')

            elif name == 'WATCH_MOVIE':
                if pos and watch_movie == 1:
                    reasons.append('영화 콘텐츠를 즐겨 시청합니다')
                elif not pos and watch_movie == 0:
                    reasons.append('영화 콘텐츠를 시청하지 않습니다')
                else:
                    reasons.append('영화 시청 패턴이 이용 수준에 영향을 줬습니다')

            elif name == 'WATCH_SHORTFORM':
                if pos and watch_shortform == 1:
                    reasons.append('숏폼 콘텐츠를 시청합니다')
                elif not pos and watch_shortform == 0:
                    reasons.append('숏폼 콘텐츠를 시청하지 않습니다')
                else:
                    reasons.append('숏폼 시청 패턴이 이용 수준에 영향을 줬습니다')





        return jsonify({
            'freq_group': freq_group,
            'insight': insight_map[freq_group],
            'group_label': group_label[freq_group],
            'reasons': reasons
        })

    errors = {field: errs for field, errs in form.errors.items()}
    return jsonify({'validation_errors': errors}), 400

@bp.route('/chart-data')
def chart_data():
    model = PredictionModel(
        user=current_app.config['USER'],
        password=current_app.config['PASSWORD'],
        dsn=current_app.config['DSN']
    )
    result = model.get_group_counts()

    return jsonify({
        'low': result[0],
        'mid': result[1],
        'high': result[2]
    })


@bp.route('/lookup', methods=['GET', 'POST'])
def lookup():
    db = PredictionModel(
        user=current_app.config['USER'],
        password=current_app.config['PASSWORD'],
        dsn=current_app.config['DSN']
    )
    max_seq = db.get_max_user_seq()

    form = LookupForm()
    result = None
    error = None
    user_seq_input = request.form.get('user_seq', '')

    if request.method == 'POST' and form.validate_on_submit():
        user_seq = form.user_seq.data
        row = db.get_user_data(user_seq)

        if row is None:
            error = '존재하지 않는 사용자 번호입니다.'
        else:
            monthly_fee_map = {1: 1500, 2: 4000, 3: 7000, 4: 10500, 5: 13500, 6: 17500, 7: 20000}
            monthly_fee = monthly_fee_map.get(row['MONTHLY_FEE_CODE'], 0)
            total_min = row['AVG_MIN_WEEKDAY'] * 5 + row['AVG_MIN_WEEKEND'] * 2
            total_min_safe = total_min if total_min > 0 else np.nan
            explore_idx = ((row['CONTENT_DIVERSITY'] / total_min_safe) * row['SEARCH_VIEW']) if total_min > 0 else 0
            explore_idx = min(explore_idx, 0.12480769230769248)
            cherry_pick_idx = (row['OTT_COUNT'] / total_min_safe) if total_min > 0 else 0
            cherry_pick_idx = min(cherry_pick_idx, 0.018030303030303167)

            feature_cols = [
                'RECOMMEND_VIEW', 'TOTAL_MIN', 'EXPLORE_IDX',
                'CHERRY_PICK_IDX', 'MONTHLY_FEE', 'FAMILY_TYPE',
                'BINGE_WATCH', 'OTT_COUNT', 'DEVICE_COUNT', 'USED_LAST_WEEK',
                'WATCH_ORIGINAL', 'WATCH_MOVIE', 'WATCH_SHORTFORM'
            ]

            features = pd.DataFrame([[
                row['RECOMMEND_VIEW'], total_min, explore_idx,
                cherry_pick_idx, monthly_fee, row['FAMILY_TYPE'],
                row['BINGE_WATCH'], row['OTT_COUNT'], row['DEVICE_COUNT'], row['USED_LAST_WEEK'],
                row['WATCH_ORIGINAL'], row['WATCH_MOVIE'], row['WATCH_SHORTFORM']
            ]], columns=feature_cols)

            prediction = current_app.config['MODEL'].predict(features)
            freq_group = int(prediction[0])

            explainer = current_app.config['EXPLAINER']
            shap_vals = explainer.shap_values(features)

            exclude_features = ['FAMILY_TYPE', 'USED_LAST_WEEK', 'EXPLORE_IDX', 'CHERRY_PICK_IDX']
            shap_importance = [(i, abs(shap_vals[0, i, freq_group])) for i, col in enumerate(feature_cols) if
                               col not in exclude_features]
            shap_importance.sort(key=lambda x: x[1], reverse=True)
            top_features = [i for i, _ in shap_importance[:3]]

            reasons = []
            for i in top_features:
                name = feature_cols[i]
                shap_val = shap_vals[0, i, freq_group]
                pos = shap_val > 0

                if name == 'TOTAL_MIN':
                    if pos and total_min >= 540:
                        reasons.append('주간 시청 시간이 많습니다')
                    elif pos and total_min < 540:
                        reasons.append('시청 시간이 이용 패턴에 영향을 줬습니다')
                    elif not pos and total_min >= 540:
                        reasons.append('시청 시간이 많음에도 이탈 위험이 존재합니다')
                    else:
                        reasons.append('주간 시청 시간이 적습니다')
                elif name == 'BINGE_WATCH':
                    if pos and row['BINGE_WATCH'] >= 3:
                        reasons.append('몰아보기를 즐겨 이용에 적극적입니다')
                    elif pos and row['BINGE_WATCH'] < 3:
                        reasons.append('몰아보기 성향이 이용 패턴에 영향을 줬습니다')
                    elif not pos and row['BINGE_WATCH'] >= 3:
                        reasons.append('몰아보기를 즐기나 이탈 위험이 존재합니다')
                    else:
                        reasons.append('몰아보기를 즐기지 않습니다')
                elif name == 'OTT_COUNT':
                    if pos and row['OTT_COUNT'] >= 2:
                        reasons.append('여러 OTT를 구독해 콘텐츠 의존도가 높습니다')
                    elif pos and row['OTT_COUNT'] < 2:
                        reasons.append('OTT 구독 패턴이 이용 수준에 영향을 줬습니다')
                    elif not pos and row['OTT_COUNT'] >= 2:
                        reasons.append('여러 OTT를 구독 중이나 이탈 위험이 존재합니다')
                    else:
                        reasons.append('단일 OTT만 구독 중입니다')
                elif name == 'DEVICE_COUNT':
                    if pos and row['DEVICE_COUNT'] >= 2:
                        reasons.append('다양한 기기로 시청해 이용 환경이 유연합니다')
                    elif pos and row['DEVICE_COUNT'] < 2:
                        reasons.append('기기 이용 패턴이 이용 수준에 영향을 줬습니다')
                    elif not pos and row['DEVICE_COUNT'] >= 2:
                        reasons.append('다양한 기기를 이용하나 이탈 위험이 존재합니다')
                    else:
                        reasons.append('단일 기기로만 시청합니다')
                elif name == 'RECOMMEND_VIEW':
                    if pos and row['RECOMMEND_VIEW'] >= 4:
                        reasons.append('추천 콘텐츠를 적극적으로 시청합니다')
                    elif pos and row['RECOMMEND_VIEW'] < 4:
                        reasons.append('추천 콘텐츠 시청이 이용 패턴에 영향을 줬습니다')
                    elif not pos and row['RECOMMEND_VIEW'] >= 4:
                        reasons.append('추천 콘텐츠를 즐기나 이탈 위험이 존재합니다')
                    else:
                        reasons.append('추천 콘텐츠 시청이 적습니다')
                elif name == 'MONTHLY_FEE':
                    if pos and monthly_fee >= 13500:
                        reasons.append('높은 요금제를 이용해 서비스 몰입도가 높습니다')
                    elif pos and monthly_fee < 13500:
                        reasons.append('현재 요금제가 이용 패턴에 영향을 줬습니다')
                    elif not pos and monthly_fee >= 13500:
                        reasons.append('높은 요금제 부담이 이탈 위험 요인입니다')
                    else:
                        reasons.append('낮은 요금제로 이용 중입니다')
                elif name == 'WATCH_ORIGINAL':
                    if pos and row['WATCH_ORIGINAL'] == 1:
                        reasons.append('OTT 오리지널 콘텐츠를 시청합니다')
                    elif not pos and row['WATCH_ORIGINAL'] == 0:
                        reasons.append('OTT 오리지널 콘텐츠를 시청하지 않습니다')
                    else:
                        reasons.append('콘텐츠 시청 패턴이 이용 수준에 영향을 줬습니다')
                elif name == 'WATCH_MOVIE':
                    if pos and row['WATCH_MOVIE'] == 1:
                        reasons.append('영화 콘텐츠를 즐겨 시청합니다')
                    elif not pos and row['WATCH_MOVIE'] == 0:
                        reasons.append('영화 콘텐츠를 시청하지 않습니다')
                    else:
                        reasons.append('영화 시청 패턴이 이용 수준에 영향을 줬습니다')
                elif name == 'WATCH_SHORTFORM':
                    if pos and row['WATCH_SHORTFORM'] == 1:
                        reasons.append('숏폼 콘텐츠를 시청합니다')
                    elif not pos and row['WATCH_SHORTFORM'] == 0:
                        reasons.append('숏폼 콘텐츠를 시청하지 않습니다')
                    else:
                        reasons.append('숏폼 시청 패턴이 이용 수준에 영향을 줬습니다')

            insight_map = {
                2: '안정적인 구독자입니다.',
                1: '이탈 위험 관찰 대상입니다.',
                0: '이탈 위험군입니다.'
            }
            group_label = {
                2: '고빈도(안정 구독자)',
                1: '중빈도(관찰 필요)',
                0: '저빈도(이탈 위험)'
            }

            result = {
                'user_seq': user_seq,
                'freq_group': freq_group,
                'insight': insight_map[freq_group],
                'group_label': group_label[freq_group],
                'reasons': reasons
            }



    return render_template('lookup/S02_lookup.html',
                       form=form, result=result, error=error,
                       user_seq_input=user_seq_input, max_seq=max_seq,
                       active_page='lookup')
