from flask_wtf import FlaskForm
from wtforms.fields import SelectField, FloatField, SelectMultipleField, IntegerField
from wtforms.validators import DataRequired, NumberRange


class ChurnForm(FlaskForm):
    family_type = SelectField('가족구성', coerce=int,choices=[
        (1, '1인 가구'), (2, '1세대 가구'), (3, '2세대 가구'),
        (4, '3세대 가구'), (5, '기타')
    ], validators=[DataRequired()])

    monthly_fee_code = SelectField('월 구독료', coerce=int,choices=[
        (1, '3,000원 미만'), (2, '3,000~5,000원 미만'), (3, '5,000~9,000원 미만'),
        (4, '9,000~12,000원 미만'), (5, '12,000~15,000원 미만'),
        (6, '15,000~20,000원 미만'), (7, '20,000원 이상')
    ], validators=[DataRequired()])

    total_min = FloatField('일주일 총 이용시간(분)', validators=[
        DataRequired(), NumberRange(min=0)
    ])

    recommend_view = SelectField('추천 시청 정도', coerce=int, choices=[
        (1, '전혀 그렇지 않다'), (2, '그렇지 않다'), (3, '보통이다'),
        (4, '그렇다'), (5, '매우 그렇다')
    ], validators=[DataRequired()])

    search_view = SelectField('검색 시청 정도', coerce=int, choices=[
        (1, '전혀 그렇지 않다'), (2, '그렇지 않다'), (3, '보통이다'),
        (4, '그렇다'), (5, '매우 그렇다')
    ], validators=[DataRequired()])

    binge_watch = SelectField('몰아보기 정도', coerce=int, choices=[
        (1, '전혀 그렇지 않다'), (2, '그렇지 않다'), (3, '보통이다'),
        (4, '그렇다'), (5, '매우 그렇다')
    ], validators=[DataRequired()])

    ott_services = SelectMultipleField('사용하는 OTT 종류', coerce=int, choices=[
        (1, '넷플릭스'), (2, '웨이브'), (3, '티빙'),
        (4, '쿠팡플레이'), (5, '디즈니플러스'), (6, '기타')
    ], validators=[DataRequired('필수 입력 항목입니다.')])

    devices = SelectMultipleField('이용하는 기기', coerce=int, choices=[
        (1, 'TV'), (2, '데스크탑 PC'), (3, '노트북'), (4, '스마트폰'), (5, '태블릿 PC')
    ], validators=[DataRequired('필수 입력 항목입니다.')])

    content_types = SelectMultipleField('이용하는 컨텐츠', coerce=int, choices=[
        (1, '지상파 프로그램'), (2, '유료방송 프로그램'), (3, 'OTT 자체 제작'),
        (4, '영화'), (5, '숏폼'), (6, '기타')
    ], validators=[DataRequired('필수 입력 항목입니다.')])

class LookupForm(FlaskForm):
    user_seq = IntegerField('User SEQ', validators=[DataRequired('필수 입력 항목입니다.'), NumberRange(min=1)])