# OTT 구독자 이탈 예측 서비스

2024 방송매체 이용행태조사 데이터를 기반으로 OTT 구독자의 이탈 위험을 예측하는 Flask 웹 서비스입니다.

## 주요 기능

- **S-02 기존 사용자 조회**: DB에 등록된 사용자 번호로 이탈 위험도 조회
- **S-03 이탈 위험 예측**: 사용자 정보를 입력하면 OTT 이용 빈도 그룹(고빈도/중빈도/저빈도)을 분류하고 이탈 위험도를 예측
- SHAP 기반 예측 원인 분석 제공

## 기술 스택

- **Backend**: Flask, oracledb, scikit-learn, SHAP, pandas
- **Frontend**: Bootstrap 4.6.2, jQuery, Chart.js
- **DB**: Oracle XE
- **모델**: Random Forest Classifier

## 실행 방법

### 1. 패키지 설치

```
pip install -r requirements.txt
```

### 2. 환경변수 설정

`.env.example`을 복사해 `.env` 파일 생성 후 값 입력:

```
SECRET_KEY=your_secret_key
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_DSN=localhost:1521/xe
```

### 3. Flask 실행

`ottproject.cmd`를 실행하면 가상환경 활성화 및 Flask 서버가 자동으로 시작됩니다.

- 실행 경로: `C:\Flask_projects\first_project`
- 호스트: `0.0.0.0`, 포트: `5000`

직접 실행할 경우:
```
set FLASK_APP=ott
flask run --host=0.0.0.0 --port=5000
```

## 데이터 출처

2024 방송매체 이용행태조사 (방송통신위원회)

## 버전 히스토리
- v1.0 (2026.03.20): 최초 기획 - 광고형 요금제 수용도 예측
- v2.0 (2026.03.25): 기획 변경 - 이용빈도 기반 이탈 위험군 분류로 모델 변경

