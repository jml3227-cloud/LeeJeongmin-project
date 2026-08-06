# cellsam-web

CellSAM, LLM, VLM(H&E), FineBio VLM 네 모듈을 하나로 묶는 Flask 웹 서비스. Blueprint 패턴으로 라우트를 분리하고, 실제 추론은 RunPod의 단일 추론 서버(`cellsam_server/app.py`)에 REST API로 위임한다.

## 구조

```
cellsam-web/
    app/
        static/
        templates/
        views/
            main_views.py      # 메인 페이지
            cellsam_views.py   # CellSAM 분석
            llm_views.py       # LLM 챗봇
            vlm_views.py       # VLM(H&E) 분석 + 대화
            finebio_views.py   # FineBio VLM 영상 분석
        __init__.py            # Blueprint 등록
    cellsam_server/
        app.py                 # RunPod 추론 서버 (5개 라우트 그룹 전부 서빙)
        inference.py
```

## Blueprint 구성

| Blueprint | URL prefix | 라우트 | 비고 |
|---|---|---|---|
| `main` | `/` | `/` | 인덱스 페이지 |
| `cellsam` | `/cellsam` | `/`, `/analyze`, `/analyze_video` | 단일 이미지 분석 + 다중 이미지→MP4 트래킹 영상 생성 |
| `llm` | `/llm` | `/`, `/chat` | 세션 기반 대화, 최근 3턴(6개 메시지)까지 유지 |
| `vlm` | `/vlm` | `/`, `/analyze`, `/chat` | 이미지는 첫 턴(`/analyze`)에만 입력, 이후 `/chat`은 텍스트 히스토리만 주고받음. 최근 5턴(10개 메시지) 유지 |
| `finebio` | `/finebio` | `/`, `/analyze` | 영상 업로드 → 단발성 분석(대화형 아님) |

모든 라우트 등록은 `__init__.py`의 `create_app()`에서 이뤄진다:

```python
from .views import main_views, cellsam_views, llm_views, vlm_views, finebio_views
app.register_blueprint(main_views.bp)
app.register_blueprint(cellsam_views.bp)
app.register_blueprint(llm_views.bp)
app.register_blueprint(vlm_views.bp)
app.register_blueprint(finebio_views.bp)
```

## 추론 서버 연동

Flask 쪽 각 Blueprint는 `current_app.config`에 등록된 서버 URL로 요청을 위임하는 얇은 프록시 역할만 한다. 실제 모델 로딩과 추론은 `cellsam_server/app.py` 하나가 전부 담당한다.

- `CellSAM`(CellFinder+SAM), `LLM`(QLoRA 병합 모델)은 항상 메모리에 로드된 상태로 서빙.
- `VLM`과 `FineBio`는 같은 베이스 모델(LLaVA-OneVision-Qwen2-7B)을 공유하고, PEFT 어댑터만 `set_adapter("vlm")` / `set_adapter("finebio")`로 교체해서 쓴다 — 7B 모델을 두 번 로드하지 않기 위한 메모리 절약 방식.
- FineBio는 `load_adapter()`(LoRA)만으로는 완성되지 않는다. Projector는 4bit 양자화 대상에서 제외하고 별도 float32 파일(`projector_fp32.bin`)로 저장했기 때문에, adapter 로드 후 `load_state_dict()`로 한 번 더 얹어줘야 한다.

## 세션 기반 대화 히스토리

- `llm_views.py`: `session['history']`에 user/assistant 메시지를 계속 append하다가 `[-6:]`로 슬라이싱해 최근 3턴만 유지, 매 요청마다 이 히스토리를 추론 서버에 함께 전달.
- `vlm_views.py`: `/analyze`(이미지 업로드) 시점에 `session['vlm_history']`를 초기화하고, 이후 `/chat`에서 텍스트만 주고받으며 `[-10:]`로 최근 5턴 유지.
- `finebio_views.py`: 세션 상태 없음 — 영상 하나 업로드하면 그 결과만 반환하는 stateless 방식.

## 실행

```bash
cd /workspace/LeeJeongmin-project/cellsam-web
python -m flask run --host 0.0.0.0

# 추론 서버 (RunPod)
cd /workspace/LeeJeongmin-project/cellsam-web/cellsam_server
PYTHONPATH=/workspace/LeeJeongmin-project/cellsam python app.py
```
