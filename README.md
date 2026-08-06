# CellLens

병리 영상 분석을 위한 End-to-End AI 파이프라인. 세포 분할, 병리 판독 LLM, H&E 이미지 VLM, 실험 영상 분석 VLM 네 개 모듈을 Flask 웹 서비스로 통합했다.

## 모듈 구성

| 모듈 | 내용 | 핵심 결과 |
|---|---|---|
| [CellSAM](./cellsam/README.md) | 세포/조직 인스턴스 분할 (Anchor DETR + SAM ViT-B) 논문 재현 | 11개 데이터셋 그룹 F1 0.325~0.948 (공식 eval_main.py 기준) |
| [LLM](./llm-finetuning/README.md) | 병리 도메인 QA (Llama 3.2 3B, CPT→SFT→QLoRA) | BERTScore F1 0.7160, ROUGE-1 0.1199 |
| [VLM (H&E)](./vlm/README.md) | H&E 병리 이미지 판독 (LLaVA-OneVision-Qwen2-7B, QLoRA) | BERTScore F1 0.9061, ROUGE-1 0.5774 |
| [FineBio VLM](./vlm/README.md) | 실험 절차 영상 분석 (작업명 + 진행률 분류) | 태스크 분류 정확도 80% |
| [cellsam-web](./cellsam-web/README.md) | 4개 모듈을 묶는 Flask 웹 서비스 | Blueprint 기반, RunPod 추론 서버와 REST API 연동 |

## 시스템 구조

```
사용자 → cellsam-web (Flask, Blueprint 패턴)
              ├─ cellsam_views.py ─┐
              ├─ llm_views.py     ─┼─ REST API → 단일 RunPod 추론 서버 (CellSAM + LLM + VLM/FineBio 겸용)
              └─ vlm_views.py     ─┘
```

- 백엔드: Flask + AJAX
- 학습 인프라: RunPod A100 80GB

## 기술적 의사결정 (요약)

- **VLM 학습 데이터 검증**: CellSAM이 산출하는 세포 정량 지표(세포 수, 면적 등)를 병리 진단 텍스트 생성에 그대로 활용할 계획이었으나, 통계 검증 결과 실제 진단과 유의미한 상관관계가 없었다(Cohen's d < 0.3). 해당 지표를 제거하고 VLM 학습 데이터셋을 재구성.
- **평가 파이프라인과 학습 데이터 정합성**: eval 코드의 SAM 정규화 방식(ImageNet 기준)에 맞춰 eval 코드를 고치는 대신 학습 데이터·파이프라인을 맞춤. eval 기준이 고정된 경우 학습 쪽을 맞추는 게 재현성 측면에서 맞다고 판단.
- **메모리 관리**: SAM mask decoder가 박스별로 순차 호출되며 backward() 전까지 전체 연산 그래프를 누적하는 구조라, 인스턴스 수에 비례해 메모리가 늘어남. `max_instances=400`으로 학습 시 랜덤 서브샘플, 검증 시 고정 truncation 적용해 OOM 방지.
- **FineBio VLM을 생성형이 아닌 분류로 구현**: 영상 226개, 클립 약 3,541개로 데이터가 희소한 상황에서 사전학습된 멀티모달 인코더를 활용한 분류가 데이터 효율 측면에서 더 defensible한 선택이라고 판단. 생성형 대화 인터페이스는 향후 과제로 남김.

각 모듈의 상세 아키텍처, 데이터셋, 실패 분석은 하위 폴더 README 참고.

## 향후 계획

- FineBio VLM 생성형(대화형) 출력으로 재학습
- LLM에 RAG 파이프라인 추가
- VLM 품질 개선을 위한 DPO 적용
