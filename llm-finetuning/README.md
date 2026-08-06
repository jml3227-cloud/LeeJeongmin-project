# LLM Fine-tuning

Llama 3.2 3B(base)를 병리/세포생물학 도메인 QA에 맞게 파인튜닝. **서로 독립적인 두 파이프라인**이 있고, 최종 배포 모델은 QLoRA 파이프라인 쪽이다.

## 두 파이프라인

풀 파인튜닝과 QLoRA 두 방법론을 실제로 비교해보기 위해 둘 다 구현했다.

- **CPT → SFT**: 풀 파인튜닝(양자화 없음) 경로. 도메인 지식을 CPT로 먼저 주입하고 SFT로 QA 형식을 학습.
- **QLoRA (phase1 → phase2)**: base 모델에서 바로 시작하는 4bit 양자화 + LoRA 경로. 어댑터를 두 번에 나눠 학습한다 — phase1은 일반 instruction-following 능력(Alpaca), phase2는 도메인 QA 지식. 각 phase 끝에 어댑터를 merge해서 다음 phase의 시작점으로 넘긴다.

두 파이프라인은 서로의 결과물을 이어받지 않는 독립 실험이다. 학습 시간이 짧고 성능도 더 나은 QLoRA 쪽을 최종 배포 모델로 채택했다 (`outputs/qlora_final`). 아래 결과 지표도 이 모델 기준이다.

```
[CPT+SFT 경로]
Llama-3.2-3B base ─(CPT, full FT, 1 epoch)→ outputs/cpt
                  ─(SFT, full FT, 2 epoch, 도메인QA 500+Alpaca 200)→ outputs/sft

[QLoRA 경로 — 최종 채택]
Llama-3.2-3B base ─(QLoRA phase1: 4bit+LoRA r=16, Alpaca 200)→ merge → outputs/qlora_merged
                  ─(QLoRA phase2: 4bit+LoRA r=64, 도메인QA 2975+Alpaca 1275)→ merge → outputs/qlora_final ★
```

## 코드 구성

| 파일 | 역할 |
|---|---|
| `cpt/cpt_train.py` | CPT. PMC 논문+위키피디아 청크 텍스트로 causal LM 풀 파인튜닝 (1 epoch) |
| `sft/sft_train.py` | SFT. `outputs/cpt`에서 이어받아 도메인 QA 500개 + Alpaca EN 200개로 풀 파인튜닝. `DataCollatorForCompletionOnlyLM`으로 답변 부분만 loss 계산 |
| `qlora/qlora_phase1.py` | QLoRA phase1. base 모델에서 바로 시작, 4bit 양자화 + LoRA(r=16)로 Alpaca 200개 학습 후 merge |
| `qlora/qlora_phase2.py` | QLoRA phase2. phase1 merge 모델에서 이어서 LoRA(r=64)로 도메인 QA 2975개 + Alpaca 1275개(7:3 비율) 학습 후 merge → 최종 모델 |
| `qlora/prepare_data.py` | PubMedQA(`pqa_artificial`)에서 `cell` + `tumor/pathology/histology/morphology/cancer` 키워드로 영문 3,000개 필터링·추출 |
| `chunk.py` | CPT용 원문(PMC, 위키) 토큰 기준 1024개 단위 청킹 (overlap 없음) |
| `eval.py` | 최종 모델(`qlora_final`) 응답 생성 + BERTScore, ROUGE 계산. 생성 결과를 2문장으로 truncate하고 URL/영문 잔여물 제거하는 후처리 포함 |
| `perplexity.py` | 최종 모델 held-out 평가셋 Perplexity 계산 |
| `inference_test.py` | `--mode {base,sft,qlora_phase1,qlora}`로 단계별 모델의 응답을 같은 질문으로 비교하는 정성 평가 스크립트 |

**참고**: 영문 PubMedQA 3,000개 → 한국어 도메인 QA 2,975개로의 번역은 Google Translate API로 로컬에서 별도 처리했고, 해당 스크립트는 레포에 포함되어 있지 않다.

## 결과 (QLoRA 최종 모델, `outputs/qlora_final`)

| 지표 | 값 |
|---|---|
| BERTScore F1 | 0.7160 |
| ROUGE-1 | 0.1199 |
| Perplexity | 5.0523 |

## 실행

```bash
cd /workspace/LeeJeongmin-project/llm-finetuning

# CPT
python cpt/cpt_train.py

# SFT (CPT 완료 후)
python sft/sft_train.py

# QLoRA phase1 (base 모델에서 바로 시작)
python qlora/qlora_phase1.py

# QLoRA phase2 (phase1 완료 후)
python qlora/qlora_phase2.py

# 평가
python eval.py
python perplexity.py

# 단계별 정성 비교
python inference_test.py --mode qlora
```
