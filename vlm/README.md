# VLM (Vision-Language Model)

LLaVA-OneVision-Qwen2-7B를 QLoRA로 파인튜닝한 두 개의 서브모듈. 코드베이스는 공유하고 학습 데이터/타깃만 다르다.

- **VLM-1 (H&E)**: 위암·유방암 H&E 염색 조직검사 이미지 → 정상/위염/위암/유방암 판독 + 병리 소견 텍스트 생성
- **VLM-2 (FineBio)**: 세포 배양 실험실 절차 영상(다중 프레임) → 수행 중인 task 분류 + 완료율(%) 예측

두 서브모듈 모두 [lmms-finetune](https://github.com/zjysteven/lmms-finetune) 레퍼런스 구조를 참고해 직접 재구현했다. 원본은 하드코딩 기반 인자 관리였는데, 실험마다 코드를 고치는 게 번거로워서 인자(argument) 기반으로 바꿔 재현성과 실험 확장성을 확보했다 — LLM 파인튜닝 파이프라인에서 하드코딩으로 인해 겪었던 불편함을 여기서 개선한 것.

## 아키텍처

```
이미지/비디오 → Vision Tower(frozen) → Multi-modal Projector(full FT) → LLM(QLoRA)
```

- **Vision Tower**: freeze (학습 안 함)
- **Multi-modal Projector**: 4bit 양자화 대상에서 제외하고 float로 유지, full fine-tuning — 세포/조직 이미지가 일반 자연 이미지와 feature 분포가 크게 달라서 projection도 도메인에 맞게 다시 학습해야 한다고 판단
- **LLM**: 4bit(NF4) 양자화 + LoRA — 모든 Linear 레이어(`lm_head` 제외) 대상

## 코드 구성

| 파일 | 역할 |
|---|---|
| `arguments.py` | `ModelArguments`, `DataArguments`, `TrainingArguments`, `QLoraArguments` dataclass 정의. `MODULE_KEYWORDS`로 vision_encoder/projector/llm 모듈 구분 |
| `train.py` | 학습 진입점. 4bit 양자화 로드 → vision encoder·`image_newline` freeze → LLM Linear 레이어에 LoRA 적용 → projector는 `llm_int8_skip_modules`로 양자화 제외 + `modules_to_save`로 full fine-tuning |
| `datasets.py` | `LazySupervisedDataset`. 이미지는 PIL, 비디오는 PyAV(`read_video_pyav`)로 균등 간격 프레임 샘플링 후 `__getitem__` 시점에 로드 |
| `collator.py` | `LLaVAOnevisionDataCollator`. 채팅 템플릿 적용 → `<image>`/`<video>` 토큰을 실제 패치/프레임 토큰 수만큼 `repeat_interleave`로 확장 → `assistant_masks`로 답변 토큰만 loss 계산 → truncate/pad |
| `utils.py` | `find_all_linear_names`(LoRA 대상 Linear 레이어 탐색, `lm_head` 제외), `save_model`, `rank0_print` |
| `eval.py` | VLM-1(H&E) 평가. 생성 응답에 대해 BERTScore, ROUGE 계산 |
| `finebio_inference.py` | VLM-2 단일 비디오 추론 CLI |
| `finebio_eval.py` | VLM-2 val set 전체 평가. `finebio_inference.py`를 서브프로세스로 호출해 정규식으로 task/완료율 파싱 후 정확도·MAE 집계 |

## 결과

| | VLM-1 (H&E) | VLM-2 (FineBio) |
|---|---|---|
| 지표 | BERTScore F1 0.9061 / ROUGE-1 0.5774 | Task 분류 정확도 82.6% |

## VLM-1 하이퍼파라미터 선택 이유

- `model_max_length=4096`: LLaVA-OneVision은 이미지 하나당 최대 수천 개의 이미지 토큰을 생성한다. 기본값(1024)으로는 `collator.py`의 truncation 로직이 이미지 토큰만으로 시퀀스를 다 채우고 답변 텍스트(assistant 토큰)를 잘라버려서 학습 loss가 0이 되는 문제가 있었음.
- `lora_r=64`: 세포/조직 이미지가 일반 자연 이미지와 feature 분포 차이가 커서, 낮은 rank로는 도메인 표현력이 부족하다고 판단.
- `gradient_accumulation_steps=4`: 단일 GPU 환경에서 effective batch size를 유지하면서 OOM 방지.

## VLM-2 (FineBio) 실패 분석

Output collapse 발생(모든 입력에 "pipetting"만 출력). 처음엔 프레임 수(4개) 부족으로 오진해 8개로 늘렸지만 재발했고, 실제 원인은 projector가 4bit 양자화 대상에서 제외되지 않아 그레디언트가 흐르지 않았던 것. `llm_int8_skip_modules`로 projector를 양자화에서 제외해 해결 (최종 설정: `num_frames=8`, `lora_r=8`, `model_max_length=4096`).

## 실행

```bash
cd /workspace/LeeJeongmin-project/vlm

# VLM-1 (H&E) 학습
python train.py \
    --model_local_path /workspace/llava-onevision-qwen2-7b-ov-hf \
    --data_path <H&E QA json 경로> \
    --output_dir outputs/he \
    --lora_r 64 \
    --model_max_length 4096 \
    --gradient_accumulation_steps 4 \
    --bf16 True

# VLM-2 (FineBio) 학습 — 최종 설정 기준
python train.py \
    --model_local_path /workspace/llava-onevision-qwen2-7b-ov-hf \
    --data_path data/finebio_train.json \
    --eval_data_path data/finebio_valid.json \
    --video_folder data/vlm_videos \
    --num_frames 8 \
    --lora_r 8 \
    --model_max_length 4096 \
    --output_dir outputs/finebio \
    --gradient_accumulation_steps 16 \
    --bf16 True

# 평가
python eval.py
python finebio_eval.py
```
