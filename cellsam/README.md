# CellSAM
 
세포/조직 인스턴스 분할 모델. [CellSAM 논문](https://github.com/vanvalenlab/cellSAM)의 아키텍처(Anchor DETR 기반 객체 탐지기 + SAM ViT-B)를 공식 데이터셋(`cellsam_v1.2`)으로 재현했다.
 
## 아키텍처
 
두 단계로 구성된다.
 
1. **CellFinder** (박스 탐지): Anchor DETR 구조에 SAM ViT-B의 image encoder를 backbone으로 붙인 탐지기. 이미지에서 세포 인스턴스의 bounding box를 예측한다. 학습은 Hungarian matching 기반 set prediction loss(classification + bbox L1 + GIoU)로 진행.
2. **Neck fine-tuning**: SAM image encoder의 neck(768→256 채널 변환 레이어)만 별도로 fine-tuning해서, ImageNet으로 사전학습된 SAM의 특징 분포와 세포 이미지 도메인 사이의 gap을 보정.
3. **추론 (SAM Mask Decoder)**: CellFinder가 예측한 box를 SAM의 prompt encoder + mask decoder에 넣어 인스턴스 마스크 생성. Box 단위로 순차 처리 후 인스턴스 ID를 합성.
```
입력 이미지 → CellFinder (박스 예측) → box threshold 필터링
                                              ↓
                        SAM Image Encoder (neck fine-tuned) → embedding
                                              ↓
                    box별 SAM Mask Decoder 순차 호출 → 인스턴스 마스크 합성
```
 
## 코드 구성
 
| 파일 | 역할 |
|---|---|
| `cellsam_inference.py` | `CellFinder`(탐지기 wrapper), `CellSAM`(탐지+분할 통합 추론 클래스) 정의. `predict()`가 전체 추론 파이프라인 |
| `cellsam.py` | CLI 엔트리포인트. `visualize`(단일 이미지 시각화) / `eval`(자체 F1 평가) 모드 |
| `train.py` | CellFinder 학습 스크립트 (박스 탐지, set prediction loss) |
| `train_neck.py` | SAM neck fine-tuning 스크립트 |
| `backbone.py` | SAM ViT-B image encoder를 Anchor DETR backbone 인터페이스에 맞게 래핑 |
| `anchor_detr.py`, `transformer.py`, `row_column_decoupled_attention.py`, `matcher.py` | Anchor DETR 아키텍처 구성 요소 (RCDA attention, Hungarian matcher 등) |
| `dataset.py`, `transform.py` | 데이터셋 로더 및 augmentation |
| `utils.py` | 후처리 유틸 (마스크 홀 채우기, 작은 인스턴스 제거 등) |
| `modelconfig.yaml` | 모델 하이퍼파라미터 설정 |
 
## 결과
 
CellFinder 50 epoch 학습 + neck 7 epoch fine-tuning 후, [vanvalenlab/cellSAM](https://github.com/vanvalenlab/cellSAM) 공식 레포의 `eval_main.py`로 평가한 결과 11개 데이터셋 그룹에서 F1 0.325(YeaZ) ~ 0.948(YeastNet).
 
> 참고: 이 폴더의 `cellsam.py` `eval` 모드는 개발 중 빠른 확인용으로 짠 자체 IoU≥0.5 매칭 F1 계산이고, 위 최종 수치는 저자 공식 eval 코드로 산출한 값이다.
 
## 실패 분석
 
숫자만으로는 재현이 잘 됐는지 판단할 수 없어서, 낮은 점수가 나온 케이스는 원인을 진단했다.
 
- **YeaZ (F1 0.325) — 인스턴스 병합**: 밀집된 효모 세포 이미지에서 인접 인스턴스를 하나로 병합해 검출하는 경향.
- **TissueNet — 고밀도 상황 대형 객체 검출 실패**: 세포 밀도가 높은 조직 이미지에서 큰 스케일의 인스턴스 검출이 무너짐.
- **NMS 부재로 인한 중복 검출**: 현재 추론 파이프라인(`predict()`)은 score threshold만 적용하고 Non-Maximum Suppression을 적용하지 않는다. 학습이 충분히 수렴하지 않은 경우 동일 인스턴스에 대해 여러 박스가 살아남아 중복 마스크가 생성됨.
## 실행
 
```bash
cd /workspace/LeeJeongmin-project/cellsam
 
# CellFinder 학습
PYTHONPATH=/workspace/LeeJeongmin-project python -m cellsam_models.train.train \
    --output_dir /workspace/LeeJeongmin-project/cellsam/outputs
 
# Neck fine-tuning
PYTHONPATH=/workspace/LeeJeongmin-project python -m cellsam_models.train.train_neck \
    --cellfinder_checkpoint /workspace/LeeJeongmin-project/cellsam/outputs/checkpoint_best.pth
 
# 추론 / 시각화
PYTHONPATH=/workspace/LeeJeongmin-project/cellsam python cellsam_models/cellsam.py \
    --image_path <이미지 경로> \
    --output_path <결과 저장 경로>
```
 
공식 F1 평가(`eval_main.py`)는 이 폴더 코드가 아니라 [vanvalenlab/cellSAM](https://github.com/vanvalenlab/cellSAM) 공식 레포를 별도로 설치해서 실행했다. `cellSAM/sam_inference.py`, `paper_evaluation/eval_main.py`의 `num_classes` 설정을 이 프로젝트의 체크포인트 클래스 수에 맞게 패치해야 하고, `fastremap`, `dask[distributed]`, `cellpose==3.0.11` 등 저자 레포 전용 의존성 설치가 필요하다.
