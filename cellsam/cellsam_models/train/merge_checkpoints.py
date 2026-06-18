"""
CellFinder(checkpoint_best.pth)와 Neck(neck_checkpoint_best.pth) 두 체크포인트를
저자 official cellSAM.sam_inference.CellSAM 클래스가 기대하는 단일 state_dict로 합친다.

CellSAM 구조 (sam_inference.py 기준):
  - self.model            : sam_model_registry["vit_b"]() 전체
                            (image_encoder, prompt_encoder, mask_decoder)
  - self.cellfinder        : CellfinderAnchorDetr
      - self.cellfinder.decode_head : build_inference_model()이 만든 모듈
                                      (transformer.* + backbone.body.* 413개 키,
                                       checkpoint_best.pth의 'model' state_dict와
                                       키 이름이 정확히 동일하다는 것을 확인함)
  - self.model_cp          : self.model의 deepcopy (실제 forward에 쓰이는 쪽)
                            -> load_state_dict가 model_cp 키가 없으면 자동으로
                               model의 값을 복사해주므로 우리가 따로 채울 필요 없음

매핑:
  1) cellfinder.decode_head.*  <- checkpoint_best.pth['model']  (그대로, prefix 변경 없음)
  2) model.image_encoder.*     <- checkpoint_best.pth['model']의 'backbone.body.*'
                                   (prefix를 'backbone.body.' -> 'image_encoder.' 로 변경)
  3) model.image_encoder.neck.* <- neck_checkpoint_best.pth['neck_state_dict']
                                   (prefix를 '' -> 'image_encoder.neck.' 로 변경)
  4) model.prompt_encoder.*, model.mask_decoder.* <- 원본 SAM vit_b 사전학습 값 그대로
     (CellFinder/Neck 둘 다 이 부분은 학습하지 않았으므로)
"""

import torch
from segment_anything import sam_model_registry


CELLFINDER_CKPT = '/workspace/LeeJeongmin-project/cellsam/outputs/checkpoint_best.pth'
NECK_CKPT = '/workspace/LeeJeongmin-project/cellsam/outputs/neck_checkpoint_best.pth'
SAM_CHECKPOINT = '/workspace/sam_vit_b_01ec64.pth'
OUTPUT_PATH = '/workspace/LeeJeongmin-project/cellsam/outputs/cellsam_merged.pth'


def main():
    print('1) 원본 SAM ViT-B 로드 (prompt_encoder, mask_decoder 베이스용)...')
    sam = sam_model_registry['vit_b'](checkpoint=SAM_CHECKPOINT)
    sam_state = sam.state_dict()
    # sam_state의 키는 'image_encoder.*', 'prompt_encoder.*', 'mask_decoder.*' 형태

    print('2) CellFinder 체크포인트 로드...')
    cellfinder_ckpt = torch.load(CELLFINDER_CKPT, map_location='cpu')
    cellfinder_state = cellfinder_ckpt['model']
    print(f'   CellFinder state_dict 키 개수: {len(cellfinder_state)}')

    print('3) Neck 체크포인트 로드...')
    neck_ckpt = torch.load(NECK_CKPT, map_location='cpu')
    neck_state = neck_ckpt['neck_state_dict']
    print(f'   Neck state_dict 키 개수: {len(neck_state)}')

    merged = {}

    # ----- (1) cellfinder.decode_head.* : CellFinder 체크포인트 그대로 -----
    for k, v in cellfinder_state.items():
        merged[f'cellfinder.decode_head.{k}'] = v

    # ----- (2) model.image_encoder.* : 원본 SAM 값으로 베이스를 깔고 -----
    for k, v in sam_state.items():
        merged[f'model.{k}'] = v

    # ----- (2 계속) backbone.body.* 를 image_encoder.* 로 덮어쓰기 -----
    overwritten = 0
    for k, v in cellfinder_state.items():
        if k.startswith('backbone.body.'):
            new_key = 'model.image_encoder.' + k[len('backbone.body.'):]
            if new_key not in merged:
                print(f'   경고: {new_key} 가 SAM 베이스에 없음 (키 이름 불일치 가능성)')
            merged[new_key] = v
            overwritten += 1
    print(f'   image_encoder 중 CellFinder 학습값으로 덮어쓴 키: {overwritten}개')

    # ----- (3) neck 덮어쓰기 -----
    neck_overwritten = 0
    for k, v in neck_state.items():
        new_key = 'model.image_encoder.neck.' + k
        if new_key not in merged:
            print(f'   경고: {new_key} 가 SAM 베이스에 없음 (키 이름 불일치 가능성)')
        merged[new_key] = v
        neck_overwritten += 1
    print(f'   neck 덮어쓴 키: {neck_overwritten}개')

    print(f'\n최종 merged state_dict 키 개수: {len(merged)}')
    torch.save(merged, OUTPUT_PATH)
    print(f'저장 완료: {OUTPUT_PATH}')


if __name__ == '__main__':
    main()