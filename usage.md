# Pipeline Usage (Shell Endpoints)

파이프라인별 실행 엔드포인트만 빠르게 모아둔 문서입니다.

# Pre filtering
```bash
bash scripts/filter_pre.sh
```
* txt 파일에 pass/fail 리스트 저장됨

# Retargeting
## 1. SMPL-X -> Robot 리타게팅
```bash
bash scripts/retarget_smplx.sh
```
* 개별 모션 리타게팅
* 디렉토리 단위 리타게팅
* AMASS, OMOMO, kimodo 결과 호환

## 2. BVH -> Robot 리타게팅
```bash
bash scripts/retarget_bvh.sh
```
* lafan dataset 전용

## 3. GVHMR -> Robot 리타게팅
```bash
bash scripts/retarget_gvhmr.sh
```
* video to mesh 돌린 후 .pt 파일 지정해야함

## 4. Teleop 스트림 -> Robot 리타게팅
```bash
bash scripts/retarget_teleop.sh
```

## 5. LAFAN1 BVH -> SMPL-X(npz) 변환
```bash
bash scripts/lafan_to_smplx.sh
```

## 6. LAFAN1 스트림 -> Robot (motion matching)
```bash
bash scripts/mm_stream.sh
```
* 모션 매칭 스트림 결과 저장

# Packing
```python
python script/pack_retargeted_motions.py --<dir>
```
* 디렉토리 내 모든 single motion pkl 들을 하나의 pkl 로 패킹


# PICO Tele-op
* Xrobo toolkit PC 실행
* PICO 헤드셋, 트래커 올바르게 착용
* 헤드셋에서 캘리브레이션
* 헤드셋에서 xrobo third party app 실행 및 연결
* GMR teleop 스크립트 실행