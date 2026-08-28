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


# PICO Tele-op Retargeting Setup
1. Xrobo toolkit PC 실행
2. PICO 헤드셋, 트래커 올바르게 착용
3. 헤드셋에서 캘리브레이션
4. 헤드셋에서 xrobo third party app 실행 및 연결
5. GMR teleop 스크립트 실행

# PICO Tele-op
1. PICO setup
2. GMR teleop 스크립트 실행
```python
python script/pack_retargeted_motions.py --<dir>
```
3. ros topic publish 스크립트 실행
```python
PYTHON_BIN=/usr/bin/python3 bash scripts/run_ros2_redis_bridge.sh
```

## On-board execution
* run xrobot toolkit bia terminal
```
nohup bash /opt/apps/roboticsservice/runService.sh >/dev/null 2>&1 &
```
* run `retarget_teleop.sh` with `--headless` option
* run web debug
```python scripts/pico_web_debug.py --ros-domain-id <ROS_DOMAIN_ID> --port <optional>```



# Motion View & Curation
* single motion, directory, single packed motion 선택해서 view 할 수 있음

```bash
bash scripts/view_motion.sh
```

* `vis_robot_motion_npz.py` 는 curation 기능 있음