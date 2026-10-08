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
SDK 원본 헤드셋, 양쪽 컨트롤러, 모션 트래커 pose를 MuJoCo viewer에서 확인하려면:
```bash
bash scripts/retarget_teleop.sh --show_raw_pico_devices true
```
장치 마커 전체를 이동하려면 `--raw_pico_devices_offset DX DY DZ`를 추가합니다.
예: `bash scripts/retarget_teleop.sh --raw_pico_devices_offset 0 1.2 0` (미터, `retarget_teleop.sh` 설정값은 `0 -1.2 0`).
헤드셋은 분홍색, 왼쪽 컨트롤러는 파란색, 오른쪽 컨트롤러는 초록색,
트래커는 노란색 계열의 구체로 표시됩니다. RGB 화살표는 pose 회전에 따라 움직이는 각 장치의 로컬 X/Y/Z 축입니다.
SDK pose는 몸 관절과 동일한 축으로 변환하고, 몸통 위치를 기준으로 로봇 viewer의
스케일·바닥 높이에 맞춰 표시합니다. 몸 관절은 viewer 배치 기준에만 사용하며 장치 pose를 대체하지 않습니다.
SDK 장치 pose가 없거나 쿼터니언이 유효하지 않으면 해당 마커를 표시하지 않습니다.
표시 이름은 `Headset`, `L Hand`, `R Hand`입니다.

원본 장치 pose를 CSV로 저장하려면:
```bash
bash scripts/retarget_teleop.sh --save_pkl_enabled true --save_raw_xrobot_data true --save_raw_device_poses true --save_pkl_dir pico/debug/
```
`scripts/retarget_teleop.sh` 상단의 `save_raw_device_poses`와 `save_pkl_dir`로도 설정할 수 있습니다.
클립마다 같은 번호의 세 파일이 `save_pkl_dir`에 저장됩니다: `pico_0000.pkl` (retargeted motion),
`pico_0000_raw.pkl` (raw XRobot), `pico_0000_raw_devices.csv` (raw device pose).
CSV는 motion pkl과 같은 녹화 토글, 재개 후 프레임 제외, 클립 경계를 따르며 데이터 행 수가 motion 프레임 수와 같습니다.
`save_pkl_every_n_steps`에 도달하거나 녹화를 끌 때 세 파일을 함께 저장합니다. `--save_pkl_whole_session`이면 종료 시 한 클립으로 저장합니다.
`save_pkl_enabled=true`여야 raw device CSV도 저장됩니다.
CSV는 `timestamp_ns` 다음에 `headset`, `l_hand`, `r_hand` 순서로 각 장치의 `x,y,z,qx,qy,qz,qw`를 저장합니다. 발목과 허리 트래커는 수집하지 않습니다.
값은 SDK 트래킹 원점 기준 pose 그대로이며 viewer의 축 변환, 스케일, 바닥 정렬, offset은 적용하지 않습니다.
SDK가 장치 pose 배열을 제공하면 7개 값을 그대로 저장합니다. SDK가 0 pose를 보내면 0이 기록되며 viewer에서는 유효하지 않은 회전으로 취급해 표시하지 않습니다.
pose 배열 자체가 없으면 해당 7개 값은 `NaN`입니다. 몸 관절이나 로봇 관절 값으로 채우지 않습니다.
이전에 저장한 6장치 CSV도 viewer에서 읽을 수 있지만 헤드셋과 양손만 표시합니다. 이전 CSV에서는 0 쿼터니언이 `NaN`으로 기록됐을 수 있습니다.

저장된 PICO 클립 세 파일을 동기화해 보려면:
```bash
python scripts/vis_robot_motion_directory.py --robot robros_igris_c_v2 --robot_motion_dir pico/debug/ --pico_view --root_quat_scalar_first true
```
`--pico_view`에서는 같은 클립 번호의 motion pkl, `*_raw.pkl`, `*_raw_devices.csv`를 프레임 인덱스로 맞춥니다.
이 모드에서는 `--raw_xrobot_dir`을 지정하지 않아도 `--robot_motion_dir` 아래에서 companion 파일을 찾습니다.
로봇, raw XRobot 스켈레톤, raw device 마커는 옆으로 약 1m씩 떨어져 표시됩니다.
`--pico_view`는 CSV에서 유효한 장치 pose만 표시하며 `NaN`과 0 쿼터니언은 그리지 않습니다. 몸 관절은 마커 그룹의 배치 기준에만 사용합니다.
raw 파일의 프레임 수가 motion과 다르면 해당 raw 표시를 건너뜁니다. 옵션을 끄면 기존 viewer 동작을 유지합니다.

`--save_pkl_enabled false`이면 retargeted motion pkl 저장이 꺼지고, `--save_raw_xrobot_data`도 작동하지 않습니다.
여기서 `--save_raw_xrobot_data`는 전체 XRobot 입력 프레임 저장 옵션이며 SMPL-X 파라미터 저장 옵션은 아닙니다.
`--save_raw_device_poses`도 `--save_pkl_enabled false`이면 작동하지 않습니다. 두 raw 저장 옵션을 모두 끄면 raw 저장용 추가 장치 조회와 파일 쓰기는 없지만,
`--show_raw_pico_devices true`인 경우 viewer 표시를 위한 SDK 조회와 렌더링 비용은 계속 발생합니다.
고정된 기본 pose나 몸 관절 대체값은 넣지 않습니다.

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
