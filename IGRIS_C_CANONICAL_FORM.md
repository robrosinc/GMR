# IGRIS-C v2 motion canonical form (GMR PKL)

파일은 Python `pickle`로 직렬화한 `dict`이며, NumPy 배열을 포함한다. 로봇 모델은 [`assets/robros_igris_c_v2/igris_c_v2.xml`](assets/robros_igris_c_v2/igris_c_v2.xml), 로봇 식별자는 `robros_igris_c_v2`이다.

## 공통 규칙

- `T` = 프레임 수, `K` = `len(link_body_list)`. 전체 키바디를 포함하면 `K=33`이다. `fps`는 모션마다 다른 양의 값이며 시간 간격은 `dt=1/fps`이다. 별도 timestamp 배열은 없다. MuJoCo XML의 `timestep=0.005`는 저장 모션의 fps가 아니다.
- canonical 수치 배열은 NumPy `float32` (`fp32`), `fps`는 Python `float`로 저장한다. 기존 GMR 파일은 `float64`일 수도 있으므로 읽을 때 dtype을 확인하고 변환 결과를 `float32`로 맞춘다.
- MuJoCo 좌표계: 월드 `+Z`가 위, 위치 단위는 m, 회전 관절은 rad, 선속도는 m/s, 각속도는 rad/s. 쿼터니언 배열의 마지막 차원 순서는 **`[w, x, y, z]`**이다. SciPy `Rotation`에 전달할 때는 `[x, y, z, w]`로 재배열한다.
- 자유 베이스는 `base_link`이고, `qpos[t] = concat(root_pos[t], root_rot[t], dof_pos[t])`로 재구성한다. shape은 `(T, 38)`이며 `qpos[:3]`는 월드 위치, `qpos[3:7]`은 월드 기준 베이스 방향, `qpos[7:]`은 아래 31개 관절각이다. `root_rot`은 첫 번째 키바디인 `Link_Waist_Yaw`의 회전과 같지 않을 수 있다.

## 최상위 key-value

| key | 타입 / shape | 의미 |
| --- | --- | --- |
| `fps` | `float` | 프레임/초 (`> 0`). |
| `root_pos` | `float32 (T, 3)` | `base_link` 원점의 월드 XYZ 위치. |
| `root_rot` | `float32 (T, 4)` | `base_link`의 월드 방향, `wxyz`. |
| `root_vel` | `float32 (T, 3)` | 월드 XYZ 선속도. |
| `root_angvel` | `float32 (T, 3)` | 베이스 상대 회전에서 계산한 XYZ 각속도. 아래 계산식 참조. |
| `dof_pos` | `float32 (T, 31)` | MuJoCo `qpos[7:]`의 관절각, 아래 순서. |
| `dof_vel` | `float32 (T, 31)` | 같은 순서의 관절 각속도. |
| `keybody_pos_world` | `float32 (T, K, 3)` | `link_body_list` 순서의 MuJoCo body 원점 월드 위치 (`data.xpos`). |
| `keybody_rot_world` | `float32 (T, K, 4)` | 같은 body의 월드 방향 (`data.xquat`), `wxyz`. |
| `keybody_pos_local` | `float32 (T, K, 3)` | 같은 위치를 `base_link` 좌표계로 변환. |
| `keybody_rot_local` | `float32 (T, K, 4)` | 같은 방향을 `base_link` 좌표계로 변환, `wxyz`. **각 링크의 부모 관절 기준 회전이 아니다.** |
| `keybody_pos` | `float32 (T, K, 3)` | `keybody_pos_world`와 값이 같은 레거시 별칭. |
| `link_body_list` | `list[str]`, 길이 `K` | `keybody_*`의 두 번째 축에 대응하는 body 이름. |
| `local_body_link_body_list` | `None` | 이 형식에서는 사용하지 않음. |
| `local_body_pos` | `None` | 이 형식에서는 사용하지 않음. |
| `retarget_meta` | `dict` (선택) | 생성 경로별 추가 메타데이터. 고정 키는 없으며 포즈 재구성에 필요하지 않음. |

키바디를 생략한 기존 파일은 `K=0`, `link_body_list=[]`, `keybody_*`의 두 번째 차원이 0일 수 있다. 전체 키바디가 필요한 변환 결과에는 아래 33개 이름을 순서대로 사용한다.

## 관절 및 키바디 순서

`dof_pos`/`dof_vel` 열 인덱스 (`0`부터 시작):

```text
 0 Joint_Waist_Yaw           1 Joint_Waist_Roll          2 Joint_Waist_Pitch
 3 Joint_Hip_Pitch_Left      4 Joint_Hip_Roll_Left       5 Joint_Hip_Yaw_Left
 6 Joint_Knee_Pitch_Left     7 Joint_Ankle_Pitch_Left    8 Joint_Ankle_Roll_Left
 9 Joint_Hip_Pitch_Right    10 Joint_Hip_Roll_Right     11 Joint_Hip_Yaw_Right
12 Joint_Knee_Pitch_Right   13 Joint_Ankle_Pitch_Right  14 Joint_Ankle_Roll_Right
15 Joint_Shoulder_Pitch_Left 16 Joint_Shoulder_Roll_Left 17 Joint_Shoulder_Yaw_Left
18 Joint_Elbow_Pitch_Left   19 Joint_Wrist_Yaw_Left     20 Joint_Wrist_Roll_Left
21 Joint_Wrist_Pitch_Left   22 Joint_Shoulder_Pitch_Right 23 Joint_Shoulder_Roll_Right
24 Joint_Shoulder_Yaw_Right 25 Joint_Elbow_Pitch_Right  26 Joint_Wrist_Yaw_Right
27 Joint_Wrist_Roll_Right   28 Joint_Wrist_Pitch_Right  29 Joint_Neck_Yaw
30 Joint_Neck_Pitch
```

`link_body_list`의 인덱스 (`keybody_*[:, i]`):

```text
 0 Link_Waist_Yaw           1 Link_Waist_Roll          2 Link_Waist_Pitch
 3 Link_Hip_Pitch_Left      4 Link_Hip_Roll_Left       5 Link_Hip_Yaw_Left
 6 Link_Knee_Pitch_Left     7 Link_Ankle_Pitch_Left    8 Link_Ankle_Roll_Left
 9 Link_Hip_Pitch_Right    10 Link_Hip_Roll_Right     11 Link_Hip_Yaw_Right
12 Link_Knee_Pitch_Right   13 Link_Ankle_Pitch_Right  14 Link_Ankle_Roll_Right
15 Link_Shoulder_Pitch_Left 16 Link_Shoulder_Roll_Left 17 Link_Shoulder_Yaw_Left
18 Link_Elbow_Pitch_Left   19 Link_Wrist_Yaw_Left     20 Link_Wrist_Roll_Left
21 Link_Wrist_Pitch_Left   22 Left_Hand               23 Link_Shoulder_Pitch_Right
24 Link_Shoulder_Roll_Right 25 Link_Shoulder_Yaw_Right 26 Link_Elbow_Pitch_Right
27 Link_Wrist_Yaw_Right    28 Link_Wrist_Roll_Right   29 Link_Wrist_Pitch_Right
30 Right_Hand              31 Link_Neck_Yaw           32 Link_Neck_Pitch
```

손가락 관절은 `dof_pos`에 없다. `Left_Hand`/`Right_Hand`는 관절 열이 아니라 키바디 이름이다.

## 파생값 계산 및 변환

`R(q)`를 `wxyz` 쿼터니언이 나타내는 회전, `p=root_pos[t]`, `q=root_rot[t]`라고 하면 다음과 같다. `keybody_*_local`은 **베이스 좌표계** 값이다.

```text
keybody_pos_local[t, i] = R(q)^(-1) · (keybody_pos_world[t, i] - p)
keybody_rot_local[t, i] = q^(-1) ⊗ keybody_rot_world[t, i]
keybody_pos[t, i]       = keybody_pos_world[t, i]
```

`root_vel`과 `dof_vel`은 첫/마지막 프레임에서 1차 차분, 내부에서 중심 차분을 쓴다: `v[0]=(x[1]-x[0])/dt`, `v[t]=(x[t+1]-x[t-1])/(2dt)`, `v[T-1]=(x[T-1]-x[T-2])/dt`. `root_angvel[t]`은 내부에서 `log(R(q[t-1])^-1 R(q[t+1]))/(2dt)`, 양 끝에서 각각 `log(R(q[0])^-1 R(q[1]))/dt`와 `log(R(q[T-2])^-1 R(q[T-1]))/dt`이다 (`log`는 SciPy `Rotation.as_rotvec`). 상대 회전의 기준은 첫 번째 쿼터니언이므로 이 배열을 월드 축 각속도로 취급하지 않는다. `T=1`이면 속도는 0이다.

다른 형식에서 변환할 때는 관절 이름을 위 31열로 **이름 기반 재배열**한 뒤, 위치·방향·관절각으로 `qpos`를 만들고 `fps`에 맞춰 속도를 계산한다. 키바디가 필요하면 해당 XML에 `qpos`를 넣고 `mj_forward` 후 `data.xpos`/`data.xquat`를 위 33개 body 이름 순으로 추출한다. 공통 생성 로직: [`motion_utils.py`](general_motion_retargeting/utils/motion_utils.py).
