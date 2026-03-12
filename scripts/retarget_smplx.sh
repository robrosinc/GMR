# # robros igris_c
# python scripts/smplx_to_robot.py \
# --robot robros_igris_c_v2 \
# --smplx_file /home/robros/workspace/GMR/data/manual/smplx_like.npz \
# --record_video \
# --save_path output/tmp.pkl

python scripts/vis_robot_motion.py \
--robot robros_igris_c_v2 \
--robot_motion_path output/tmp.pkl

# # robros igris_max
# python scripts/smplx_to_robot.py \
# --robot robros_igris_max \
# --smplx_file /home/robros/workspace/GMR/data/ACCAD/Male1General_c3d/General_A5_-_Pick_Up_Box_stageii.npz \
# --record_video \
# --save_path output/pickup.pkl

# python scripts/vis_robot_motion.py \
# --robot robros_igris_max \
# --robot_motion_path output/pickup.pkl

# # fourier n1
# python scripts/smplx_to_robot.py \
# --robot fourier_n1 \
# --smplx_file /home/robros/workspace/GMR/data/CMU/90/90_01_stageii.npz \
# --record_video \
# --save_path tmp.pkl

# python scripts/vis_robot_motion.py \
# --robot fourier_n1 \
# --robot_motion_path tmp.pkl


# # unitree h1
# python scripts/smplx_to_robot.py \
# --robot unitree_h1 \
# --smplx_file /home/robros/workspace/GMR/data/ACCAD/Female1General_c3d/A5_-_pick_up_box_stageii.npz \
# --record_video \
# --save_path tmp.pkl

# python scripts/vis_robot_motion.py \
# --robot unitree_h1 \
# --robot_motion_path tmp.pkl