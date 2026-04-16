# # robros igris_c
# python scripts/smplx_to_robot.py \
# --robot robros_igris_c_v2 \
# --smplx_file /home/robros/workspace/GMR/output/stand_to_jog_new.npz \
# --record_video \
# --save_path output/stand_to_jog_new.pkl

# python scripts/vis_robot_motion.py \
# --robot robros_igris_c_v2 \
# --robot_motion_path /home/robros/workspace/GMR/output/stand_to_jog_new.pkl

#################################################

# robros igris_c directory
python scripts/smplx_to_robot_dataset.py \
--robot robros_igris_c_v2 \
--src_folder /home/robros/workspace/GMR/data/ACCAD/Female1Gestures_c3d \
--tgt_folder /home/robros/workspace/GMR/output/retargeted/ \
--num_cpus 4 \

python scripts/vis_robot_motion_directory.py \
--robot robros_igris_c_v2 \
--robot_motion_dir /home/robros/workspace/GMR/output/retargeted

###############################################

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
# --smplx_file /home/robros/workspace/GMR/data/ACCAD/Male2MartialArtsStances_c3d/D1_-_stand_to_ready_stageii.npz \
# --record_video \
# --save_path tmp.pkl

# python scripts/vis_robot_motion.py \
# --robot fourier_n1 \
# --robot_motion_path tmp.pkl


# # adam lite
# python scripts/smplx_to_robot.py \
# --robot pnd_adam_lite \
# --smplx_file /home/robros/workspace/GMR/data/ACCAD/Male2MartialArtsStances_c3d/D1_-_stand_to_ready_stageii.npz \
# --record_video \
# --save_path tmp.pkl

# python scripts/vis_robot_motion.py \
# --robot pnd_adam_lite \
# --robot_motion_path tmp.pkl


# # kuavo s45
# python scripts/smplx_to_robot.py \
# --robot kuavo_s45 \
# --smplx_file /home/robros/workspace/GMR/data/ACCAD/Male2MartialArtsStances_c3d/D1_-_stand_to_ready_stageii.npz \
# --record_video \
# --save_path tmp.pkl

# python scripts/vis_robot_motion.py \
# --robot kuavo_s45 \
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
