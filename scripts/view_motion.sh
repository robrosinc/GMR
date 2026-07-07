########################################
# single retargeted motion viewer

# # robros igris c
# python scripts/vis_robot_motion.py \
# --robot robros_igris_c_v2 \
# --robot_motion_path /home/robros/workspace/GMR/output/AMASS-smplx/new_clips/retargetd/BMLrub/rub002/0016_lifting_heavy2_stageii.pkl \

###################################
# motion viewer - all single motions in directory

# igris c
python scripts/vis_robot_motion_directory.py \
--robot robros_igris_c_v2 \
--robot_motion_dir /home/robros/workspace/GMR/pico/recorded \
--curation_txt_path unseen_datasets.txt \
--root_quat_scalar_first true

# # unitree g1 23 dof - TWIST dataset
# python scripts/vis_robot_motion_directory.py \
# --robot unitree_g1_23dof \
# --robot_motion_dir /home/robros/workspace/motion_datas/retargeted/AMASS_filtered_retargeted/ \
# --root_quat_scalar_first false

# # unitree g1 29 dof
# python scripts/vis_robot_motion_directory.py \
# --robot unitree_g1 \
# --robot_motion_dir /home/robros/workspace/motion_datas/AMASS_g1_2/ \
# --root_quat_scalar_first false

###################################
# motion viewer - unified single motion dataset

# # igris c - refined motion npz
# python scripts/vis_robot_motion_npz.py \
#   --robot robros_igris_c_v2 \
#   --motion_npz_path /home/robros/workspace/ROBROS_LAB/source/robros_lab/robros_lab/tasks/tracking/motion_tracking/config/igris_c/dataset/motions_20h.npz \
#   --curation_txt_path filter_out.txt \
#   --root_quat_scalar_first true
