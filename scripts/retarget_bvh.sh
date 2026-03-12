# robros igris_c
python scripts/bvh_to_robot.py \
--robot robros_igris_c_v2 \
--bvh_file /home/robros/workspace/motion_datas/lafan1/fallAndGetUp1_subject1.bvh \
--record_video \
--save_path output/fallAndGetUp1_subject1.pkl \
--format lafan1

python scripts/vis_robot_motion.py \
--robot robros_igris_c_v2 \
--robot_motion_path output/fallAndGetUp1_subject1.pkl

# # unitree g1
# python scripts/bvh_to_robot.py \
# --robot unitree_g1 \
# --bvh_file /home/robros/workspace/motion_datas/lafan1/walk1_subject1.bvh \
# --record_video \
# --save_path output/walk1_subject1.bvh \
# --format lafan1