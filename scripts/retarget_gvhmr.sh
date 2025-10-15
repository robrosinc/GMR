# # robros igris_c
# python scripts/gvhmr_to_robot.py \
# --robot robros_igris_c_v2 \
# --gvhmr_pred_file /home/robros/workspace/GVHMR/outputs/demo/walk_slow_woohyun/hmr4d_results.pt \
# --record_video \
# --save_path output/walk_slow_woohyun.pkl

# python scripts/vis_robot_motion.py \
# --robot robros_igris_c_v2 \
# --robot_motion_path output/walk_slow_woohyun.pkl


# robros igris_max
python scripts/gvhmr_to_robot.py \
--robot robros_igris_max \
--gvhmr_pred_file /home/robros/workspace/GVHMR/outputs/demo/bbiggi/hmr4d_results.pt \
--record_video \
--save_path output/bbiggi.pkl

python scripts/vis_robot_motion.py \
--robot robros_igris_max \
--robot_motion_path output/bbiggi.pkl


# # fourier n1
# python scripts/gvhmr_to_robot.py \
# --robot fourier_n1 \
# --gvhmr_pred_file /home/robros/workspace/GVHMR/outputs/demo/kick/hmr4d_results.pt \
# --record_video \
# --save_path kick.pkl

# python scripts/vis_robot_motion.py \
# --robot fourier_n1 \
# --robot_motion_path kick.pkl