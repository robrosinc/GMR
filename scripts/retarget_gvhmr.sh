# robros igris_c
python scripts/gvhmr_to_robot.py \
--robot robros_igris_c_v2 \
--gvhmr_pred_file /home/robros/workspace/GVHMR/outputs/demo/20251102_205108_rotate/hmr4d_results.pt \
--record_video \
--save_path output/20251102_205108_rotate.pkl

python scripts/vis_robot_motion.py \
--robot robros_igris_c_v2 \
--robot_motion_path output/20251102_205108_rotate.pkl

######################################################

# # robros igris_max
# python scripts/gvhmr_to_robot.py \
# --robot robros_igris_max \
# --gvhmr_pred_file /home/robros/workspace/GVHMR/outputs/demo/pickup_1hand/hmr4d_results.pt \
# --record_video \
# --save_path output/pickup_1hand_max.pkl

# python scripts/vis_robot_motion.py \
# --robot robros_igris_max \
# --robot_motion_path output/pickup_each_hand_max.pkl

#####################################################

# # fourier n1
# python scripts/gvhmr_to_robot.py \
# --robot fourier_n1 \
# --gvhmr_pred_file /home/robros/workspace/GVHMR/outputs/demo/kick/hmr4d_results.pt \
# --record_video \
# --save_path kick.pkl

# python scripts/vis_robot_motion.py \
# --robot fourier_n1 \
# --robot_motion_path kick.pkl