# python scripts/pico_offline_to_robot.py \
#   --raw_pico_dir pico/offline_turn \
#   --output_dir pico/offline_turn_retargeted \
#   --robot robros_igris_c_v2 \
#   --actual_human_height 1.5 \
#   --motion_fps 100 \
#   --overwrite

# igris c
python scripts/vis_robot_motion_directory.py \
--robot robros_igris_c_v2 \
--robot_motion_dir /home/robros/workspace/GMR/pico/offline_turn_retargeted/ \
--root_quat_scalar_first true

