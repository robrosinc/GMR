# this is my unitree g1's ip in wifi
# redis_ip="192.168.110.24"
# localhost if you are using laptop to verify sim2sim or sim2real
redis_ip="localhost"

# the height (empirically) should be smaller than the actual human height, due to inaccuracy of the PICO estimation.
actual_human_height=1.6
python scripts/xrobot_teleop_to_robot_w_hand.py --robot robros_igris_c_v2 \
             --actual_human_height $actual_human_height \
             --redis_ip $redis_ip \
             --target_fps 100 \
             --measure_fps 1 \
             --save_pkl_enabled true \
             --save_pkl_dir output/pico_data \
             --save_pkl_every_n_steps 1000 \
             --save_pkl_fps 100 \
             --save_pkl_prefix pico \
             --save_pkl_toggle_with_right_key_one \
             --smooth \
             --pinch_mode
