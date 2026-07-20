# localhost if you are using laptop to verify sim2sim or sim2real
redis_ip="localhost"

# the height (empirically) should be smaller than the actual human height, due to inaccuracy of the PICO estimation.
actual_human_height=1.5
python scripts/xrobot_teleop_to_robot_w_hand.py --robot robros_igris_c_v2 \
             --actual_human_height $actual_human_height \
             --redis_ip $redis_ip \
             --target_fps 100 \
             --measure_fps 1 \
             --retarget_velocity_lpf_enabled true \
             --retarget_velocity_lpf_cutoff_hz 8.0 \
             --save_pkl_enabled true \
             --save_pkl_dir pico/lpf_debug/ \
             --save_pkl_every_n_steps 1000 \
             --save_pkl_fps 100 \
             --save_pkl_prefix pico \
             --pinch_mode \
             --show_raw_xrobot_skeleton false \
             --raw_xrobot_skeleton_offset 0.0 1.2 0.0 \
             --save_raw_xrobot_data false \
            #  --save_pkl_whole_session \
            #  --smooth \
