python scripts/lafan_stream_to_robot.py \
--fifo_path /tmp/motion_frames.pipe \
--robot robros_igris_c_v2 \
--format lafan1 \
--motion_fps 30 \
--save_path output/lafan_stream.pkl \
--save_num_frames 300