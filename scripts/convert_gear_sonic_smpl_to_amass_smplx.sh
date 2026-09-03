# # convert single file
# python3 scripts/convert_gear_sonic_smpl_to_amass_smplx.py \
#   /home/robros/workspace/GR00T-WholeBodyControl/data/smpl_filtered/high_jump_R_002__A442_M.pkl \
#   /home/robros/workspace/motion_datas/bones_seed_smpl_amass_like/high_jump_R_002__A442_M.npz \
#   --root-rotation-axis x \
#   --root-rotation-degrees 90 \
#   --overwrite

# convert all files in directory
python3 scripts/convert_gear_sonic_smpl_to_amass_smplx.py \
  /home/robros/workspace/GR00T-WholeBodyControl/data/motions_sample/smpl_100 \
  /home/robros/workspace/motion_datas/bones_seed_smpl_amass_like \
  --skip-existing \
  --root-rotation-axis x \
  --root-rotation-degrees 90 \
