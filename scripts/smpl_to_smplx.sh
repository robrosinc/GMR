# for SMPL converted motion_million test. root rotate, offset
python scripts/smpl_to_smplx.py \
  --input_file /home/robros/workspace/motion-toolbox/outputs/format272_to_smpl/000000_smpl_params.npz \
  --output_file motion_million_test.npz \
  --gender neutral \
  --root_rotation_axis x \
  --root_rotation_degrees 90 \
  --rotate_translation \
  --root_height_axis z \
  --root_height_offset 0.97
