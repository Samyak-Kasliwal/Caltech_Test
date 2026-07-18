# sh ./Model/asymmetric_pose_case/validation/rollout.sh

python -m Model.asymmetric_pose_case.validation.rollout \
  --run-dir Model/asymmetric_pose_case/runs/20260718_081453_asymmetric_pose \
  --fold 1 \
  --checkpoint-name best.pt \
  --sequence-id task1/train/mouse003_task1_annotator1 \
  --start-t 30 \
  --rollout-length 50 \
  --output-name mouse003_start30_len50.npz \
  --device cuda