@REM python -m Model.asymmetric_pose_case.train ^
@REM   --epochs 10 ^
@REM   --experiment-name baseline_lstm_10epochs


python -m Model.asymmetric_pose_case.validation.rollout ^
  --run-dir Model/asymmetric_pose_case/runs/你的run目录 ^
  --fold 5 ^
  --checkpoint-name best.pt ^
  --sequence-id task1/train/mouse001_task1_annotator1 ^
  --start-t 100 ^
  --rollout-length 20

python -m Model.asymmetric_pose_case.validation.rollout ^
  --run-dir Model/asymmetric_pose_case/runs/20260711_090246_baseline_lstm_10epochs ^
  --fold 1 ^
  --checkpoint-name best.pt ^
  --sequence-id task1/train/mouse003_task1_annotator1 ^
  --start-t 500 ^
  --rollout-length 50 ^
  --output-name mouse003_start500_len50.npz