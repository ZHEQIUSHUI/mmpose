python tools/deploy.py \
configs/mmpose/pose-detection_onnxruntime_static.py \
../mmpose/work_dirs/rtmpose-s_8xb256-420e_custom-256x256/rtmpose-s_8xb256-420e_custom-256x256.py \
../mmpose/work_dirs/rtmpose-s_8xb256-420e_custom-256x256/epoch_240.pth \
demo/resources/f013311cc39bd2bc13260a124b355f6.png \
--work-dir work-dir --device cpu