CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_train.sh configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_custom-256x256.py 4 --resume
