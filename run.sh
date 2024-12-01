DEVICE="cuda"
BATCH_SIZE="5"
ENTRY='main.py'
MAX_ITER=60

for TASK in task1, task2_merged, task3
do
    
    echo $TASK
    #CUDA_VISIBLE_DEVICES=1 python3 $ENTRY --model_name "image_only_$TASK" --mode image_only --task $TASK --batch_size $BATCH_SIZE --device $DEVICE --max_iter $MAX_ITER --image_model_to_load "./checkpoints/mae_visualize_vit_huge.pth" --use_tensorboard --debug

    # train text-only model (bert)
    #CUDA_VISIBLE_DEVICES=1 python3 $ENTRY --model_name "text_only_$TASK" --mode text_only --task $TASK --batch_size $BATCH_SIZE --device $DEVICE --max_iter $MAX_ITER --debug 

    # CUDA_VISIBLE_DEVICES=1 python3 $ENTRY --model_name "full_$TASK" --mode both --task $TASK --batch_size $BATCH_SIZE --device $DEVICE --max_iter $MAX_ITER \
    # --image_model_to_load "./output/image_only_$TASK/best.pt"  --text_model_to_load "./output/text_only_$TASK/best.pt" --debug

    # For pretrained MAE
    CUDA_VISIBLE_DEVICES=1 python3 $ENTRY --model_name "full_$TASK" --mode both --task $TASK --batch_size $BATCH_SIZE --device $DEVICE --max_iter $MAX_ITER --model_to_load "./output/full_task1/best_orig.pth" --eval #--image_model_to_load "./output/image_only_$TASK/best.pth" --text_model_to_load "./output/text_only_$TASK/best.pt" --use_tensorboard --debug
done
