#train
python3 main.py --config ./configs/Template-LBBDM-f4.yaml --train --sample_at_start --save_top --gpu_ids 0 \
--resume_model path/to/model_ckpt --resume_optim path/to/optim_ckpt

#test
python3 main.py --config ./configs/Template-LBBDM-f4.yaml --sample_to_eval --gpu_ids 0 \
--resume_model  /home/nwu-kiki/mydisk/pycharmprojects/BBDM+Trasformer+light/BBDM-main/results/LG_test_s_d_2/LBBDM-f4/checkpoint/last_model.pth--resume_optim path/to/optim_ckpt

#preprocess and evaluation
## rename
#python3 preprocess_and_evaluation.py -f rename_samples -r root/dir -s source/dir -t target/dir

## copy
#python3 preprocess_and_evaluation.py -f copy_samples -r root/dir -s source/dir -t target/dir

## LPIPS
#python3 preprocess_and_evaluation.py -f LPIPS -s source/dir -t target/dir -n 1

## max_min_LPIPS
#python3 preprocess_and_evaluation.py -f max_min_LPIPS -s source/dir -t target/dir -n 1

## diversity
#python3 preprocess_and_evaluation.py -f diversity -s source/dir -n 1

## fidelity
#fidelity --gpu 0 --fid --input1 path1 --input2 path2