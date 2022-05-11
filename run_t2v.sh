# main do_t2v
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 \
main.py --do_t2v 1 --workers 0 --n_display 50 --epochs 5 --lr 1e-4 --coef_lr 1e-3 --batch_size 1 \
--batch_size_val 1 --anno_path data/MSR-VTT/anns --video_path data/MSR-VTT/videos --datatype msrvtt \
--max_words 32 --max_frames 12 --video_framerate 1 \
--base_encoder ViT-B/32 --agg_module seqTransf \
--interaction wti --wti_arch 2 --cdcr 3 --cdcr_alpha1 0.11 --cdcr_alpha2 0.0 --cdcr_lambda 0.001 \
--output_dir ckpts_hj/ckpt_msrvtt_wti_cdcr_5t6n381x/output_2 \
--init_model ckpts_hj/ckpt_msrvtt_wti_cdcr_5t6n381x/pytorch_model.bin.2