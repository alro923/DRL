# Disentangled Representation Learning for Text-Video Retrieval
[![MSR-VTT](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/disentangled-representation-learning-for-text/video-retrieval-on-msr-vtt-1ka)](https://paperswithcode.com/sota/video-retrieval-on-msr-vtt-1ka?p=disentangled-representation-learning-for-text)
[![DiDeMo](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/disentangled-representation-learning-for-text/video-retrieval-on-didemo)](https://paperswithcode.com/sota/video-retrieval-on-didemo?p=disentangled-representation-learning-for-text)

This is a forked repo from [a PyTorch implementation](https://github.com/foolwood/DRL) of the paper [Disentangled Representation Learning for Text-Video Retrieval](https://arxiv.org/abs/2203.07111):
<p align="center">
  <img src="demo/pipeline.png" width="800">
</p>

```
@Article{DRLTVR2022,
  author  = {Qiang Wang and Yanhao Zhang and Yun Zheng and Pan Pan and Xian-Sheng Hua},
  journal = {arXiv:2203.07111},
  title   = {Disentangled Representation Learning for Text-Video Retrieval},
  year    = {2022},
}
```

#### Setup code environment in V100
```shell
# Starting from my workspace
bash run pd.sh 
git clone git@github.com:alro923/DRL.git
cd DRL
```

#### Download CLIP Model (as pretraining)

```shell
cd tvr/models
wget https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
wget https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
```

#### Datasets : using soft link

```shell
cd data/MSR-VTT
ln -s /data5/datasets/MSRVTT/videos/all videos
```

#### Train : fine-tuning on MSR-VTT 1k

```shell
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 \
main.py --do_train 1 --workers 8 --n_display 50 \
--epochs 5 --lr 1e-4 --coef_lr 1e-3 --batch_size 128 --batch_size_val 128 \
--anno_path data/MSR-VTT/anns --video_path data/MSR-VTT/videos --datatype msrvtt \
--max_words 32 --max_frames 12 --video_framerate 1 \
--base_encoder ViT-B/32 --agg_module seqTransf \
--interaction wti --wti_arch 2 --cdcr 3 \
--cdcr_alpha1 0.11 --cdcr_alpha2 0.0 --cdcr_lambda 0.001 \
--output_dir ckpts/ckpt_msrvtt_wti_cdcr
```

#### Test : using a sentence from MSR-VTT 1k test set
```sheell
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
python -m torch.distributed.launch --nproc_per_node=4 \
main.py --do_eval 1 --workers 8 --n_display 50 \
--epochs 5 --lr 1e-4 --coef_lr 1e-3 \
--batch_size 128 --batch_size_val 128 \
--anno_path data/MSR-VTT/anns --video_path data/MSR-VTT/videos --datatype msrvtt \
--max_words 32 --max_frames 12 --video_framerate 1 \
--base_encoder ViT-B/32 --agg_module seqTransf \
--interaction wti --wti_arch 2 --cdcr 3 \
--cdcr_alpha1 0.11 --cdcr_alpha2 0.0 --cdcr_lambda 0.001 \
--output_dir output/t2v \
--init_model ckpts/[CKPT_FOR_TEST]
```

---
> License and Acknowledgements from original repo !
### License
See [LICENSE](LICENSE) for details.

### Acknowledgments
[Our code](https://github.com/foolwood/DRL) is partly based on [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip).
