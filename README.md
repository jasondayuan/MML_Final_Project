# Improved Linguistic-Aware Patch Slimming Framework for Fine-grained Cross-Modal Alignment
Repository for final project of Multimodal Learning (2024 Autumn Semester).

This project builds upon the paper *Linguistic-Aware Patch Slimming Framework for Fine-grained Cross-Modal Alignment* ([GitHub Repo](https://github.com/CrossmodalGroup/LAPS)).

## Environment
```
conda create --name laps python=3.9
conda activate laps
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers==4.44.0
pip install opencv-python tensorboard
pip install tensorboard_logger
pip install tensorflow
```

## Datasets
The caption files are already prepared for Flickr30K in the original repository, therefore it is only needed to download the images of the Flickr30K dataset. The Flickr30K images can be down loaded from https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset. After downloading, all images should be placed under ```data/flickr30k-images```.

## Training
The training arguments are pretty much the same as the original repository, refer to ```arguments.py``` for more information. In addition to the original arguments, we added some more arguments to accommodate for the improvements we proposed, the new arguments are delimited with hashtags in ```arguments.py```:
```
    ### New Arguments ###
    # LPS
    parser.add_argument('--use_cls_as_glob_embd', type=int, help='whether use the cls token as global embedding')
    parser.add_argument('--use_token_selection', type=int, help='whether use the token selection')
    parser.add_argument('--token_selection_strategy', default='ratio', type=str, help='the strategy for token selection')
    parser.add_argument('--token_selection_ratio', default=1.0, type=float, help='the ratio for token selection')
    # SPA
    parser.add_argument('--similarity_calc_method', type=str, help='the method for similarity calculation')
    parser.add_argument('--use_learnable_temp', default=0, type=int, help='whether use learnable temperature')
    parser.add_argument('--score_threshold', default=0.7, type=float, help='the threshold for score')
    #####################
```

For example, the command for training the final framework in our report would be:
```
python train.py \
--dataset f30k \
--gpu-id 0 \
--logger_name runs/f30k_vit \
--batch_size 64 \
--vit_type vit \
--embed_size 512 \
--sparse_ratio 0.5 \
--aggr_ratio 0.4 \
--f30k_img_path data/flickr30k-images \
--use_cls_as_glob_embd 1 \
--use_token_selection 1 \
--token_selection_strategy 'ratio' \
--token_selection_ratio 0.8 \
--similarity_calc_method 'attn' \
--use_learnable_temp 0
```

## Evaluation
Run ```eval.py``` to evaluate the trained models.
```
python eval.py --dataset f30k --data_path data/ --gpu-id 0
```

## Checkpoints
Below are the checkpoints for the experiments described in the report.

3.1, 3.2, 3.4 and Final Framework: https://disk.pku.edu.cn/link/AA32BA9A054E8B4B979D891BB5EACCF8E4

3.3: https://disk.pku.edu.cn/link/AA8081F61B76954E218D6EF1870A034976