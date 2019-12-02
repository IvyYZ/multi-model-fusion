# multi-model-fusion


# Person ReID

The codes are expanded on a [ReID-baseline](https://github.com/michuanhaohao/reid-strong-baseline.git)

Bag of tricks
- Warm up learning rate
- Random erasing augmentation
- Label smoothing
- Last stride

## Loss function
1. "ranked_loss", SoftMax(LS) + w*RLL, results as "Results". For RLL I use [Ranked List Loss for Deep Metric Learning](https://arxiv.org/abs/1903.03238), what the difference is I set the max distance of two feature to ALPHA. 
    
2. "cranked_loss", SoftMax(LS) + w*RLL(kmeans), before compute RLL i use kemeans to cluster features to help find hard samples. Now I can only get same performance to ranked_loss, so not report results. And for I need to find job, I haven’t studied it for a long time.


## Results

[models](https://drive.google.com/drive/folders/1CRQJYoCwx3mARckLb6dZpKzMZ_s3SwI3?usp=sharing)

NOTE: For the the limitations of the device (GTX1060 6G), while training Se_ResNet50 and Se_ResNext50 i can only set batchsize = 48, others is 64.

## Get Started

1. `cd` to folder where you want to download this repo

2. Run `git clone https://github.com/Qidian213/Ranked_Person_ReID`

3. Install dependencies:
    - [pytorch>=0.4](https://pytorch.org/)
    - torchvision
    - [ignite=0.1.2](https://github.com/pytorch/ignite) (Note: V0.2.0 may result in an error)
    - [yacs](https://github.com/rbgirshick/yacs)

4. Prepare dataset

    Create a directory to store reid datasets under this repo or outside this repo. Remember to set your path to the root of the dataset in `config/defaults.py` for all training and testing or set in every single config file in `configs/` or set in every single command.

    You can create a directory to store reid datasets under this repo via

    ```bash
    cd data
    ```

    （1）Market1501
    ```bash
    data
        market1501(match dataset) # this folder contains 7 files
            bounding_box_train/
            bounding_box_test_val/
            query_val
            bounding_box_test_A/
            query_A            
            bounding_box_test_B/
            query_B


  

5. Prepare pretrained model if you don't have

    （1）ResNet

    ```python
    from torchvision import models
    models.resnet50(pretrained=True)
    ```
    （2）Senet

    ```python
    import torch.utils.model_zoo as model_zoo
    model_zoo.load_url('the pth you want to download (specific urls are listed in  ./modeling/backbones/senet.py)')
    ```
    Then it will automatically download model in `~/.torch/models/`, you should set this path in `config/defaults.py` for all training or set in every single training config file in `configs/` or set in every single command.

    （3）ResNet_IBN_a , faster implementation

    You can download from here [[link]](https://drive.google.com/file/d/13lprTFafpXORqs7XXMLYaelbtw6NxQM1/view?usp=sharing)

    （4）Load your self-trained model
    If you want to continue your train process based on your self-trained model, you can change the configuration `PRETRAIN_CHOICE` from 'imagenet' to 'self' and set the `PRETRAIN_PATH` to your self-trained model.

6. If you want to know the detailed configurations and their meaning, please refer to `config/defaults.py`. If you want to set your own parameters, you can follow our method: create a new yml file, then set your own parameters.  Add `--config_file='configs/your yml file'` int the commands described below, then our code will merge your configuration. automatically.

## Train

1. Market1501

```bash
python tools/train.py --config_file='configs/softmax_ranked.yml' DATASETS.NAMES "('market1501')" 
```


## Test

```bash
python tools/test.py --config_file='configs/softmax_ranked.yml' MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('market1501')" TEST.FEAT_NORM "('yes')" TEST.RE_RANKING "('no')" MODEL.PRETRAIN_CHOICE "('self')" TEST.WEIGHT "('./work_space/resnet50_ibn_a_model_120.pth')"
```

#Multi-model
Different model will produce different txt files.
If U want to fusing three model, you need to modify data/dataset/eval_reid.py "flag".
flag=1 # To get mAP and Top1 for query_val
flag=2 # To get match submit file
flag=3 # To achieve multi-model fusion.

## Using and Compute threshold of negative and postive samples
model [[link]](https://pan.baidu.com/s/1n3YO87e8XSmgKHyrJB3YHg)
model result[[link]](https://pan.baidu.com/s/1IkwFvT68pnN4L81d_x3MjQ)

Download models from model. Put resnet50_ibn_a_model_120.pth in ./Ranked_Person_ReID/ work_place/

Download files(Rank_dist_tA.txt  MGN_dist_tA.txt  MHN_dist_tA).txt from model result, put them in ./Ranked_Person_ReID/ 
run test code, you can get test_fusion_A.txt. 
The file I submitted is from test_fusion_A.txt/test_fusion_B.txt
