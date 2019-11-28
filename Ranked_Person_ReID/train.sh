#python3 tools/train.py --config_file='configs/softmax_ranked.yml' DATASETS.NAMES "('market1501')" 

#test
python3 tools/test.py --config_file='configs/softmax_ranked.yml' MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('market1501')" TEST.FEAT_NORM "('yes')" TEST.RE_RANKING "('yes')" MODEL.PRETRAIN_CHOICE "('self')" TEST.WEIGHT "('./work_space/best_model/resnet50_ibn_a_model_120.pth')"


#python tools/train.py --config_file='configs/softmax_ranked.yml' DATASETS.NAMES "('dukemtmc')" 


#test
#python tools/test.py --config_file='configs/softmax_ranked.yml' MODEL.DEVICE_ID "('your device id')" DATASETS.NAMES "('market1501')" TEST.FEAT_NORM "('yes')" TEST.RE_RANKING "('no')" MODEL.PRETRAIN_CHOICE "('self')" TEST.WEIGHT "('your path to trained checkpoints')"


#python tools/demo.py 

#python tools/compute_threshold.py --config_file='configs/softmax_ranked.yml' MODEL.PRETRAIN_CHOICE "('self')"  DATASETS.NAMES "('market1501')" TEST.WEIGHT "('models/resnet50_ibn_a/mar_resnet50_ibn_a_model.pth')"

