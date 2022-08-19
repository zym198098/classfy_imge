import autogluon.core as ag
from autogluon.vision import ImagePredictor, ImageDataset
from autogluon.vision.configs.presets_configs import *
# train_dataset, _, test_dataset = ImageDataset.from_folders('https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip')
train_dataset,_,_= ImageDataset.from_folders('/home/zym/下载/egg1')
print(train_dataset)
predictor = ImagePredictor()
models=predictor.list_models()
print(models)

my_hy={
                    'hyperparameters': {
                        'model': ('resnet50'),
                        'lr': Real(1e-6, 1e-3, log=True),
                        'batch_size': Categorical(8, 16, 32,64,128),
                        'epochs': 200,
                        'early_stop_patience': 5
                        },
                   
                    'time_limit': 1*3600,
                }


# since the original dataset does not provide validation split, the `fit` function splits it randomly with 90/10 ratio
predictor.fit(train_dataset,hyperparameters={'epochs': 10})  # you can trust the default config, we reduce the # epoch to save some build time