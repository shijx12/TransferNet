# TransferNet

## MetaQA-KB
1. Preprocess
```
python -m MetaQA-KB.preprocess --input_dir <PATH/TO/METAQA> --output_dir <PATH/TO/PROCESSED/FILES>
```
2. Train
```
python -m MetaQA-KB.train --glove_pt <PATH/TO/GLOVE/PICKLE> --input_dir <PATH/TO/PROCESSED/FILES> --save_dir <PATH/TO/CHECKPOINT>
```

## MetaQA-Text
1. Preprocess
```
python -m MetaQA-Text.preprocess --input_dir <PATH/TO/METAQA> --output_dir <PATH/TO/PROCESSED/FILES>
```
2. Train
```
python -m MetaQA-Text.train --glove_pt <PATH/TO/GLOVE/PICKLE> --input_dir <PATH/TO/PROCESSED/FILES> --save_dir <PATH/TO/CHECKPOINT>
```

## MetaQA-Text + 50% KB
1. Preprocess
```
python -m MetaQA-Text.preprocess --input_dir <PATH/TO/METAQA> --output_dir <PATH/TO/PROCESSED/FILES> --kb_ratio 0.5
```
2. Train, it needs more active paths than only MetaQA-Text
```
python -m MetaQA-Text.train --input_dir <PATH/TO/PROCESSED/FILES> --save_dir <PATH/TO/CHECKPOINT> --max_active 800 --batch_size 32
```


## WebQSP
Dowload data from [https://drive.google.com/drive/folders/1RlqGBMo45lTmWz9MUPTq-0KcjSd3ujxc?usp=sharing], which has been processed by [EmbedKGQA](https://github.com/malllabiisc/EmbedKGQA).
Train
```
python -m WebQSP.train --input_dir <PATH/TO/UNZIPPED/DATA> --save_dir <PATH/TO/CHECKPOINT>
```
