# TransferNet

## MetaQA-Text
1. Preprocess
```
python -m MetaQA-Text.preprocess --input_dir <PATH/TO/METAQA> --output_dir <PATH/TO/PROCESSED/FILES>
```
2. Train
```
python -m MetaQA-Text.train --input_dir <PATH/TO/PROCESSED/FILES> --save_dir <PATH/TO/CHECKPOINT>
```

## MetaQA-Text + 50% KB
1. Preprocess
```
python -m MetaQA-Text.preprocess --input_dir <PATH/TO/METAQA> --output_dir <PATH/TO/PROCESSED/FILES> --kb_ratio 0.5
```
2. Train
```
python -m MetaQA-Text.train --input_dir <PATH/TO/PROCESSED/FILES> --save_dir <PATH/TO/CHECKPOINT>
```
