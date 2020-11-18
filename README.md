# TransferNet

## dependencies
- [transformers](https://github.com/huggingface/transformers)

## Datasets download
- [MetaQA](https://goo.gl/f3AmcY), we only use its vanilla version
- [MovieQA](http://www.thespermwhale.com/jaseweston/babi/movieqa.tar.gz), we need its `knowledge_source/wiki.txt` as the text corpus for our MetaQA-Text experiments. Copy the file into the folder of MetaQA, and put it together with `kb.txt`. The files of MetaQA should be something like
```
MetaQA
+-- kb
|   +-- kb.txt
|   +-- wiki.txt
+-- 1-hop
|   +-- vanilla
|   |   +-- qa_train.txt
|   |   +-- qa_dev.txt
|   |   +-- qa_test.txt
+-- 2-hop
+-- 3-hop
```
- [WebQSP](https://drive.google.com/drive/folders/1RlqGBMo45lTmWz9MUPTq-0KcjSd3ujxc?usp=sharing), which has been processed by [EmbedKGQA](https://github.com/malllabiisc/EmbedKGQA).



## Experiments
### MetaQA-KB
1. Preprocess
```
python -m MetaQA-KB.preprocess --input_dir <PATH/TO/METAQA> --output_dir <PATH/TO/PROCESSED/FILES>
```
2. Train
```
python -m MetaQA-KB.train --glove_pt <PATH/TO/GLOVE/PICKLE> --input_dir <PATH/TO/PROCESSED/FILES> --save_dir <PATH/TO/CHECKPOINT>
```
3. Predict
```
python -m MetaQA-KB.predict --input_dir <PATH/TO/PROCESSED/FILES> --ckpt <PATH/TO/CHECKPOINT>
```

### MetaQA-Text
1. Preprocess
```
python -m MetaQA-Text.preprocess --input_dir <PATH/TO/METAQA> --output_dir <PATH/TO/PROCESSED/FILES>
```
2. Train
```
python -m MetaQA-Text.train --glove_pt <PATH/TO/GLOVE/PICKLE> --input_dir <PATH/TO/PROCESSED/FILES> --save_dir <PATH/TO/CHECKPOINT>
```

### MetaQA-Text + 50% KB
1. Preprocess
```
python -m MetaQA-Text.preprocess --input_dir <PATH/TO/METAQA> --output_dir <PATH/TO/PROCESSED/FILES> --kb_ratio 0.5
```
2. Train, it needs more active paths than MetaQA-Text
```
python -m MetaQA-Text.train --input_dir <PATH/TO/PROCESSED/FILES> --save_dir <PATH/TO/CHECKPOINT> --max_active 800 --batch_size 32
```

### WebQSP
```
python -m WebQSP.train --input_dir <PATH/TO/UNZIPPED/DATA> --save_dir <PATH/TO/CHECKPOINT>
```
