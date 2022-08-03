# XLM-Align Word Aligner

An implementation of optimal-transport word aligner proposed in the XLM-Align paper, which is used for word alignment on translation pairs.

The results are shown in the following table. We re-implement the xlm-align word aligner for the huggingface model hub. Our original implementation is based on fairseq, so the results are slightly different with the reported ones.


| Aligner | Pre-trained Model | de-en | en-fr | en-hi | ro-en |
| ------  | ----------------- | ----- | ----- | ----- | ----- |
| SimAlign (argmax) | XLM-R   | 19.   | 7.    | 39.   | 29.   |
| Our Aligner  | XLM-R | 19.11 | 6.83 | 33.73 | 28.73 |
| Our Aligner | XLM-Align | **17.42** | **6.71** | **31.93** | **26.50** |


## How to Use

**Prepare environment**

```
git clone https://github.com/CZWin32768/fairseq
cd fairseq
git checkout czw
pip install --editable .
pip install --user sentencepiece==0.1.95
pip install --user transformers==4.3.2
```

**Prepare model and data**

- Test data `alignment-test-sets` are following SimAlign.

- Download pre-trained XLM-Align from huggingface:

```
git lfs install
git clone https://huggingface.co/microsoft/xlm-align-base
```

**Run Word Aligner with pre-trained XLM-Align**

```
python ./xlmalign-ot-aligner.py \
--bpe_path /path/to/xlm-align-base/sentencepiece.bpe.model \
--vocab_path ./fairseq-dict.txt \
--model_type xlmr \
--model_name_or_path /path/to/xlm-align-base \
--config_name /path/to/xlm-align-base/config.json \
--test_set_dir /path/to/alignment-test-sets
```

Results:
```
Language pair: de-en
p: 47.74, 64.12, 68.40, 73.77, 80.24, 83.47, 87.31, 87.35, 88.17, 88.27, 86.23, 85.34, 85.38
r: 7.85, 53.18, 55.50, 60.25, 67.35, 70.32, 74.64, 75.52, 76.58, 77.47, 75.81, 76.62, 76.76
f1: 13.49, 58.14, 61.28, 66.33, 73.23, 76.33, 80.48, 81.01, 81.97, 82.52, 80.69, 80.74, 80.84
aer: 86.47, 41.83, 38.68, 33.63, 26.72, 23.61, 19.46, 18.93, 17.97, 17.42, 19.25, 19.20, 19.10
Language pair: en-fr
p: 58.33, 70.10, 76.71, 82.73, 88.16, 89.87, 92.40, 92.19, 92.63, 93.17, 91.75, 91.15, 91.14
r: 17.24, 72.36, 77.86, 82.44, 88.48, 89.92, 92.52, 92.47, 93.12, 93.46, 92.30, 92.55, 91.78
f1: 26.61, 71.21, 77.28, 82.59, 88.32, 89.90, 92.46, 92.33, 92.87, 93.31, 92.03, 91.84, 91.46
aer: 71.28, 28.95, 22.80, 17.39, 11.70, 10.11, 7.55, 7.70, 7.18, 6.71, 8.03, 8.30, 8.61
Language pair: en-hi
p: 46.55, 62.14, 64.26, 67.82, 74.31, 77.56, 81.25, 80.98, 81.75, 81.57, 78.19, 77.61, 78.70
r: 7.67, 37.97, 41.73, 44.43, 49.89, 53.23, 57.20, 56.49, 57.56, 58.41, 54.72, 55.36, 56.64
f1: 13.16, 47.14, 50.60, 53.69, 59.70, 63.13, 67.14, 66.56, 67.56, 68.07, 64.38, 64.62, 65.87
aer: 86.84, 52.86, 49.40, 46.31, 40.30, 36.87, 32.86, 33.44, 32.44, 31.93, 35.62, 35.38, 34.13
Language pair: ro-en
p: 42.81, 56.87, 64.11, 71.65, 79.04, 82.36, 86.65, 86.73, 88.06, 88.11, 86.20, 85.61, 86.12
r: 5.38, 36.58, 40.35, 45.84, 53.19, 55.83, 60.08, 60.76, 61.85, 63.04, 61.81, 62.39, 61.51
f1: 9.57, 44.52, 49.53, 55.91, 63.59, 66.55, 70.96, 71.46, 72.67, 73.50, 72.00, 72.18, 71.77
aer: 90.43, 55.48, 50.47, 44.09, 36.41, 33.45, 29.04, 28.54, 27.33, 26.50, 28.00, 27.82, 28.23
```

There are 13 scores in each line, which are the results using the outputs of the 0-12th layers. The 0-th layer means the word embedding layer.

**Run Word Aligner with pre-trained XLM-R**

```
python ./xlmalign-ot-aligner.py \
--bpe_path /path/to/xlm-roberta-base/sentencepiece.bpe.model \
--vocab_path ./fairseq-dict.txt \
--model_type xlmr \
--model_name_or_path /path/to/xlm-roberta-base \
--config_name /path/to/xlm-roberta-base/config.json \
--test_set_dir /path/to/alignment-test-sets
```

Results:
```
Language pair: de-en
p: 58.41, 61.19, 65.12, 70.81, 76.82, 81.26, 85.44, 86.65, 88.19, 87.68, 87.24, 86.45, 85.26
r: 15.86, 49.98, 52.88, 57.48, 63.68, 67.81, 71.67, 72.55, 74.60, 72.58, 68.66, 63.57, 65.65
f1: 24.95, 55.02, 58.36, 63.46, 69.64, 73.93, 77.96, 78.98, 80.83, 79.42, 76.84, 73.26, 74.18
aer: 74.97, 44.95, 41.60, 36.50, 30.32, 26.02, 21.99, 20.96, 19.11, 20.51, 23.07, 26.64, 25.74
Language pair: en-fr
p: 65.50, 68.42, 73.33, 79.01, 84.55, 88.27, 91.73, 92.99, 93.35, 92.37, 91.45, 90.86, 89.59
r: 39.05, 69.99, 74.07, 79.20, 85.39, 87.87, 91.51, 92.42, 92.92, 91.21, 87.54, 82.12, 84.87
f1: 48.93, 69.19, 73.70, 79.10, 84.97, 88.07, 91.62, 92.70, 93.13, 91.78, 89.45, 86.27, 87.17
aer: 49.24, 30.92, 26.35, 20.91, 15.10, 11.90, 8.37, 7.25, 6.83, 8.11, 10.19, 13.00, 12.47
Language pair: en-hi
p: 53.88, 60.38, 61.66, 64.38, 70.34, 75.51, 81.10, 81.86, 83.01, 82.39, 83.03, 81.01, 77.30
r: 9.87, 38.61, 39.96, 41.94, 46.63, 49.68, 53.30, 53.80, 55.15, 53.44, 51.03, 45.71, 46.42
f1: 16.68, 47.10, 48.49, 50.80, 56.08, 59.93, 64.33, 64.93, 66.27, 64.83, 63.21, 58.44, 58.00
aer: 83.32, 52.90, 51.51, 49.20, 43.92, 40.07, 35.67, 35.07, 33.73, 35.17, 36.79, 41.56, 42.00
Language pair: ro-en
p: 56.32, 57.39, 63.10, 69.10, 75.72, 80.34, 84.86, 86.19, 88.26, 87.48, 85.88, 85.05, 83.82
r: 15.58, 36.50, 40.00, 44.65, 49.75, 52.87, 56.81, 57.78, 59.77, 59.15, 56.90, 52.55, 49.39
f1: 24.40, 44.62, 48.96, 54.24, 60.05, 63.77, 68.06, 69.18, 71.27, 70.58, 68.45, 64.96, 62.16
aer: 75.60, 55.38, 51.04, 45.76, 39.95, 36.23, 31.94, 30.82, 28.73, 29.42, 31.55, 35.04, 37.84
```

## References

Please cite the paper if you found the resources in this repository useful.

[1] **XLM-Align** (ACL 2021, [paper](https://aclanthology.org/2021.acl-long.265/), [repo](https://github.com/CZWin32768/XLM-Align), [model](https://huggingface.co/microsoft/xlm-align-base)) Improving Pretrained Cross-Lingual Language Models via Self-Labeled Word Alignment

```
@inproceedings{xlmalign,
  title = "Improving Pretrained Cross-Lingual Language Models via Self-Labeled Word Alignment",
  author={Zewen Chi and Li Dong and Bo Zheng and Shaohan Huang and Xian-Ling Mao and Heyan Huang and Furu Wei},
  booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
  month = aug,
  year = "2021",
  address = "Online",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2021.acl-long.265",
  doi = "10.18653/v1/2021.acl-long.265",
  pages = "3418--3430",}
```
