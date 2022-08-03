# XLM-Align

Code and models for the paper **Improving Pretrained Cross-Lingual Language Models via Self-Labeled Word Alignment**.

**Update**: release the aligner for word alignment on translation pairs. See [this page](https://github.com/CZWin32768/XLM-Align/tree/master/word_aligner).

**The XLM-Align pretraining code has uploaded to the [unilm](https://github.com/microsoft/unilm/tree/master/infoxlm) repo.**

## Introduction

XLM-Align is a pretrained cross-lingual language model that supports 94 languages. See details in our [paper](https://aclanthology.org/2021.acl-long.265/).

**Our Cross-Lingual Language Models**

- [1] **XLM-Align** (ACL 2021, [paper](https://aclanthology.org/2021.acl-long.265/), [repo](https://github.com/CZWin32768/XLM-Align), [model](https://huggingface.co/microsoft/xlm-align-base)) Improving Pretrained Cross-Lingual Language Models via Self-Labeled Word Alignment

- [2] **InfoXLM** (NAACL 2021, [paper](https://arxiv.org/pdf/2007.07834.pdf), [repo](https://github.com/microsoft/unilm/tree/master/infoxlm), [model](https://huggingface.co/microsoft/infoxlm-base)) InfoXLM: An Information-Theoretic Framework for Cross-Lingual Language Model Pre-Training.

- **XNLG** (AAAI 2020, [paper](https://arxiv.org/pdf/1909.10481.pdf), [repo](https://github.com/CZWin32768/XNLG)) multilingual/cross-lingual pre-trained model for natural language generation, e.g., finetuning XNLG with English abstractive summarization (AS) data and directly performing French AS or even Chinese-French AS.

- **mT6** ([paper](https://arxiv.org/abs/2104.08692)) mT6: Multilingual Pretrained Text-to-Text Transformer with Translation Pairs

- **XLM-E** ([paper](https://arxiv.org/pdf/2106.16138.pdf)) XLM-E: Cross-lingual Language Model Pre-training via ELECTRA

**Example Application Scenarios**

- **Learning natural language understanding (such as question answering) models for low-resource languages.** XLM-Align can perform cross-lingual transfer from a language to another one. Thus, you only need English QA training data to learn a multilingual QA model. Also, training data in other languages can also be jointly used for finetuning, if available.
- **Learning natural language generation models.** For example, XLM-Align can be used as a stage-1 model for pre-training a [XNLG](https://github.com/CZWin32768/XNLG) so that the model can perform cross-lingual transfer for generation tasks such as question generation, abstrative summarization, etc.
- **An initialization for neural machine translation** It has been shown that initalizing the NMT model with a pre-trained cross-lingual encoder significantly improves the results. See more details in [XLM-T](https://arxiv.org/pdf/2012.15547.pdf).
- **A word aligner** XLM-Align can serve as a word aligner that finds corresponding words between translation pairs. 

## How to Use

### From huggingface model hub

We provide the models in huggingface format, so you can use the model directly with huggingface API:

**XLM-Align**
```python
model = AutoModel.from_pretrained("microsoft/xlm-align-base")
tokenizer = AutoTokenizer.from_pretrained("microsoft/xlm-align-base")
```

Note: We have moved the XLM-Align model from `CZWin32768/xlm-align` to `microsoft/xlm-align-base`. We will also preseve the original repo for compatibility. So, there is no difference between these two repositories.

**InfoXLM-base**
```python
model = AutoModel.from_pretrained("microsoft/infoxlm-base")
tokenizer = AutoTokenizer.from_pretrained("microsoft/infoxlm-base")
```

**InfoXLM-large**
```python
model = AutoModel.from_pretrained("microsoft/infoxlm-large")
tokenizer = AutoTokenizer.from_pretrained("microsoft/infoxlm-large")
```

### Finetuning on end tasks

Our models use the same vocabulary, tokenizer, and architecture with XLM-Roberta. So you can directly use the existing codes for finetuning XLM-R, **just by replacing the model name from `xlm-roberta-base` to `microsoft/xlm-align-base`, `microsoft/infoxlm-base`, or `microsoft/infoxlm-base`**.

For example, you can evaluate our model with [xTune](https://github.com/bozheng-hit/xTune) on the XTREME benchmark.

### Evaluation Results

XTREME cross-lingual understanding tasks:

| Model | POS | NER  | XQuAD | MLQA | TyDiQA | XNLI | PAWS-X | Avg |
|:----:|:----:|:----:|:----:|:-----:|:----:|:-----:|:----:|:----:|
| XLM-R_base | 75.6 | 61.8 | 71.9 / 56.4 | 65.1 / 47.2 | 55.4 / 38.3 | 75.0 | 84.9 | 66.4 |
| InfoXLM_base | - | - | - | **68.1** / 49.6 | - | **76.5** | - | - |
| XLM-Align_base | **76.0** | **63.7** | **74.7 / 59.0** | **68.1 / 49.8**  |  **62.1 / 44.8** | 76.2  | **86.8**  | **68.9** |


(The models are finetuned under the cross-lingual transfer setting, i.e., finetuning only with Enlgish training data but directly test on target langauges)

## Pretraining XLM-Align

**We have uploaded the pretraining code to the [unilm](https://github.com/microsoft/unilm/tree/master/infoxlm) repo.**

Here is an example for pretraining XLM-Align-base:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python src-infoxlm/train.py ${MLM_DATA_DIR} \
--task xlm_align --criterion dwa_mlm_tlm \
--tlm_data ${TLM_DATA_DIR} \
--arch xlm_align_base --sample-break-mode complete --tokens-per-sample 512 \
--optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 \
--clip-norm 1.0 --lr-scheduler polynomial_decay --lr 0.0002 \
--warmup-updates 10000 --total-num-update 200000 --max-update 200000 \
--dropout 0.0 --attention-dropout 0.0 --weight-decay 0.01 \
--max-sentences 16 --update-freq 16 --log-format simple \
--log-interval 1 --disable-validation --save-interval-updates 5000 --no-epoch-checkpoints \
--fp16 --fp16-init-scale 128 --fp16-scale-window 128 --min-loss-scale 0.0001 \
--seed 1 \
--save-dir .${SAVE_DIR} \
--tensorboard-logdir .${SAVE_DIR}/tb-log \
--roberta-model-path /path/to/model.pt \
--num-workers 2 --ddp-backend=c10d --distributed-no-spawn \
--wa_layer 10 --wa_max_count 2 --sinkhorn_iter 2
```

- `${MLM_DATA_DIR}`: directory to mlm training data.
- `${SAVE_DIR}`: checkpoints are saved in this folder.
- `--max-sentences 8`: batch size per GPU.
- `--update-freq 32`: gradient accumulation steps. (total batch size = TOTAL_NUM_GPU x max-sentences x update-freq = 8 x 16 x 16 = 2048)
- `--roberta-model-path`: the checkpoint path to an existing roberta model (as the initialization of the current model). For learning from scratch, remove this line.
- `--wa_layer`: the layer to perform word alignment self-labeling
- `--wa_max_count`: the number of iterative alignment filtering
- `--sinkhorn_iter`: the number of the iteration in Sinkhorn's algorithm

See more details at the [InfoXLM page](https://github.com/microsoft/unilm/tree/master/infoxlm).

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

[2] **InfoXLM** (NAACL 2021, [paper](https://arxiv.org/pdf/2007.07834.pdf), [repo](https://github.com/microsoft/unilm/tree/master/infoxlm), [model](https://huggingface.co/microsoft/infoxlm-base)) InfoXLM: An Information-Theoretic Framework for Cross-Lingual Language Model Pre-Training.

```
@inproceedings{chi-etal-2021-infoxlm,
  title = "{I}nfo{XLM}: An Information-Theoretic Framework for Cross-Lingual Language Model Pre-Training",
  author={Chi, Zewen and Dong, Li and Wei, Furu and Yang, Nan and Singhal, Saksham and Wang, Wenhui and Song, Xia and Mao, Xian-Ling and Huang, Heyan and Zhou, Ming},
  booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
  month = jun,
  year = "2021",
  address = "Online",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2021.naacl-main.280",
  doi = "10.18653/v1/2021.naacl-main.280",
  pages = "3576--3588",}
```

### Contact Information

Zewen Chi (`chizewen@outlook.com`)

