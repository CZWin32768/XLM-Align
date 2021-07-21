# XLM-Align

Code and models for the paper **Improving Pretrained Cross-Lingual Language Models via Self-Labeled Word Alignment** (ACL-2021, [paper](https://arxiv.org/pdf/2106.06381.pdf), [github](https://github.com/CZWin32768/XLM-Align)).

## Introduction

XLM-Align is a pretrained cross-lingual language model that supports 94 languages. See details in our [paper](https://arxiv.org/pdf/2106.06381.pdf).

**Example Application Scenarios**

- **Learning natural language understanding (such as question answering) models for low-resource languages.** XLM-Align can perform cross-lingual transfer from a language to another one. Thus, you only need English QA training data to learn a multilingual QA model. Also, training data in other languages can also be jointly used for finetuning, if available.
- **Learning natural language generation models.** For example, XLM-Align can be used as a stage-1 model for pre-training a [XNLG](https://github.com/CZWin32768/XNLG) so that the model can perform cross-lingual transfer for generation tasks such as question generation, abstrative summarization, etc.
- **An initialization for neural machine translation** It has been shown that initalizing the NMT model with a pre-trained cross-lingual encoder significantly improves the results. See more details in [XLM-T](https://arxiv.org/pdf/2012.15547.pdf).
- **A word aligner** XLM-Align can serve as a word aligner that finds corresponding words between translation pairs. 

## How to Use

XLM-Align has been uploaded to huggingface hub. You can use this model directly with huggingface API:

```python
model = AutoModel.from_pretrained("CZWin32768/xlm-align")
tokenizer = AutoTokenizer.from_pretrained("CZWin32768/xlm-align")
```

or directly download the model from [this page](https://huggingface.co/CZWin32768/xlm-align).


MD5:

```
b9d214025837250ede2f69c9385f812c  config.json
6005db708eb4bab5b85fa3976b9db85b  pytorch_model.bin
bf25eb5120ad92ef5c7d8596b5dc4046  sentencepiece.bpe.model
eedbd60a7268b9fc45981b849664f747  tokenizer.json
```

## Evaluation Results

XTREME cross-lingual understanding tasks:

| Model | POS | NER  | XQuAD | MLQA | TyDiQA | XNLI | PAWS-X | Avg |
|:----:|:----:|:----:|:----:|:-----:|:----:|:-----:|:----:|:----:|
| XLM-R_base | 75.6 | 61.8 | 71.9 / 56.4 | 65.1 / 47.2 | 55.4 / 38.3 | 75.0 | 84.9 | 66.4 |
| XLM-Align | **76.0** | **63.7** | **74.7 / 59.0** | **68.1 / 49.8**  |  **62.1 / 44.8** | **76.2**  | **86.8**  | **68.9** |

(The models are finetuned under the cross-lingual transfer setting, i.e., finetuning only with Enlgish training data but directly test on target langauges)

## References

Please cite the paper if you found the resources in this repository useful.

```
@article{xlmalign,
  title={Improving Pretrained Cross-Lingual Language Models via Self-Labeled Word Alignment},
  author={Zewen Chi and Li Dong and Bo Zheng and Shaohan Huang and Xian-Ling Mao and Heyan Huang and Furu Wei},
  journal={arXiv preprint arXiv:2106.06381},
  year={2021}
}
```
