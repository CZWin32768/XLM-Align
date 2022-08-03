import argparse
import os
from collections import OrderedDict

import numpy as np
import sentencepiece as spm
import torch

from fairseq import utils
from fairseq.data import FairseqDataset, data_utils
from fairseq.data.dictionary import Dictionary
from torch.utils.data.dataloader import DataLoader
from transformers import XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizer


MODEL_CLASSES = {
    "xlmr": (XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizer),
}


def move_to_device(sample, device):

  def _move_to_device(tensor):
    return tensor.to(device=device)
  
  return utils.apply_to_sample(_move_to_device, sample)


def load_gold_alignments(args):
  aid2align = OrderedDict()
  with open(args.gold_align) as fp:
    for line in fp:
      line = line.strip()
      if line == "": continue
      _split = line.split()
      aid = int(_split[0])
      if aid not in aid2align: aid2align[aid] = (set(), set())
      S, P = aid2align[aid]

      isP = False
      if len(_split) == 3:
        isP = False
      elif len(_split) == 4:
        if _split[3] == "P": isP = True
        elif _split[3] == "S": isP = False
        else: raise ValueError("split[1] should be P or S")
      else:
        raise ValueError("len split should be 3 or 4")
      
      sid, tid = int(_split[1]), int(_split[2])
      if not isP: S.add((sid, tid))
      P.add((sid, tid))

  return aid2align

def eval_results(gold_align, align_results):

  assert len(gold_align) == len(align_results), (len(gold_align), len(align_results))
  num_A = 0
  num_S = 0
  inter_A_S = 0
  inter_A_P = 0

  for SP, A in zip(gold_align.values(), align_results):
    S, P = SP
    for a in A:
      if a in S: inter_A_S += 1
      if a in P: inter_A_P += 1
    num_A += len(A)
    num_S += len(S)

  precision = inter_A_P / num_A
  recall = inter_A_S / num_S
  f1 = 2.0 * precision * recall / (precision + recall)
  aer = 1 - (inter_A_P + inter_A_S) / (num_A + num_S)
  # print("prec=%.2f rec=%.2f f1=%.2f aer=%.2f" % (precision * 100, recall * 100, f1 * 100, aer * 100))
  return precision * 100.0, recall * 100.0, f1 * 100.0, aer * 100.0


class RawPair2WordAlignDataset(FairseqDataset):

  def __init__(self, src_lines, trg_lines, bpe_path, vocab):
    assert len(src_lines) == len(trg_lines)
    self.src_lines = src_lines
    self.trg_lines = trg_lines
    self.bpe_path = bpe_path
    self.spp = spm.SentencePieceProcessor(model_file=self.bpe_path)
    self.vocab = vocab
    self.bos_idx = self.vocab.bos()
    self.eos_idx = self.vocab.eos()
    self.bos_word = self.vocab[self.bos_idx]
    self.eos_word = self.vocab[self.eos_idx]

  @property
  def sizes(self):
    # WALKAROUND
    return [1] * len(self)
  
  def size(self, index):
    # WALKAROUND
    return 1
  
  def word2indices(self, word):
    assert isinstance(word, str)
    pieces = self.spp.encode(word, out_type=str, enable_sampling=False)
    indices = [self.vocab.index(piece) for piece in pieces]
    return indices
  
  def get_indices_and_gid(self, line, offset=0):
    line_indices = []
    token_group_id = []
    words = line.split()
    for gid, word in enumerate(words):
      for subword_idx in self.word2indices(word):
        line_indices.append(subword_idx)
        token_group_id.append(gid + offset)
    return line_indices, token_group_id, words
  
  def __getitem__(self, index):

    src_line = self.src_lines[index]
    src_indices, src_gid, src_words = self.get_indices_and_gid(src_line, offset=0)

    trg_gid_offset = src_gid[-1]
    trg_line = self.trg_lines[index]
    # trg_indices, trg_gid, trg_words = self.get_indices_and_gid(trg_line, offset=trg_gid_offset)
    trg_indices, trg_gid, trg_words = self.get_indices_and_gid(trg_line, offset=0)


    line_indices = [self.vocab.bos()] + src_indices + [self.vocab.eos()] + trg_indices + [self.vocab.eos()]
    token_group_id = [-1] + src_gid + [-1] + trg_gid + [-1]
    assert len(line_indices) == len(token_group_id)

    src_fr = 1
    src_to = src_fr + len(src_indices)
    trg_fr = src_to + 1
    trg_to = trg_fr + len(trg_indices)
    assert trg_to + 1 == len(line_indices)

    # return torch.LongTensor(line_indices), src_fr, src_to, trg_fr, trg_to
    return {
      "tensor": torch.LongTensor(line_indices),
      "token_group": token_group_id,
      "src_words": src_words,
      "trg_words": trg_words,
      "offsets": (src_fr, src_to, trg_fr, trg_to),
    }

  def __len__(self):
    return len(self.src_lines)

  def collater(self, samples):
    if len(samples) == 0:
      return {}
    src_fr = [s["offsets"][0] for s in samples]
    src_to = [s["offsets"][1] for s in samples]
    trg_fr = [s["offsets"][2] for s in samples]
    trg_to = [s["offsets"][3] for s in samples]

    tensor_samples = [s["tensor"] for s in samples]
    token_groups = [s["token_group"] for s in samples]
    src_words = [s["src_words"] for s in samples]
    trg_words = [s["trg_words"] for s in samples]

    pad_idx = self.vocab.pad()
    eos_idx = self.vocab.eos()
    
    tokens = data_utils.collate_tokens(tensor_samples, pad_idx, eos_idx)
    lengths = torch.LongTensor([s.numel() for s in tensor_samples])
    ntokens = sum(len(s) for s in tensor_samples)


    batch = {
      'nsentences': len(samples),
      'ntokens': ntokens,
      'net_input': {
        'src_tokens': tokens,
        'src_lengths': lengths,
      },
      'token_groups': token_groups,
      "src_words": src_words,
      "trg_words": trg_words,
      "offsets": (src_fr, src_to, trg_fr, trg_to),
    }

    return batch


def load_model(args):
  args.model_type = args.model_type.lower()
  config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
  config = config_class.from_pretrained(
    args.config_name if args.config_name else args.model_name_or_path,
    cache_dir=args.cache_dir if args.cache_dir else None,
  )
    
  config.output_hidden_states = True
  config.return_dict = False

  tokenizer = tokenizer_class.from_pretrained(
    args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    do_lower_case=args.do_lower_case,
    cache_dir=args.cache_dir if args.cache_dir else None,
  )
  state_dict = None
  model = model_class.from_pretrained(
    args.model_name_or_path,
    from_tf=bool(".ckpt" in args.model_name_or_path),
    config=config,
    state_dict=state_dict,
  )
  return config, tokenizer, model


def get_dataset(args, vocab):
  with open(args.src_fn) as fp: src_lines = [l for l in fp]
  with open(args.trg_fn) as fp: trg_lines = [l for l in fp]
  dataset = RawPair2WordAlignDataset(src_lines, trg_lines, args.bpe_path, vocab)
  return dataset


def extract_wa_from_pi_xi(pi, xi):
  m, n = pi.size()
  forward = torch.eye(n)[pi.argmax(dim=1)]
  backward = torch.eye(m)[xi.argmax(dim=0)]
  inter = forward * backward.transpose(0, 1)
  ret = []
  for i in range(m):
    for j in range(n):
      if inter[i, j].item() > 0:
        ret.append((i, j))
  return ret


def _sinkhorn_iter(S, num_iter=2):
  if num_iter <= 0:
    return S, S
  # assert num_iter >= 1
  assert S.dim() == 2
  # S[S <= 0] = 1e-6
  S[S<=0].fill_(1e-6)
  # pi = torch.exp(S*10.0)
  pi = S
  xi = pi
  for i in range(num_iter):
    pi_sum_over_i = pi.sum(dim=0, keepdim=True)
    xi = pi / pi_sum_over_i
    xi_sum_over_j = xi.sum(dim=1, keepdim=True)
    pi = xi / xi_sum_over_j
  return pi, xi


def sinkhorn(sample, batch_sim, src_fr, src_to, trg_fr, trg_to, num_iter=2):
  pred_wa = []
  for i, sim in enumerate(batch_sim):
    sim_wo_offset = sim[src_fr[i]: src_to[i], trg_fr[i]: trg_to[i]]
    if src_to[i] - src_fr[i] <= 0 or trg_to[i] - trg_fr[i] <= 0:
      print("[W] src or trg len=0")
      pred_wa.append([])
      continue
    pi, xi = _sinkhorn_iter(sim_wo_offset, num_iter)
    pred_wa_i_wo_offset = extract_wa_from_pi_xi(pi, xi)
    pred_wa_i = []
    for src_idx, trg_idx in pred_wa_i_wo_offset:
      pred_wa_i.append((src_idx + src_fr[i], trg_idx + trg_fr[i]))
    pred_wa.append(pred_wa_i)
  
  return pred_wa


def convert_batch_fairseq2hf(net_input):
  tokens = net_input["src_tokens"]
  lengths = net_input["src_lengths"]
  _, max_len = tokens.size()
  device = tokens.device
  attention_mask = (torch.arange(max_len)[None, :].to(device) < lengths[:, None]).float()
  return tokens, attention_mask


def get_wa(args, model, sample):
  model.eval()
  with torch.no_grad():
    tokens, attention_mask = convert_batch_fairseq2hf(sample['net_input'])
  last_layer_outputs, first_token_outputs, all_layer_outputs = model(input_ids=tokens, attention_mask=attention_mask)
  wa_features = all_layer_outputs[args.wa_layer]
  rep = wa_features

  src_fr, src_to, trg_fr, trg_to = sample["offsets"]
  batch_sim = torch.bmm(rep, rep.transpose(1,2))
  wa = sinkhorn(sample, batch_sim, src_fr, src_to, trg_fr, trg_to, num_iter=args.sinkhorn_iter)
  ret_wa = []
  for i, wa_i in enumerate(wa):
    gid = sample["token_groups"][i]
    ret_wa_i = set()
    for a in wa_i:
      ret_wa_i.add((gid[a[0]] + 1, gid[a[1]] + 1))
    ret_wa.append(ret_wa_i)
  return ret_wa


def run(args):
  args.tokens_per_sample = 512
  vocab = Dictionary.load(args.vocab_path)

  config, tokenizer, model = load_model(args)
  if args.device == torch.device("cuda"):
    model.cuda()

  for lang_pair in args.lang_pairs.split(","):

    src_lang, trg_lang = lang_pair.split("-")
    test_set_dir = os.path.join(args.test_set_dir, lang_pair)
    args.src_fn = os.path.join(test_set_dir, "test.%s" % src_lang)
    args.trg_fn = os.path.join(test_set_dir, "test.%s" % trg_lang)
    args.gold_align = os.path.join(test_set_dir, "alignv2.txt")
    gold_align = load_gold_alignments(args)
    dataset = get_dataset(args, vocab)
    dl = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=dataset.collater)

    print("Language pair: %s" % lang_pair)
    all_results = {key:[] for key in ["p", "r", "f1", "aer"]}
    for wa_layer in range(13):
      args.wa_layer = wa_layer
      result_wa = []
      for sample in dl:
        sample = move_to_device(sample, args.device)
        batch_wa = get_wa(args, model, sample)
        # batch_wa = get_wa_v2(args, model, sample)
        result_wa.extend(batch_wa)
      p, r, f1, aer = eval_results(gold_align, result_wa)
      all_results["p"].append(p)
      all_results["r"].append(r)
      all_results["f1"].append(f1)
      all_results["aer"].append(aer)
    for key, value in all_results.items():
      print("%s: %s" % (key, ", ".join(["%.2f" % i for i in value])))
  

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--bpe_path', default="", type=str)
  parser.add_argument("--test_set_dir", type=str, default="")
  parser.add_argument("--lang_pairs", type=str, default="de-en,en-fr,en-hi,ro-en")
  parser.add_argument("--wa_layer", type=int, default=8)
  parser.add_argument("--sinkhorn_iter", type=int, default=2)
  parser.add_argument('--vocab_path', default="", type=str)
  parser.add_argument(
    "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
  )
  parser.add_argument(
    "--tokenizer_name",
    default="",
    type=str,
    help="Pretrained tokenizer name or path if not the same as model_name",
  )
  parser.add_argument(
    "--cache_dir",
    default="",
    type=str,
    help="Where do you want to store the pre-trained models downloaded from s3",
  )
  parser.add_argument("--model_type", default=None, type=str, required=True)
  parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                      help="Path to pre-trained model or shortcut name selected in the list:")
  parser.add_argument("--do_lower_case", action='store_true',
                      help="Set this flag if you are using an uncased model.")
  args = parser.parse_args()
  args.device = torch.device("cuda")
  run(args)
