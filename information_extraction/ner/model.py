import torch
import torch.autograd as autograd
import torch.nn as nn
import os
import json
from fuzzywuzzy import fuzz
from unidecode import unidecode
import re
import string


torch.manual_seed(1)
START_TAG = "<START>"
STOP_TAG = "<STOP>"


def create_vocab(vocab_file):
  with open(vocab_file, 'r') as f:
    word_to_ix = json.load(f)

  return word_to_ix


def norm_dict_from_json(norm_file):
  with open(norm_file, 'r') as f:
    norm_dict = json.load(f)

  return norm_dict


def argmax(vec):
  # return the argmax as a python int
  _, idx = torch.max(vec, 1)
  return idx.item()


def log_sum_exp(vec):
  max_score = vec[0, argmax(vec)]
  max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
  return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF(nn.Module):
  def __init__(self, vocab_size: int, tag_to_ix, embedding_dim: int,
    hidden_dim: int):
    super(BiLSTM_CRF, self).__init__()
    self.embedding_dim = embedding_dim
    self.hidden_dim = hidden_dim
    self.vocab_size = vocab_size
    self.tag_to_ix = tag_to_ix
    self.tagset_size = len(tag_to_ix)

    self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
    self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1,
                        bidirectional=True)

    # maps the output of the LSTM into tag space.
    self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

    # matrix of transition parameters. Entry i,j is the score of
    # transitioning *to* i *from* j.
    self.transitions = nn.Parameter(
      torch.randn(self.tagset_size, self.tagset_size))

    # these two statements enforce the constraint that we never transfer
    # to the start tag and we never transfer from the stop tag
    self.transitions.data[tag_to_ix[START_TAG], :] = -10000
    self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

    self.hidden = self.init_hidden()

  def init_hidden(self):
    # return (torch.randn(2, 1, self.hidden_dim // 2),
    #         torch.randn(2, 1, self.hidden_dim // 2))

    return (torch.zeros(2, 1, self.hidden_dim // 2),
            torch.zeros(2, 1, self.hidden_dim // 2))

  def _forward_alg(self, feats):
    init_alphas = torch.full((1, self.tagset_size), -10000.)
    # START_TAG has all of the score.
    init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

    # wrap in a variable so that we will get automatic backprop
    forward_var = init_alphas

    # iterate through the sentence
    for feat in feats:
      alphas_t = []  # the forward tensors at this timestep
      for next_tag in range(self.tagset_size):
        # broadcast the emission score: it is the same regardless of
        # the previous tag
        emit_score = feat[next_tag].view(
          1, -1).expand(1, self.tagset_size)
        # the ith entry of trans_score is the score of transitioning to
        # next_tag from i
        trans_score = self.transitions[next_tag].view(1, -1)
        # the ith entry of next_tag_var is the value for the
        # edge (i -> next_tag) before we do log-sum-exp
        next_tag_var = forward_var + trans_score + emit_score
        # the forward variable for this tag is log-sum-exp of all the
        # scores.
        alphas_t.append(log_sum_exp(next_tag_var).view(1))
      forward_var = torch.cat(alphas_t).view(1, -1)
    terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
    alpha = log_sum_exp(terminal_var)
    return alpha

  def _get_lstm_features(self, sentence):
    self.hidden = self.init_hidden()
    embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
    lstm_out, self.hidden = self.lstm(embeds, self.hidden)
    lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
    lstm_feats = self.hidden2tag(lstm_out)
    return lstm_feats

  def _score_sentence(self, feats, tags):
    # gives the score of a provided tag sequence
    score = torch.zeros(1)
    tags = torch.cat(
      [torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])

    for i, feat in enumerate(feats):
      score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
    score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
    return score

  def _viterbi_decode(self, feats):
    backpointers = []

    # initialize the viterbi variables in log space
    init_vvars = torch.full((1, self.tagset_size), -10000.)
    init_vvars[0][self.tag_to_ix[START_TAG]] = 0

    # forward_var at step i holds the viterbi variables for step i-1
    forward_var = init_vvars
    for feat in feats:
      bptrs_t = []  # holds the backpointers for this step
      viterbivars_t = []  # holds the viterbi variables for this step

      for next_tag in range(self.tagset_size):
        # next_tag_var[i] holds the viterbi variable for tag i at the
        # previous step, plus the score of transitioning
        # from tag i to next_tag.
        # we don't include the emission scores here because the max
        # does not depend on them (we add them in below)
        next_tag_var = forward_var + self.transitions[next_tag]
        best_tag_id = argmax(next_tag_var)
        bptrs_t.append(best_tag_id)
        viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
      # now add in the emission scores, and assign forward_var to the set
      # of viterbi variables we just computed
      forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
      backpointers.append(bptrs_t)

    # transition to STOP_TAG
    terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
    best_tag_id = argmax(terminal_var)
    path_score = terminal_var[0][best_tag_id]

    # follow the back pointers to decode the best path
    best_path = [best_tag_id]
    for bptrs_t in reversed(backpointers):
      best_tag_id = bptrs_t[best_tag_id]
      best_path.append(best_tag_id)
    # pop off the start tag (we dont want to return that to the caller)
    start = best_path.pop()
    assert start == self.tag_to_ix[START_TAG]  # Sanity check
    best_path.reverse()
    return path_score, best_path

  def neg_log_likelihood(self, sentence, tags):
    feats = self._get_lstm_features(sentence)
    forward_score = self._forward_alg(feats)
    gold_score = self._score_sentence(feats, tags)
    return forward_score - gold_score

  def forward(self, sentence):  # dont confuse this with _forward_alg above.
    # get the emission scores from the BiLSTM
    lstm_feats = self._get_lstm_features(sentence)
    # print('Extract feature from LSTM {}'.format(lstm_feats.size()))

    # find the best path, given the features
    score, tag_seq = self._viterbi_decode(lstm_feats)
    return score, tag_seq


def remove_special_tokens(text: str):
  # . and - are often used to in account number. ex: 190.1231.1231
  text = text.replace('.', '')
  text = text.replace('-', '')

  text = re.sub(f"[{string.punctuation}\s\t]", ' ', text)
  output_text = ''

  for token in text.split():
    output_text += token + ' '
  output_text = output_text.strip()

  return output_text


def add_start_of_num(text: str):
  """
  Add a token as the begin of numeric tokens
  Ex: 0945022712 0945022700 -> split 0945022712 split 0945022700
  """
  output_tokens = []
  input_tokens = text.split()
  for token_id in range(len(input_tokens)):
    token = input_tokens[token_id]
    if token[0].isnumeric():
      output_tokens.append('split')
    output_tokens.append(token)

  text = ''
  for token in output_tokens:
    text += token + ' '
  text = text.strip()

  return text


def split_num_tokens(text):
  """
  Split the series of numbers into tokens
  Ex: 0945022712 -> 0 9 4 5 0 2 2 7 1 2
  """
  output_text = ''
  for indx, c in enumerate(text):
    if c.isnumeric():
      output_text += ' ' + c + ' '
    else:
      output_text += c

  output_text = output_text.strip()

  return output_text


def prepare_sequence(text: str, to_ix: dict):
  """
  Preprocess input text line and convert token to id for NER's format
  Args:
    text: input text line
    to_ix: dictionary {[token]:id}

  Returns:

  """
  text = remove_special_tokens(text)
  text = add_start_of_num(text)
  text = split_num_tokens(text)
  vietnamese_text = text

  # convert "thuong mai co phan" -> "TMCP"
  vietnamese_text = vietnamese_text.replace('thương mại cổ phần', 'tmcp')
  vietnamese_text = vietnamese_text.replace('thuong mai co phan', 'tmcp')

  # convert to english
  eng_text = unidecode(vietnamese_text.lower())

  # convert to NER input
  idxs = []
  ner_text = ''
  # known tag seq is the mask for the position of tokens in dictionary
  known_tag_seq = []

  seq = eng_text.split()

  for idx, w in enumerate(seq):
    # encode digit
    if w.isdigit():
      for j, number in enumerate(w):
        idxs.append(to_ix[number])
        known_tag_seq.append('known')
        if idx == 0 and j == 0:
          ner_text += number
        else:
          ner_text += ' ' + number

    # encode not digit
    else:
      # with the word that appears in word2vec
      if w in to_ix.keys():
        idxs.append(to_ix[w])
        known_tag_seq.append('known')
        if idx == 0:
          ner_text += w
        else:
          ner_text += ' ' + w
      else:
        # with the word not in word2vec
        # and important cases bank name
        if 'bank' in w.lower() or 'ank' in w.lower():
          keys = list(to_ix.keys())
          for key in keys:
            distance = fuzz.ratio(w.lower(), key.lower())
            if distance > 70:
              idxs.append(to_ix[key])
              known_tag_seq.append('known')
              if idx == 0:
                ner_text += w
              else:
                ner_text += ' ' + w
              break

          if distance <= 70:
            idxs.append(to_ix['bank'])
            known_tag_seq.append('known')
            if idx == 0:
              ner_text += 'bank'
            else:
              ner_text += ' ' + 'bank'
        else:
          known_tag_seq.append('unknown')

    ner_text = ner_text.strip()

  return torch.tensor(idxs, dtype=torch.long), \
         ner_text, \
         vietnamese_text, eng_text, \
         known_tag_seq


def process_tag(tag_seq: list, known_tag_seq: list):
  """
  Convert tag sequence of tokens that appear in dictionary
  into full tokens contain both known and unknown tokens
  Args:
    tag_seq:
    known_tag_seq:

  Returns:

  """
  known_pos = [i for i, x in enumerate(known_tag_seq) if x == 'known']
  for idx, pos in enumerate(known_pos):
    known_tag_seq[pos] = tag_seq[idx]

  return known_tag_seq


def normalize_text(text, norm_dict):
    for key in list(norm_dict.keys()):
      for i in range(2):
        if i == 0:
          map_key = key
        else:
          if not norm_dict[key].isnumeric():
            map_key = unidecode(key)
          else:
            continue

        value = norm_dict[key]
        # text = text.replace(' ' + key, value)
        patterns = ['^' + map_key + '\s', '\s' + map_key + '\s', '\s' + map_key + '$']
        for p in patterns:
          while re.search(p, text):
            text = re.sub(p, ' ' + value + ' ', text)

    # merge number
    p = r'(\d)\s(\d)'
    while re.search(p, text):
      text = re.sub(p, r'\1\2', text)

    text = re.sub(r'chuyển tiền', '', text)

    return text


class TextClassifyModel:
  def __init__(self, ner_checkpoint: str):
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    tag_to_ix = {"B": 0, "A": 1, "N": 2, "O": 3, START_TAG: 4, STOP_TAG: 5}

    EMBEDDING_DIM = 128
    HIDDEN_DIM = 128

    vocab_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              'vocab.json')
    normalize_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              'norm_dict.json')
    self.word_to_ix = create_vocab(vocab_file)
    self.norm_dict = norm_dict_from_json(normalize_file)

    self.model = BiLSTM_CRF(len(self.word_to_ix), tag_to_ix, EMBEDDING_DIM,
                            HIDDEN_DIM)
    if os.path.isfile(ner_checkpoint):
      self.model.load_state_dict(torch.load(ner_checkpoint))
    else:
      print('Cannot load checkpoint, create dummy checkpoint')
    self.model.eval()

  def __call__(self, info):
    predictions = {}

    for text_id, text in enumerate(info):
      # convert to vietnamese lowercase
      text = text.lower()
      # normalize text
      text = normalize_text(text, self.norm_dict)
      print(text)
      text_tensor, ner_text, vietnamese_text, eng_text, known_tag_seq = prepare_sequence(
        text, self.word_to_ix)

      if text_tensor.size(0) != 0:
        with torch.no_grad():
          score, tag_seq = self.model(text_tensor)
          full_tag_seq = process_tag(tag_seq, known_tag_seq)
          predictions[text_id] = {'score': score.item(), 'org_text': text,
                                  'text': eng_text, 'viet_text': vietnamese_text,
                                  'tag': full_tag_seq}
      else:
        continue

    return predictions
