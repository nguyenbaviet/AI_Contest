import numpy as np
from unidecode import unidecode
import torch
from fuzzywuzzy import fuzz
import json
import string
import re
import os
import matplotlib.pyplot as plt
from dataclasses import dataclass
import time

from ner.model import TextClassifyModel
from utils import extract_info


def remove_special_tokens(text):
  # . and - are often used to in account number. ex: 190.1231.1231
  text = text.replace('.', '')
  text = text.replace('-', '')

  # remove special tokens
  text = re.sub(f"[{string.punctuation}\s\t]", ' ', text)
  output_text = ''

  for token in text.split():
    output_text += token + ' '
  output_text = output_text.strip()

  return output_text


def add_start_of_num(text):
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
  output_text = ''
  for indx, c in enumerate(text):
    if c.isnumeric():
      output_text += ' ' + c + ' '
    else:
      output_text += c

  output_text = output_text.strip()

  return output_text


def create_vocab(vocab_file):
  with open(vocab_file, 'r') as f:
    word_to_ix = json.load(f)

  return word_to_ix


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


def bank_name_from_json(json_file):
  bank_name_references = {}
  with open(json_file, 'r') as f:
    bank_info = json.loads(f.read())['napasBanks']
    for item in bank_info:
      bank_name = item['bankName']
      bank_name_search = item['bankNameSearch'].split(',')

      for b in bank_name_search:
        b = unidecode(b.strip()).lower()
        bank_name_references[b] = bank_name
  
  return bank_name_references


def phone_from_json(json_file):
  phone_references = []
  with open(json_file, 'r') as f:
    phone_info = json.loads(f.read())['phones']
    for item in phone_info:
      phone_codes = item['old_phoneCode'].split(',')
      for p in phone_codes:
        if p != '':
          phone_references.append(p.strip())
          phone_references.append(p.strip().replace('0', '84'))

      phone_codes = item['phoneCode'].split(',')
      for p in phone_codes:
        phone_references.append(p.strip())
        phone_references.append(p.strip().replace('0', '84'))

  return phone_references


def name_from_json(json_file):
  with open(json_file, 'r') as f:
    name_references = json.loads(f.read())

  return name_references


def process_tag(tag_seq, known_tag_seq):
  known_pos = [i for i, x in enumerate(known_tag_seq) if x == 'known']
  for idx, pos in enumerate(known_pos):
    known_tag_seq[pos] = tag_seq[idx]
  
  return known_tag_seq


@dataclass
class BankInfo:
  phone_number: str
  bank_id: str
  account: str
  bank_name: str
  amount: str
  score: float


class BankExtractor:
  def __init__(self, ner_checkpoint):
    self.classify_model = TextClassifyModel(ner_checkpoint)
    bank_json = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             'banklist.json')
    phone_json = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             'phonelist.json')
    name_json = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             'namelist.json')

    self.bank_references = bank_name_from_json(bank_json)
    self.phone_references = phone_from_json(phone_json)
    self.name_references = name_from_json(name_json)

  def init_bank_info(self):
   bank_infor = BankInfo(phone_number='', bank_id='', account='', bank_name='', amount='', score=0)
   return bank_infor

  def __call__(self, info):
    bank_info = self.init_bank_info()

    # classify text
    predictions = self.classify_model(info)
    print(predictions)

    # post process to get bank item
    org_bank_name, org_bank_id, org_phone_number, org_name, \
    bank_name, bank_id, phone_number, name, confidence_score = extract_info(predictions,
                                                                            self.bank_references,
                                                                            self.phone_references,
                                                                            self.name_references)



if __name__ == "__main__":
  ner_model_path = '/home/thiendo/Desktop/AI_Contest/information_extraction/ner/ner_20220516.pth'
  bank_extractor = BankExtractor(ner_checkpoint=ner_model_path)
  info = ['Chuyển tiền cho Nguyễn Văn A ngan hang Techcombank không chín bốn năm không hai hai bảy một hai']
  bank_extractor(info)