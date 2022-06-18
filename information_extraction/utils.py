from hashlib import new
import re
from fuzzywuzzy import fuzz
import difflib
from nltk.util import ngrams
import numpy as np
import pandas as pd
from unidecode import unidecode
import pandas as pd
import json


def bank_name_from_json(json_file):
  bank_name_references = {}
  with open(json_file, 'r') as f:
    bank_info = json.loads(f.read())['napasBanks']
    for item in bank_info:
      bank_name = item['bankName']
      bank_name_search = item['bankNameSearch'].split(',')

      for b in bank_name_search:
        # convert bank name to lower and no vietnamese accent
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
          p = p.strip()
          p = '84' + p[1:]
          phone_references.append(p)

      phone_codes = item['phoneCode'].split(',')
      for p in phone_codes:
        phone_references.append(p.strip())
        phone_references.append(p.strip().replace('0', '84'))

  return phone_references


def name_from_json(json_file):
  with open(json_file, 'r') as f:
    name_references = json.loads(f.read())

  return name_references


def convert_to_isnumber(text):
  isnumber_seq = ''
  template = ''
  new_text = ''
  for c in text:

    if c.isnumeric():
      isnumber_seq += 'T'
      template += 'T'
      new_text += c
    elif c.isalpha():
      isnumber_seq += 'F'
      template += 'T'
      new_text += c
    elif c == ' ' or '.':
      continue
    else:
      template += 'T'
      isnumber_seq += '.'
      new_text += c

  return isnumber_seq, template, new_text


def get_best_match(x, y):
  if len(x) > len(y):
    corpus = x
    query = y
  else:
    corpus = y
    query = x

  ngs = ngrams(list(corpus), len(query))
  ngrams_text = [''.join(x) for x in ngs]
  return difflib.get_close_matches(query, ngrams_text, n=1, cutoff=0)


def get_bank_name_entities_by_regex(item_index, item, bank_name_references):
  """

  Args:
    item_index:
    item:
    bank_name_references:

  Returns:

  """
  org_text = item['org_text']
  text = unidecode(org_text.lower())
  # convert "thuong mai co phan" -> "TMCP"
  text = text.replace('thuong mai co phan', 'tmcp')
  bank_name_entities = []

  for bank_name in bank_name_references:
    if len(bank_name) == 0:
      continue

    # compute similarity between bank entity and bank name in references
    tmp_bank_name = bank_name.lower().replace('bank', '').strip()
    pattern = '\s' + tmp_bank_name

    for match in re.finditer(pattern, text):
      index_in_line = match.start()
      bank_name_entity = {'similarity': 1, 'bank_name': bank_name, 'text_line_index': item_index, 'index_in_line': index_in_line}
      bank_name_entities.append(bank_name_entity)

  return bank_name_entities


def get_bank_name_entities_in_single_line(item_index, item, bank_name_references):
  """

  Args:
    item_index:
    item:
    bank_name_references:

  Returns:

  """
  indices = [i for i, x in enumerate(item['tag']) if x == 0]
  groups = list(group(indices))
  text_elems = item['text'].split()

  entities = []
  bank_name_entities = []

  for gr in groups:
    first, last = gr
    gr_indices = np.arange(first, last + 1)
    entity = ''
    for index in gr_indices:
      entity += ' ' + text_elems[index]
    entities.append(entity.strip())

  for idx, entity in enumerate(entities):
    bank_name_entity = {'similarity': 0, 'bank_name': '', 'text_line_index': item_index, 'index_in_line': None}

    # compare bank entity in text with bank name references
    for bank_name in bank_name_references:
      if len(bank_name) == 0:
        continue

      tmp_bank_name = bank_name.lower().replace('bank', '').strip()

      if len(tmp_bank_name) > len(entity):
        continue

      # search pattern
      tmp_bank_name = ' ' + tmp_bank_name
      entity = ' ' + entity

      similarity = fuzz.partial_ratio(entity.lower(), tmp_bank_name.lower())
      scale_similarity = (similarity / 100)

      if scale_similarity > bank_name_entity['similarity'] and similarity > 70:
        bank_name_entity['bank_name'] = bank_name
        bank_name_entity['similarity'] = scale_similarity
        bank_name_entity['index_in_line'] = groups[idx][0]

      elif bank_name_entity['similarity'] != 0 and scale_similarity == bank_name_entity['similarity'] and \
        len(tmp_bank_name) > len(bank_name_entity['bank_name'].lower().replace('bank', '').strip()):
        bank_name_entity['bank_name'] = bank_name
        bank_name_entity['similarity'] = scale_similarity
        bank_name_entity['index_in_line'] = groups[idx][0]

    if bank_name_entity['bank_name'] != '':
      bank_name_entities.append(bank_name_entity)

  return bank_name_entities


def detect_bank_name(texts: dict, bank_name_references):
  """
  Detect bank name in all text lines
  """
  chosen_text_line_idx = None
  bank_name_entities = {}
  entity_idx = 0

  for idx, item in texts.items():
    entities_1 = get_bank_name_entities_by_regex(idx, item, bank_name_references)
    if 0 in item['tag']:
      entities_2 = get_bank_name_entities_in_single_line(idx, item, bank_name_references)
    else:
      entities_2 = []

    if len(entities_1) != 0:
      for entity in entities_1:
        bank_name_entities[entity_idx] = entity
        entity_idx += 1

    if len(entities_2) != 0:
      for entity in entities_2:
        bank_name_entities[entity_idx] = entity
        entity_idx += 1

  if len(bank_name_entities) == 0:
    return None, '', 0, bank_name_entities

  elif len(bank_name_entities) == 1:
    chosen_entity_idx = list(bank_name_entities.keys())[0]
    chosen_text_line_idx = bank_name_entities[chosen_entity_idx]['text_line_index']
    return chosen_text_line_idx, bank_name_entities[chosen_entity_idx]['bank_name'], bank_name_entities[chosen_entity_idx]['similarity'], bank_name_entities

  elif len(bank_name_entities) > 1:
    similarities = []
    entity_idxs = list(bank_name_entities.keys())

    for entity_id in entity_idxs:
      similarities.append(bank_name_entities[entity_id]['similarity'])

    # choose the best simmilarity
    chosen_entity_idx = np.argmax(similarities)
    chosen_sim_value = similarities[chosen_entity_idx]

    if similarities.count(chosen_sim_value) == 1:
      chosen_text_line_idx = bank_name_entities[chosen_entity_idx]['text_line_index']
    else:
      max_leng = 0
      for entity_idx, sim_value in enumerate(similarities):
        bank_name = bank_name_entities[entity_idx]['bank_name']
        bank_name = bank_name.replace('bank', '').strip()
        if sim_value == chosen_sim_value and len(bank_name) > max_leng:
          max_leng = len(bank_name)
          chosen_entity_idx = entity_idx
      chosen_text_line_idx = bank_name_entities[chosen_entity_idx]['text_line_index']

    return chosen_text_line_idx, bank_name_entities[chosen_entity_idx]['bank_name'], bank_name_entities[chosen_entity_idx]['similarity'], bank_name_entities


def detect_phone_number(text, phone_references):
  """
  Verify the block of tokens is phone number or not
  """
  phone_number = ''
  for ref in phone_references:
    leng = len(ref)
    if text[:leng] == ref:
      if text[:2] == '84':
        if (len(text) == 11 or len(text) == 12):
          phone_number = '0' + text[2:]
      else:
        if len(text) == 10 or len(text) == 11:
          phone_number = text

  return phone_number


def group(indices):
  first = last = indices[0]

  for index in indices[1:]:
    if index - 1 == last:
      last = index
    else:
      yield first, last
      first = last = index

  yield first, last


def merge_group(item, groups):
  """
  Merge digit fields in a line

  Args:
      groups (_type_): _description_
      pattern (_type_): regex pattern

  Yields:
      _type_: _description_
  """
  tag_seq = item['tag']
  # convert tag to text
  tag_text = ''
  for tag in tag_seq:
    if tag == 'unknown':
      tag = '*'
    tag_text += str(tag)

  patterns = ['311131111111',
              '3111311111111', # pattern for ios phone number
              ]

# find pattern
  pattern_groups = []
  for pattern in patterns:
    for match in re.finditer(pattern, tag_text):
      start_index = match.start()
      end_index = match.end()
      pattern_groups.append((start_index, end_index))

  # merge
  if len(pattern_groups) != 0:
    merge_groups = []
    for gr_idx, gr in enumerate(groups):
      for p_gr_idx, p_gr in enumerate(pattern_groups):
        set_1 = np.arange(gr[0], gr[1] + 1)
        set_2 = np.arange(p_gr[0], p_gr[1] + 1)
        intersect = np.intersect1d(set_1, set_2)
        if len(intersect) == 0:
          merge_groups.append(None)
          continue

        if all(intersect == set_1):
          merge_groups.append(p_gr_idx)
        else:
          merge_groups.append(None)

    first, last = groups[0]
    group = merge_groups[0]
    idx = 1
    for gr in groups[1:]:
      if group == merge_groups[idx]:
        last = gr[1]
        idx += 1
      else:
        yield first, last
        # reset first last
        first, last = gr
        group = merge_groups[idx]
        idx += 1
        
    yield first, last

  else:
    for gr in groups:
      yield gr


def get_bank_acc_entities_in_single_line(item_index, item):
  # get indices of entities
  indices = [i for i, x in enumerate(item['tag']) if x == 1]
  groups = list(group(indices))

  indices = []
  for gr in groups:
    first, last = gr
    indices.extend(np.arange(first, last + 1))

  merge_groups = list(merge_group(item, groups))

  merge_indices = []
  for gr in merge_groups:
    tmp_indices = []
    first, last = gr
    if last - first < 5:
      continue

    for i in np.arange(first, last + 1):
      if i in indices:
        tmp_indices.append(i)

    merge_indices.append(tmp_indices)

  # get text of entities
  text = item['text']
  text_elems = text.split()
  entities = []

  for gr_indices in merge_indices:
    entity = ''
    for index in gr_indices:
      entity += ' ' + text_elems[index]
    entities.append(entity.strip())

  # get bank id entities
  bank_id_entities = []

  for idx, entity in enumerate(entities):
    bank_id_entity = {'similarity': 0, 'bank_id': '', 'text_line_index': item_index, 'index_in_line': None}

    isnumber_seq, template, new_text = convert_to_isnumber(entity)
    similarity = fuzz.ratio(isnumber_seq, template)

    if similarity > 70:
      bank_id_entity['similarity'] = similarity
      bank_id_entity['bank_id'] = new_text
      bank_id_entity['index_in_line'] = groups[idx][0]

      bank_id_entities.append(bank_id_entity)

  return bank_id_entities


def detect_bank_acc_value(texts, phone_references, bank_text_idx=None, bank_name=None):
  """Detect Bank ID

  """
  chosen_bank_id_text_idx = None
  chosen_phone_text_idx = None
  bank_id = ''
  bank_id_score = 0
  phone_number = ''
  phone_number_score = 0

  bank_id_entities = {}
  entity_idx = 0

  for idx, item in texts.items():
    if 1 not in item['tag']:
      continue

    indices = [i for i, x in enumerate(item['tag']) if x == 1]
    if len(indices) < 6:
      continue

    entities = get_bank_acc_entities_in_single_line(idx, item)
    for entity in entities:
      bank_id_entities[entity_idx] = entity
      entity_idx += 1

  if len(bank_id_entities) == 1:
    chosen_bank_id_entity_idx = list(bank_id_entities.keys())[0]
    _bank_id = bank_id_entities[chosen_bank_id_entity_idx]['bank_id']
    _phone_number = detect_phone_number(_bank_id, phone_references)
    if _phone_number != '':
      chosen_phone_entity_idx = chosen_bank_id_entity_idx
      phone_number = _phone_number
      phone_number_score = bank_id_entities[chosen_phone_entity_idx]['similarity'] / 100
      chosen_phone_text_idx = bank_id_entities[chosen_phone_entity_idx]['text_line_index']

    bank_id = bank_id_entities[chosen_bank_id_entity_idx]['bank_id']
    bank_id_score = bank_id_entities[chosen_bank_id_entity_idx]['similarity']
    chosen_bank_id_text_idx = bank_id_entities[chosen_bank_id_entity_idx]['text_line_index']

  elif len(bank_id_entities) > 1:
    similarities = []

    # keys of bank_id_entities
    entity_idxs = list(bank_id_entities.keys())
    for entity_idx in entity_idxs:
      _bank_id = bank_id_entities[entity_idx]['bank_id']
      _phone_number = detect_phone_number(_bank_id, phone_references)
      if _phone_number != '':
        chosen_phone_entity_idx = entity_idx
        phone_number = _phone_number
        phone_number_score = bank_id_entities[chosen_phone_entity_idx]['similarity'] / 100
        chosen_phone_text_idx = bank_id_entities[chosen_phone_entity_idx]['text_line_index']

      # if we have bank name, we compare the distance between 2 indexes of 2 boxes
      if bank_text_idx is not None:
        bank_id_idx = bank_id_entities[entity_idx]['text_line_index']
        similarities.append(1 / abs(bank_text_idx - bank_id_idx + 1e-2))
      else:
        similarities.append(bank_id_entities[entity_idx]['similarity'])

    # index for best similarity
    best_idx = np.argmax(similarities)
    # entity index
    chosen_bank_id_entity_idx = entity_idxs[best_idx]

    # if we have more bank id entitites and phone number is detected
    if phone_number.replace('+', '') == bank_id_entities[chosen_bank_id_entity_idx]['bank_id']:
      similarities.pop(best_idx)
      entity_idxs.pop(chosen_bank_id_entity_idx)

      best_idx = np.argmax(similarities)
      chosen_bank_id_entity_idx = entity_idxs[best_idx]

    bank_id = bank_id_entities[chosen_bank_id_entity_idx]['bank_id']
    bank_id_score = bank_id_entities[chosen_bank_id_entity_idx]['similarity']
    chosen_bank_id_text_idx = bank_id_entities[chosen_bank_id_entity_idx]['text_line_index']

  return chosen_bank_id_text_idx, bank_id, bank_id_score, chosen_phone_text_idx, \
         phone_number, phone_number_score, bank_id_entities


def get_name_entities_in_single_line(item_index, item):
  # get indices of entities
  indices = [i for i, x in enumerate(item['tag']) if x == 2]
  groups = list(group(indices))
  text_elems = item['viet_text'].split()

  entities = []
  name_entities = []

  for gr in groups:
    first, last = gr
    gr_indices = np.arange(first, last + 1)
    entity = ''

    for index in gr_indices:
      entity += ' ' + text_elems[index]

    entities.append(entity.strip())

  for idx, entity in enumerate(entities):
    name_entity = {'similarity': 0, 'name': '', 'text_line_index': item_index, 'index_in_line': None}
    name_entity['name'] = entity
    name_entity['index_in_line'] = groups[idx][0]
    name_entities.append(name_entity)

  return name_entities


def check_name(name, name_references):
  score = 0
  name_elems = name.split()
  if len(name_elems) < 2:
    return False

  for elem in name_elems:
    if elem in name_references:
      score += 1
  score = score / len(name_elems)

  if score > 0.6:
    return True
  else:
    return False


def detect_name_value(texts, name_references):
  chosen_text_line_idx = None
  name_entities = {}
  entity_idx = 0

  for idx, item in texts.items():
    if 2 not in item['tag']:
      continue

    indices = [i for i, x in enumerate(item['tag']) if x == 2]
    if len(indices) < 2:
      continue

    entities = get_name_entities_in_single_line(idx, item)

    if len(entities) != 0:
      for entity in entities:
        name_entities[entity_idx] = entity
        entity_idx += 1

  if len(name_entities) == 0:
    return chosen_text_line_idx, ''

  elif len(name_entities) == 1:
    chosen_entity_idx = list(name_entities.keys())[0]
    name = name_entities[chosen_entity_idx]['name']
    if len(name.split()) <= 4 and check_name(name, name_references):
      chosen_text_line_idx = name_entities[chosen_entity_idx]['text_line_index']
      return chosen_text_line_idx, name_entities[chosen_entity_idx]['name']
    else:
      return chosen_text_line_idx, ''

  elif len(name_entities) > 1:
    entity_idxs = list(name_entities.keys())
    for entity_id in entity_idxs:
      name = name_entities[entity_id]['name']
      if len(name.split()) <= 4 and check_name(name, name_references):
        chosen_entity_idx = entity_id
        chosen_text_line_idx = name_entities[chosen_entity_idx]['text_line_index']

    if chosen_text_line_idx is not None:
      return chosen_text_line_idx, name_entities[chosen_entity_idx]['name']
    else:
      return chosen_text_line_idx, ''


def extract_info(texts: dict, bank_references, phone_references, name_references):
  # init
  org_bank_name = ''
  org_bank_acc = ''
  org_phone_number = ''
  org_name = ''

  bank_name = ''
  bank_acc = ''
  phone_number = ''
  name = ''

  bank_name_references = list(bank_references.keys())

  bank_name_idx, bank_name, bank_name_score, bank_entities = detect_bank_name(texts, bank_name_references)

  if bank_name_idx is not None:
    org_bank_name = texts[bank_name_idx]['org_text']
  # if len(bank_name) != 0:
  #   bank_name_score = bank_name_score / len(bank_name)
  if bank_name != '':
    bank_name = bank_references[bank_name]
  print('Detected bank name is ', org_bank_name, bank_name, bank_name_score)

  bank_id_idx, bank_acc, bank_acc_score, \
  phone_idx, phone_number, phone_score, bank_acc_enitities = detect_bank_acc_value(texts, phone_references, bank_name_idx)
  bank_acc_score = bank_acc_score / 100
  if bank_id_idx is not None:
    org_bank_acc = texts[bank_id_idx]['org_text']
    if '+84' in org_bank_acc.replace(' ', '') and bank_acc[:2] == '84':
      bank_acc = '0' + bank_acc[2:]
  print('Detected acc number is ', org_bank_acc, bank_acc, bank_acc_score)

  if phone_idx is not None:
    org_phone_number = texts[phone_idx]['org_text']
  print('Detected phone number is ', org_phone_number, phone_number, phone_score)

  name_idx, name = detect_name_value(texts, name_references)
  name = name.upper()
  if name_idx is not None:
    org_name = texts[name_idx]['org_text']
  print('Detected name is ', org_name, name)

  total_score = bank_name_score * bank_acc_score
  print('Confidence score is ', total_score)

  return org_bank_name, org_bank_acc, org_phone_number, org_name, \
         bank_name, bank_acc, phone_number, name, total_score
