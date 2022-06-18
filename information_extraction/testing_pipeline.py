from html import entities
import cv2
from cv2 import rotate
import numpy as np
import os
import glob
import easyocr
import matplotlib.pyplot as plt
import pandas as pd
from unidecode import unidecode
from information_extraction import detect_bank_name, detect_bank_acc_value, detect_name_value, extract_info

from ner.model import BiLSTM_CRF
from ner.train import BankInforDataset
import torch
from fuzzywuzzy import fuzz
import json
import pandas as pd
import string
import re


def create_vocab(vocab_file):
  with open(vocab_file, 'r') as f:
    word_to_ix = json.load(f)

  return word_to_ix


def draw_box(img, box):
    box = np.array(box, dtype=np.int)
    topleft, topright, botright, botleft = np.array(box, dtype=np.int)

    pts = box.reshape((-1, 1, 2))
    img = cv2.polylines(img, [pts], True, color=(0, 255, 0), thickness=2)
    # img = cv2.rectangle(img, tuple(topleft), tuple(botright), color=(0, 255, 0), thickness=2)
    return img


def no_accent_vietnamese(accent_fullname_df):
    no_accent_list = accent_fullname_df.tolist()
    no_accent_list = [unidecode(s) for s in no_accent_list]
    accent_fullname_df.update(pd.Series(no_accent_list, index=accent_fullname_df.index))
    return accent_fullname_df


def convert_name_2_mrz_format(fullname_df):
    fullname_df = no_accent_vietnamese(fullname_df)
    family_name = fullname_df.str.split().str[0]
    first_name = fullname_df.str.split().str[-1]
    middle_name_out = fullname_df.str.split().str[1:-1]
    middle_name = fullname_df.str.split().str[1:-1].tolist()
    processed_middle_name = []
    
    for name in middle_name:
        tmp = ''
        for index, element in enumerate(name):
            if index == 0:
                tmp += element
            else:
                tmp += '<' + element

        processed_middle_name.append(tmp)
    middle_name_out.update(pd.Series(processed_middle_name, index=middle_name_out.index))

    return family_name, middle_name_out, first_name


def get_name_dictionary(file_path):
    df = pd.read_csv(file_path)
    full_name = df['fullname_gt'].dropna()
    full_name = convert_name_2_mrz_format(full_name)
    family_name, middle_name, first_name = full_name
    # family_name = set(family_name.tolist())
    # middle_name = set(middle_name.tolist())
    # first_name = set(first_name.tolist())
    family_name = family_name.tolist()
    middle_name = middle_name.tolist()
    first_name = first_name.tolist()
    dictionary = []
    dictionary.extend(family_name)
    dictionary.extend(middle_name)
    dictionary.extend(first_name)
    return set(dictionary)


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
  

def check_rotation(img, textlines):
  avr_confidence_score = 0
  num_horizontal_line = 0

  for line in textlines:
    box, text, conf = line
    topleft, _, botright, _ = box
    box_w = botright[0] - topleft[0]
    box_h = botright[1] - topleft[1]

    if box_w > box_h:
      num_horizontal_line += 1

    avr_confidence_score += conf

  avr_confidence_score = avr_confidence_score / len(textlines)
  if num_horizontal_line / len(textlines) > 0.7:
    if avr_confidence_score < 0.8:
      rotate_img = img.copy()
      rotate_angle = [180]
      return True, rotate_img, rotate_angle
    else:
      return False, img, [0] 
  else:
    # the image is rotated with the angle {90, 270}
    rotate_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    rotate_angle = [0, 180]
    return True, rotate_img, rotate_angle


def non_max_suppression_fast(textlines, overlapThresh):
    """
    Merge box and text
    """
    boxes = []
    texts = []

    for textline in textlines:
      # box is polygon
      box, text, conf = textline
      # just get topleft and botright
      topleft, topright, botright, botleft = box
      box = np.array(box)
      x1, y1 = min(box[:, 0]), min(box[:, 1])
      x2, y2 = max(box[:, 0]), max(box[:, 1])
      boxes.append([max(x1 - 20, 0), y1, x2 + 20, y2])
      texts.append(text)

    boxes = np.array(boxes)
    # texts = np.array(texts)

    # If there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # If the bounding boxes integers, convert them to floats --
    # This is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # Initialize the list of picked indexes
    pick = []
    # Initialize the list of picked texts
    text_pick = []

    # Grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # Compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        # Grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        # pick.append(i)
        # Find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])

        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1 - 10)

        # Compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # Merge boxes
        merge_box_idxs = idxs[np.concatenate(([last], np.where(overlap > overlapThresh)[0]))]
        if len(merge_box_idxs) > 1:
          # Keep the box at last index
          keep_id = i

          # Merge texts from left to right
          merge_text_idxs = idxs[np.concatenate(([last], np.where(overlap > overlapThresh)[0]))]
          texts_2_merge = []
          for merge_text_id in merge_box_idxs:
            texts_2_merge.append(texts[merge_text_id])

          text_boxes = boxes[merge_text_idxs].astype("int")
          text_x1 = text_boxes[:, 0]

          tmp_text = ''
          for text_id in np.argsort(text_x1):
            tmp_text += texts_2_merge[text_id] + ' '
          
          texts[keep_id] = tmp_text

          text_boxes = boxes[merge_box_idxs].astype("int")
          x1, y1 = min(text_boxes[:, 0]), min(text_boxes[:, 1])
          x2, y2 = max(text_boxes[:, 2]), max(text_boxes[:, 3])
          merge_box = [x1, y1, x2, y2]

          boxes[keep_id] = merge_box

          # Delete all indexes from the index list that have
          idxs = np.delete(idxs, np.where(overlap > overlapThresh)[0])

          # Update box
          x1 = boxes[:,0]
          y1 = boxes[:,1]
          x2 = boxes[:,2]
          y2 = boxes[:,3]

          area = (x2 - x1 + 1) * (y2 - y1 + 1)

        else:
          pick.append(i)
          # Delete all indexes from the index list that have
          idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    box_pick = boxes[pick].astype("int")
    texts = np.array(texts)
    text_pick = texts[pick]

    textlines = []
    for idx, text in enumerate(text_pick):
      box = box_pick[idx]
      textlines.append([box, text, 1])

    return textlines


def process_tag(tag_seq, known_tag_seq):
  known_pos = [i for i, x in enumerate(known_tag_seq) if x == 'known']
  for idx, pos in enumerate(known_pos):
    known_tag_seq[pos] = tag_seq[idx]
  
  return known_tag_seq


def pipeline(data_dir, bank_json, phone_json, vocab_file, ner_model_path):
    reader = easyocr.Reader(lang_list=['en'], gpu=False)
    img_paths = glob.glob('{}/*.*'.format(data_dir))

    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 128
    word_to_ix = create_vocab(vocab_file)

    # {"B-BANK": 0, "B-ACC": 1, "B-NAME": 2, "O": 3, START_TAG: 4, STOP_TAG: 5}
    tag_to_ix = {"B": 0, "A": 1, "N": 2, "O": 3, START_TAG: 4, STOP_TAG: 5}
    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
    model.load_state_dict(torch.load(ner_model_path))
    model.eval()
    bank_references = bank_name_from_json(bank_json)
    phone_references = phone_from_json(phone_json)
    name_references = name_from_json('/home/thiendo/Desktop/generic_ocr_prj/dataset/name.json')

    for p in img_paths:
        import time
        str_time = time.time()
        print('Process ', p)

        img = cv2.imread(p)
        textlines = reader.readtext(img)
        if len(textlines) != 0:
          is_rotate, img, rotate_angle = check_rotation(img, textlines)
          if is_rotate:
            textlines = reader.readtext(img, rotation_info=rotate_angle)
          textlines = non_max_suppression_fast(textlines, 0.05)

        for line in textlines:
            box, text, _ = line
            print(text)
            x1, y1, x2, y2 = box
            topleft = (x1, y1)
            botright = (x2, y2)
            img = cv2.rectangle(img, tuple(topleft), tuple(botright), color=(255, 0, 0), thickness=2)

        texts = []
        boxes = []

        for line in textlines:
            box, text, _ = line
            vis_img = draw_box(img, box)
            texts.append(text)
            boxes.append(box)
        
        # texts = 'Acc chính chủ mặc dù tên như clone \nMomo 0877744176  \nSHB 2004111777 còn hơn 10 lượt chéo \nChéo uy tính 100%, chéo nh qua ib ạ'
        # texts = 'Tiếp tục chéo ngày mới nào mng \nShinhan bank \nVũ thị thanh nhàn \nTk: 700020013599 \nAi chéo chụp màn hình, nhận được mk bank lại luôn'
        # texts = 'xin chéo ạ: 0973355484 \nTPbank: 0308 0454 701 \nTất cả đều cùng tên nhé :v'
        texts = '(84) 945022711'
        texts = texts.split('\n')

        # run ner model
        predictions = {}

        for text_id, org_text in enumerate(texts):

            text = org_text.lower()
            text_tensor, ner_text, vietnamese_text, eng_text, known_tag_seq = prepare_sequence(text, word_to_ix)

            if text_tensor.size(0) != 0:
                with torch.no_grad():
                    score, tag_seq = model(text_tensor)
                    full_tag_seq = process_tag(tag_seq, known_tag_seq)
                    predictions[text_id] = {'score': score.item(), 'org_text': org_text, 
                    'text': eng_text, 'viet_text': vietnamese_text,
                    'tag': full_tag_seq}

        print(predictions)
        org_bank_name, org_bank_acc, org_phone_number, org_name, \
          bank_name, bank_acc, phone_number, name, confidence_score = extract_info(predictions, bank_references, phone_references, name_references)

        print('Processing time is ', time.time() - str_time)
        print('---------------------------------')
        plt.imshow(img)
        plt.show()


if __name__ == "__main__":
    csv_file = '/home/thiendo/Desktop/generic_ocr_prj/dataset/post_processing_gt.csv'
    data_dir = '//home/thiendo/Desktop/generic_ocr_prj/dataset/test_easy_ocr_1'
    bank_json = '/home/thiendo/Desktop/generic_ocr_prj/dataset/banklist.json'
    phone_json = '/home/thiendo/Desktop/generic_ocr_prj/dataset/phonelist.json'
    vocab_file = '/home/thiendo/Desktop/generic_ocr_prj/dataset/vocab.json'
    ner_model_path = '/home/thiendo/Desktop/generic_ocr_prj/ner/ner_20220516.pth'

    pipeline(data_dir, bank_json, phone_json, vocab_file, ner_model_path)