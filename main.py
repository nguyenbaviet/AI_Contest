from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from information_extraction.extractor import BankExtractor
import json
from time import time

app = FastAPI(debug=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/database", StaticFiles(directory="database"), name="database")

information_extractor = BankExtractor('information_extraction/weight/ner_20220516.pth')

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def get_hello(request: Request):
  return templates.TemplateResponse("main.html", {"request": request})

@app.post("/save/{item}")
async def save_json(item: str):
  with open('database/db.json') as f:
    db = json.load(f)
  db['voice_id'] = item
  with open('database/db.json', 'w') as f:
    json.dump(db, f)
  return item

@app.get("/save_token/{token}")
async def save_token(token: str):
  with open('database/db.json') as f:
    db = json.load(f)

  db['token_id'] = token
  db['timestamp'] = time()
  with open('database/db.json', 'w') as f:
    json.dump(db, f)
  return token

@app.get("/get_voice_id")
async def get_voice_id():
  with open('database/db.json') as f:
    d = json.load(f)
  is_valid = True
  if (time() - d['timestamp']) > 20 * 60 * 60:
    is_valid = False
  d['is_valid'] = is_valid
  return d

@app.get("/extraction/{text}")
async def info_extraction(text: str):
  info = information_extractor([text])
  return info


if __name__ == "__main__":
  ner_model_path = '/home/thiendo/Desktop/AI_Contest/information_extraction/ner/ner_20220516.pth'
  bank_extractor = BankExtractor(ner_checkpoint=ner_model_path)
  info = ['ngân hàng ngoại thương việt nam một chín không một chín không ba tám sáu']
  bank_extractor(info)