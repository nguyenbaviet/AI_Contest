from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json

app = FastAPI(debug=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/database", StaticFiles(directory="database"), name="database")

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def get_hello(request: Request):
  # return templates.TemplateResponse("index.html", {"request": request, "id": "chao cu"})
  return templates.TemplateResponse("main.html", {"request": request})

@app.post("/save/{item}")
async def save_json(item: str):
  with open('database/db.json', 'w') as f:
    json.dump({'voice_id': item}, f)
  return item

@app.get("/get_voice_id")
async def get_voice_id():
  with open('database/db.json') as f:
    d = json.load(f)
  return d['voice_id']

@app.get("/enroll")
async def get_hello(request: Request):
  # return templates.TemplateResponse("index.html", {"request": request, "id": "chao cu"})
  return templates.TemplateResponse("main.html", {"request": request})
