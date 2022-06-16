from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(debug=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def get_hello(request: Request):
  return templates.TemplateResponse("index.html", {"request": request, "id": "chao cu"})


@app.get("/items/{id}", response_class=HTMLResponse)
async def read_item(request: Request, id: str):
  return templates.TemplateResponse("item.html", {"request": request, "id": id})
