import mesop as me
import mesop.labs as mel
from pydantic import BaseModel
import requests as rq

class UserInput(BaseModel):
    prompt: str

def on_load(e: me.LoadEvent):
  me.set_theme_mode("system")

@me.page(
  security_policy=me.SecurityPolicy(
    allowed_iframe_parents=["https://google.github.io", "https://huggingface.co"]
  ),
  path="/",
  title="RAGcoon - The resources you need for your Startup",
  on_load=on_load,
)
def page():
  mel.chat(transform, title="RAGcoon - The resources you need for your Startup", bot_user="RAGcoon")


def transform(input: str, history: list[mel.ChatMessage]):
    try:
      response = rq.post("http://localhost:8000/chat", json=UserInput(prompt=input).model_dump())
    except Exception as e:
       response = rq.post("http://backend:8000/chat", json=UserInput(prompt=input).model_dump())
    res = response.json()["response"]
    yield res