from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel
from agent import agent

class UserInput(BaseModel):
    prompt: str

class ApiOutput(BaseModel):
    response: str

app = FastAPI(default_response_class=ORJSONResponse)

@app.post("/chat")
async def chat(inpt: UserInput):
    response = await agent.achat(message=inpt.prompt)
    response = str(response)
    return ApiOutput(response=response)