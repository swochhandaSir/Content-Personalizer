from vllm import LLM, SamplingParams
import uvicorn
from fastapi import FastAPI, Request
import json

app = FastAPI()
llm = LLM(model="souldrr/Llama-3.2-1B-fine-tune-300-movies-50-review")

@app.post("/v1/completions")
async def generate(request: Request):
    data = await request.json()
    sampling_params = SamplingParams(
        max_tokens=data.get("max_tokens", 512),
        temperature=data.get("temperature", 0.5)
    )
    
    outputs = llm.generate(data["prompt"], sampling_params)
    return {"choices": [{"text": output.outputs[0].text} for output in outputs]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)