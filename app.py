from fastapi import FastAPI, Form

from fastapi.responses import JSONResponse

import asyncio

from transformers import AutoModel, AutoTokenizer
 
app = FastAPI()

access_token ="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"
model = AutoModel.from_pretrained('ReNoteTech/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=True, use_auth_token=access_token)

tokenizer = AutoTokenizer.from_pretrained('ReNoteTech/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=True, use_auth_token=access_token)
 
@app.post("/continue-chat/")

async def continue_chat(message: str = Form(...)):

    global msgs, res, continue_msgs, image

    continue_msgs.append({'role': 'user', 'content': message})

    try:

        loop = asyncio.get_event_loop()

        res = await loop.run_in_executor(None, model.chat, image, continue_msgs, tokenizer, True, 0.1)

        continue_msgs.append({'role': 'user', 'content': res})

        return res

    except TypeError as e:

        return JSONResponse(status_code=500, content={"error": str(e)})
 
# To run the server, use: uvicorn my_app:app --reload
