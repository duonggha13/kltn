from typing import Optional

from fastapi import FastAPI, Response, Form, HTTPException, File, UploadFile
from class_ner_re import ner, re_from_file

import io
import json
import os

app = FastAPI()

@app.post("/ner/")
async def ner_from_file(file: UploadFile = File(...)):
	ner_data = ner(file.filename)
	return Response(content=ner_data, media_type="application/txt")
@app.post("/re/")
async def relationship_extract(file: UploadFile = File(...)):
 	re_data = re_from_file(file.filename)
 	return Response(content=re_data, media_type="application/txt")
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)