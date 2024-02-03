import os
import platform
from transformers import AutoTokenizer, AutoModel
from flask import Flask, request, Response
import requests
MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/chatglm3-6b')
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, device_map="auto").eval()

stop_stream = False

past_key_values, history = None, []

welcome_prompt = "欢迎使用 ChatGLM3-6B 模型，输入内容即可进行对话，clear 清空对话历史"

app = Flask(__name__)

@app.route("/")
def main():
    global stop_stream, past_key_values, history
    query = request.args.get("text")
    if not query.strip():
        return welcome_prompt
    if query.strip() == "clear":
        past_key_values, history = None, []
        return "已清空对话历史"
    for response, history, past_key_values in model.stream_chat(tokenizer, query, history=history, top_p=1,
                                                                temperature=0.01,
                                                                past_key_values=past_key_values,
                                                                return_past_key_values=True):
        if stop_stream:
            stop_stream = False
            return response

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080)