import os
import uuid
import base64
from io import BytesIO
from PIL import Image
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain_core.messages import SystemMessage
import cv2 as cv

app = Flask(__name__)

API = os.getenv("OPENAI_API_KEY")
if not API:
    raise ValueError("OPENAI_API_KEY not set in environment variables")

llm = ChatOpenAI(
    model="gpt-4.1",
    openai_api_key=API,
    temperature=0.5,
    max_tokens=20000
)

system_message = SystemMessage(
    content=(
        ""
    )
)

memory = ConversationBufferMemory(memory_key="history", return_messages=True)

TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

# Util: Encode image at path to base64
def encode_image(image_path):
    image = cv.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at path: {image_path}")
    _, buffer = cv.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    user_prompt = request.form.get('user_prompt', '')
    system_prompt = request.form.get('system_message', '')
    image_files = request.files.getlist('images')

    if system_prompt and len(memory.chat_memory.messages) == 0:
        memory.chat_memory.add_message(SystemMessage(content=system_prompt))

    user_msg_content = []
    temp_paths = []

    for image_file in image_files:
        if image_file and image_file.filename:
            temp_filename = f"{uuid.uuid4().hex}.jpg"
            temp_path = os.path.join(TEMP_DIR, temp_filename)
            image_file.save(temp_path)
            temp_paths.append(temp_path)

            try:
                image_b64 = encode_image(temp_path)
                user_msg_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}"
                    }
                })
            except Exception as e:
                print(f"Image encoding error: {e}")

    if user_prompt:
        user_msg_content.append({
            "type": "text",
            "text": user_prompt
        })

    try:
        full_messages = [system_message] + memory.load_memory_variables({})["history"] + [
            HumanMessage(content=user_msg_content)
        ]
        response = llm.invoke(full_messages)

        memory.chat_memory.add_user_message(HumanMessage(content=user_msg_content))
        memory.chat_memory.add_ai_message(AIMessage(content=response.content))

        output = []
        for msg in memory.chat_memory.messages:
            if isinstance(msg, HumanMessage):
                items = []
                for item in msg.content:
                    if item.get("type") == "image_url":
                        items.append("[Image attached]")
                    elif item.get("type") == "text":
                        items.append(item.get("text"))
                output.append({"role": "user", "content": "\n".join(items)})
            elif isinstance(msg, AIMessage):
                output.append({"role": "assistant", "content": msg.content})

        return jsonify({"messages": output})

    finally:
        for path in temp_paths:
            try:
                os.remove(path)
            except Exception as e:
                print(f"Failed to delete temp file {path}: {e}")
@app.route('/reset-session', methods=['POST'])
def reset_session():
    global memory
    memory.clear()
    return jsonify({"status": "session reset"})
if __name__ == '__main__':
    app.run(debug=True)
