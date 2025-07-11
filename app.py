import os
import uuid
import base64
import re
import json
from io import BytesIO
from PIL import Image
from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
import cv2 as cv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

API = os.getenv("OPENAI_API_KEY")
if not API:
    raise ValueError("OPENAI_API_KEY not set in environment variables")

llm = ChatOpenAI(
    model="gpt-4.1",
    openai_api_key=API,
    temperature=0,
    max_tokens=20000
)

TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

def encode_image(image_path):
    image = cv.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at path: {image_path}")
    _, buffer = cv.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")

def extract_json_response(raw):
    try:
        match = re.search(r'\{.*"output"\s*:\s*".*?".*?\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return {"output": "unknown", "summary": raw.strip()}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    user_prompt = request.form.get('user_prompt', '')
    system_prompt = request.form.get('system_message', '').strip()
    image_files = request.files.getlist('images')

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

    user_msg_content.append({
        "type": "text",
        "text": f"""{user_prompt}\n\nRespond only in strict JSON format:\n{{\n  "output": "yes" or "no",\n  "summary": "reasoning"\n}}"""
    })

    try:
        # Load message history
        messages_data = session.get("messages", [])
        messages = []

        for msg in messages_data:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
            elif msg["role"] == "system":
                messages.append(SystemMessage(content=msg["content"]))

        # Add system message only at beginning of conversation
        if system_prompt and not messages_data:
            system_msg = SystemMessage(content=system_prompt)
            messages.insert(0, system_msg)

        messages.append(HumanMessage(content=user_msg_content))
        response = llm.invoke(messages)
        messages.append(AIMessage(content=response.content))

        # Save back to session
        session["messages"] = [
            {"role": "user" if isinstance(m, HumanMessage) else
             "assistant" if isinstance(m, AIMessage) else
             "system", "content": m.content}
            for m in messages
        ]

        parsed = extract_json_response(response.content)

        return jsonify({
            "output": parsed.get("output", "unknown"),
            "summary": parsed.get("summary", response.content.strip())
        })

    finally:
        for path in temp_paths:
            try:
                os.remove(path)
            except Exception as e:
                print(f"Failed to delete temp file {path}: {e}")

@app.route('/reset-session', methods=['POST'])
def reset_session():
    session.pop("messages", None)
    return jsonify({"status": "session reset"})

if __name__ == '__main__':
    app.run(debug=True)
