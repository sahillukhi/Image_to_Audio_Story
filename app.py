import requests
import textwrap
from groq import Groq
from gtts import gTTS
import os
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from livereload import Server

# Load environment variables
load_dotenv()

# Retrieve API keys from .env
HF_API_KEYS = os.getenv("HF_API_KEYS")
GROQ_API_KEYS = os.getenv("GROQ_API_KEYS")

app = Flask(__name__)
upload_folder = r"C:\Users\sahil\Downloads\FLASK\upload"
allowed_extension = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = upload_folder
os.makedirs(upload_folder, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extension

@app.route('/', methods=['GET', 'POST'])
def upload_and_generate():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            try:
                caption = get_caption(file_path)
                story = cap2story(prompt_gen(caption))
                audio_path = text2aud(story)

                return jsonify({
                    'status': 'success',
                    'image_path': f'/uploads/{filename}',
                    'caption': caption,
                    'story': story,
                    'audio_url': f'/audio/{os.path.basename(audio_path)}'
                })
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)})
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def serve_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_from_directory('.', filename)

def get_caption(image_path, API=HF_API_KEYS):
    API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
    headers = {
        "Authorization": f"Bearer {API}",
        "Content-Type": "application/octet-stream"
    }
    with open(image_path, "rb") as image_file:
        response = requests.post(API_URL, headers=headers, data=image_file.read())
        response.raise_for_status()
        result = response.json()
        return result[0].get('generated_text', 'No caption generated')

def prompt_gen(caption):
    prompt = PromptTemplate.from_template("""
    You are a creative storyteller. Based on the following image caption, write a detailed story using key storytelling elements.

    Include the following aspects:
    - Genre: Choose one suitable genre..
    - Characters: Develop engaging characters.
    - Plot Structure: Follow a clear structure.
    - Emotion: Evoke emotions through descriptive language.
    - Twist: Add a surprising twist.
    - Do not use personal names.
    - Do not use day/night time if it is not mentioned in Caption
    Caption: {caption}
    The story should be between 90 to 100 words.
    """)
    return prompt.format(caption=caption)

def cap2story(content, API=GROQ_API_KEYS):
    client = Groq(api_key=API)
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a creative storyteller."},
            {"role": "user", "content": content},
        ],
        temperature=0.5,
        max_tokens=200,
        top_p=0.7,
        stream=True,
    )
    story = ''.join(chunk.choices[0].delta.content or '' for chunk in completion)
    return textwrap.fill(story, width=80)

def text2aud(text, lang="en"):
    audio_path = "story.mp3"
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(audio_path)
    return audio_path

if __name__ == "__main__":
    server = Server(app.wsgi_app)
    server.watch('templates/*.html')
    server.watch('static/*.*')
    server.serve(port=5000)