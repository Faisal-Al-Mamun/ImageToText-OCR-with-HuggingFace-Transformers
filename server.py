# server.py
from flask import Flask, request, jsonify
from transformers import AutoModel, AutoTokenizer
import torch
import os

app = Flask(__name__)

# Load model and tokenizer only once when the server starts
print("Loading model and tokenizer...")
model_name = 'ucaslcl/GOT-OCR2_0'
if torch.cuda.is_available():
    print("CUDA device is available. Using float16 precision.")
    torch_dtype = torch.float16
else:
    print("CUDA device is not available. Using float32 precision on CPU.")
    torch_dtype = torch.float32

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    device_map='cuda' if torch.cuda.is_available() else 'cpu',
    use_safetensors=True,
    pad_token_id=tokenizer.eos_token_id,
    torch_dtype=torch_dtype
)
model = model.eval().to('cuda' if torch.cuda.is_available() else 'cpu')

def perform_ocr(model, tokenizer, image_file):
    """Perform OCR on the input image and return the extracted text."""
    if not os.path.exists(image_file):
        raise FileNotFoundError(f"Image file {image_file} not found!")
    
    res = model.chat(tokenizer, image_file, ocr_type='format')
    return res

@app.route("/ocr", methods=["POST"])
def ocr():
    if 'image_file' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    image_file = request.files['image_file']
    image_path = f"uploads/{image_file.filename}"
    image_file.save(image_path)
    
    try:
        result = perform_ocr(model, tokenizer, image_path)
        return jsonify({"ocr_result": result})
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404

if __name__ == "__main__":
    # Create the uploads folder if it doesn't exist
    os.makedirs('uploads', exist_ok=True)
    
    # Start the Flask server
    app.run(host="0.0.0.0", port=5000)
