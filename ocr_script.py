# main.py
from transformers import AutoModel, AutoTokenizer
import torch
import argparse
import os

def load_model_and_tokenizer(model_name='ucaslcl/GOT-OCR2_0'):
    """Load the pre-trained OCR model and tokenizer, with float16 precision for GPU compatibility."""
    
    # Use float16 if a GPU is available, fallback to float32 on CPU
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
        torch_dtype=torch_dtype  # Ensure float16 for compatible GPUs
    )
    model = model.eval().to('cuda' if torch.cuda.is_available() else 'cpu')
    return model, tokenizer

def perform_ocr(model, tokenizer, image_file):
    """Perform OCR on the input image and return the extracted text."""
    if not os.path.exists(image_file):
        raise FileNotFoundError(f"Image file {image_file} not found!")
    
    res = model.chat(tokenizer, image_file, ocr_type='format')
    return res

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run OCR on an image using a pre-trained model.")
    parser.add_argument("image_file", type=str, help="Path to the image file for OCR.")
    args = parser.parse_args()

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()

    # Perform OCR
    print(f"Performing OCR on image: {args.image_file}")
    result = perform_ocr(model, tokenizer, args.image_file)

    # Output the result
    print("OCR Result:")
    print(result)

if __name__ == "__main__":
    main()
