# ImageToText - OCR Project using Hugging Face Transformers

This project provides two methods to perform Optical Character Recognition (OCR) on images using a pre-trained model from Hugging Face's `transformers` library:
1. Single-image processing using the command line.
2. Flask-based API for handling image uploads via HTTP requests.

The project is optimized for running on GPUs with `float16` precision for efficient computation.

## Requirements

- Python 3.x
- PyTorch 2.5.0
- Hugging Face Transformers 4.44.2
- Flask (for the API version)
- CUDA

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/huggingface-ocr-project.git
    cd huggingface-ocr-project
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Single Image OCR Using Command Line (Argparse)

To process a single image via the command line:

```bash
python ocr_script.py /path/to/image.jpg
```

### 2. Running the Flask API for OCR

The Flask API allows you to upload images via an HTTP request and receive OCR results in JSON format.

#### Steps to Run the Flask API:

1. **Start the Flask server**:
   ```bash
   python server.py


The server will run on http://localhost:5000 by default.

Send a POST request with an image file to the /ocr endpoint:

You can use tools like curl, Postman, or any API client to send the image. Here’s how to do it with curl:
curl -X POST -F "image_file=@/path/to/your/image.jpg" http://localhost:5000/ocr
Replace /path/to/your/image.jpg with the path to the image file you want to process.

API Response: The response will be returned in JSON format and will look something like this:
```{
    "ocr_result": "The extracted text from the image goes here."
}```

Example of API Request:

curl -X POST -F "image_file=@/path/to/image.jpg" http://localhost:5000/ocr