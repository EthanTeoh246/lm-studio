# LM Studio OCR API

This project is a FastAPI microservice that allows you to upload an Image or PDF, converts it into a Base64 image, and sends it to a local Vision LLM (hosted on LM Studio on another PC) to extract structured JSON data.

## Project Structure

* **`main.py`**: The entry point for the FastAPI server. It defines the API routes, handles Cross-Origin Resource Sharing (CORS) so other apps can talk to it, and serves the UI.
* **`ocr_engine.py`**: The "brain" of the application. It contains the business logic for handling PDFs/images and communicating with LM Studio.
* **`schemas.py`**: Uses Pydantic to strictly define what the input and output JSON should look like.
* **`templates/index.html`**: A clean, modern testing UI built with Tailwind CSS.

## How the Code Works (Step-by-Step)

### 1. The Data Models (`schemas.py`)
This file ensures that our API always returns the exact format you requested:
```json
{
    "errCode": null,
    "error": null,
    "result": { ... }
}
```
We use `BaseModel` from the `pydantic` library. `ExtractionResult` defines the fields (e.g., `documentNo`, `finalPayableAmount`). Because `from` is a reserved word in Python, we use `from_` in the code, but Pydantic's `alias_generator` automatically renames it back to `from` when converting it to JSON.

### 2. The Engine (`ocr_engine.py`)
This file has two main jobs:
1. **`process_file_to_base64`**: If you upload an image, it simply encodes the raw bytes into a Base64 string. If you upload a **PDF**, it uses `PyMuPDF` (`fitz`) to open the file, extract the very first page, render it into a JPEG image in memory, and *then* Base64 encodes it.
2. **`extract_document_data`**: This takes the Base64 string and constructs a payload that mimics the OpenAI Vision API format. It sends this payload to `192.168.1.95:1234` (your LM Studio PC). When the LLM replies with a text string containing JSON, this function strips away any markdown (like ````json ... ````) and safely converts the text into a real Python dictionary.

### 3. The API Server (`main.py`)
This file ties everything together using **FastAPI**.
* **CORS Middleware**: We enabled CORS (`CORSMiddleware`). Browsers block web pages from making API requests to different ports/domains for security. This middleware explicitly tells the browser: *"It's okay, allow future web apps to talk to this API."*
* **The Endpoint (`/api/v1/extract`)**: It waits for a file upload (`UploadFile`). When a file arrives, it hands it to the `ocr_engine`, gets the extracted JSON back, and packages it perfectly into the `APIResponse` schema to send back to the user.

### 4. The Frontend UI (`templates/index.html`)
The UI is a single HTML file. 
* It uses **Tailwind CSS** via a CDN link, which allows us to style the page beautifully without needing a complex Node.js/NPM setup.
* It uses vanilla Javascript (`fetch` API) to grab the file from the drag-and-drop zone and send it to the `/api/v1/extract` endpoint.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Start the server: `uvicorn main:app --reload`
3. Open `http://localhost:8000`