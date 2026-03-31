import base64
import json
import os
import requests
import fitz  # PyMuPDF
import re

# --- LLM CONFIGURATION ---
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", " https://jfmop-2001-e68-82ca-ee01-d88f-50c8-ab84-fa0c.a.free.pinggy.link/v1/chat/completions")
TEMPERATURE = 0.0  # Set to 0 for maximum accuracy/determinism
MAX_TOKENS = 4096

def process_file_to_base64(contents: bytes, mime_type: str) -> tuple[str, str]:
    """
    Converts uploaded file bytes into a base64 encoded JPEG image.
    Shrinks images to prevent llama.cpp batch boundary crashes.
    """
    from PIL import Image
    import io
    
    img = None
    if "pdf" in mime_type:
        try:
            doc = fitz.open(stream=contents, filetype="pdf")
            if len(doc) == 0:
                return "", "Uploaded PDF is empty."
            
            page = doc.load_page(0)
            pix = page.get_pixmap(dpi=150)  # Increased from 72 to 150 DPI for better text clarity
            # PyMuPDF to Pillow Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            doc.close()
        except Exception as e:
            return "", f"Failed to process PDF: {str(e)}"
            
    elif "image" in mime_type:
        try:
            img = Image.open(io.BytesIO(contents))
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
        except Exception as e:
            return "", f"Failed to load image: {str(e)}"
    else:
        return "", f"Unsupported file type: {mime_type}. Please upload a PDF or Image."

    if img:
        try:
            # Higher resolution to prevent misreading small text (increased from 512 to 768)
            max_size = 768
            if img.width > max_size or img.height > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)  # Slightly higher quality for clearer text
            return base64.b64encode(buffer.getvalue()).decode('utf-8'), ""
        except Exception as e:
            return "", f"Failed to compress image: {str(e)}"

def extract_document_data(base64_img: str, custom_prompt: str = None) -> tuple[dict, str, str]:
    """
    Sends Base64 image to LM Studio and parses JSON.
    Returns: (parsed_data_dict, error_code, error_message)
    """
    base_prompt = """You are a literal OCR engine. Do not summarize. Do not interpret. Do not use any internal reasoning or thought process.

STRICT EXTRACTION RULES:
1. Extract 'from' as the exact company name printed at the top of the document.
2. Extract 'documentNo' exactly as printed on the invoice.
3. Extract 'documentDate' exactly as printed on the invoice.
4. For 'summaryDescription', transcribe the text in the 'Description' column letter-for-letter. Do not rephrase or summarize.
5. For 'finalPayableAmount', extract the exact number from the 'Total' or 'Amount Due' field. Output as a plain number (e.g., 540.00).

Copy text exactly as it appears pixel-by-pixel.

OUTPUT: Return ONLY a valid JSON object. Do not include markdown blocks like ```json. Do not include trailing commas. Output raw JSON ONLY starting with { character."""
    
    if custom_prompt:
        base_prompt += f"\n\nUSER SPECIFIC INSTRUCTIONS:\n{custom_prompt}"

    payload = {
        "model": "local-model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": base_prompt
                    },
                    {
                        "type": "image_url", 
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_img}"
                        }
                    }
                ]
            }
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS
    }

    headers = {
        "User-Agent": "FastAPI-App",
        "Pinggy-Skip-Browser-Warning": "true",
        "Content-Type": "application/json"
    }

    try:
        resp = requests.post(LM_STUDIO_URL, json=payload, headers=headers, timeout=120)
    except requests.exceptions.RequestException as e:
        return {}, "503", f"Could not connect to LM Studio at {LM_STUDIO_URL}. Is PC 2 turned on? Error: {str(e)}"

    if resp.status_code != 200:
        return {}, str(resp.status_code), f"LM Studio API error: {resp.text}"

    # Parse JSON text from LLM safely
    try:
        resp_json = resp.json()
        extracted_text = resp_json['choices'][0]['message']['content'].strip()
            
        if not extracted_text:
            return {}, "500", "The AI returned an empty response. You might still be using a TEXT-ONLY model (like qwen3.5-9b). You MUST load a VISION model (like 'qwen2-vl-7b-instruct.gguf' or 'llava-1.5-7b') in LM Studio!"
            
        if extracted_text.startswith("```json"):
            extracted_text = extracted_text[7:]
        elif extracted_text.startswith("```"):
            extracted_text = extracted_text[3:]
        if extracted_text.endswith("```"):
            extracted_text = extracted_text[:-3]
            
        clean_text = extracted_text.strip()
        
        # Regex to strip trailing commas that cause python's strict json parser to crash
        clean_text = re.sub(r',\s*}', '}', clean_text)
        clean_text = re.sub(r',\s*\]', ']', clean_text)
        
        # Use raw_decode to parse ONLY the FIRST valid JSON object and ignore trailing duplicates
        start_idx = clean_text.find('{')
        if start_idx != -1:
            clean_text = clean_text[start_idx:]
            
        decoder = json.JSONDecoder()
        data, _ = decoder.raw_decode(clean_text)
    except (json.JSONDecodeError, KeyError) as e:
        return {}, "500", f"LLM returned invalid JSON. Error: {str(e)}\n\nRaw Text from AI:\n{extracted_text}"

    # Format specifically for the response requirements
    try:
        final_amount = data.get("finalPayableAmount", 0.0)
        if isinstance(final_amount, str):
            final_amount = final_amount.replace("$", "").replace(",", "")
            final_amount = float(final_amount) if final_amount else 0.0
    except ValueError:
        final_amount = 0.0

    result_obj = {
        "from": str(data.get("from", "")),
        "documentNo": str(data.get("documentNo", "")),
        "documentDate": str(data.get("documentDate", "")),
        "summaryDescription": str(data.get("summaryDescription", "")),
        "finalPayableAmount": final_amount
    }

    return result_obj, None, None