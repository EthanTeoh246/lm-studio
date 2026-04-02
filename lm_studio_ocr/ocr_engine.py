import base64
import json
import os
import requests
import fitz  # PyMuPDF
import re

# --- LLM CONFIGURATION ---
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://192.168.1.96:1234/v1/chat/completions")
TEMPERATURE = 0.0  # Set to 0 for maximum accuracy/determinism
MAX_TOKENS = 4096

def process_file_to_base64(contents: bytes, mime_type: str) -> tuple[str, str]:
    """
    Converts uploaded file bytes into a base64 encoded JPEG image (for single page).
    Returns: (base64_string, error_message)
    """
    imgs, error = process_file_to_base64_list(contents, mime_type)
    if error:
        return "", error
    if not imgs:
        return "", "Failed to process image: no images generated"
    return imgs[0], ""


def process_file_to_base64_list(contents: bytes, mime_type: str) -> tuple[list[str], str]:
    """
    Converts PDF/Image to list of base64 images (one per page).
    Returns: (list_of_base64_strings, error_message)
    """
    from PIL import Image
    import io
    
    page_images = []
    
    if "pdf" in mime_type:
        try:
            doc = fitz.open(stream=contents, filetype="pdf")
            if len(doc) == 0:
                return [], "Uploaded PDF is empty."
            
            max_pages = 10  # Max pages to process
            total_pages = len(doc)
            pages_to_process = min(total_pages, max_pages)
            print(f"[DEBUG] PDF has {total_pages} pages, processing {pages_to_process}", flush=True)
            
            for page_num in range(pages_to_process):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=150)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Resize if too large
                max_size = 768
                if img.width > max_size or img.height > max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
                # Convert to base64
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=85)
                page_images.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))
            
            doc.close()
            
            if total_pages > max_pages:
                print(f"Warning: PDF has {total_pages} pages, only processing first {max_pages}")
            
            return page_images, ""
            
        except Exception as e:
            return [], f"Failed to process PDF: {str(e)}"
    
    elif "image" in mime_type:
        try:
            img = Image.open(io.BytesIO(contents))
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            
            max_size = 768
            if img.width > max_size or img.height > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            return [base64.b64encode(buffer.getvalue()).decode('utf-8')], ""
        except Exception as e:
            return [], f"Failed to load image: {str(e)}"
    else:
        return [], f"Unsupported file type: {mime_type}. Please upload a PDF or Image."

def extract_document_data(base64_img: str, custom_prompt: str = None, max_retries: int = 2) -> tuple[dict, str, str]:
    """
    Sends Base64 image to LM Studio and parses JSON.
    Returns: (parsed_data_dict, error_code, error_message)
    Implements retry logic for transient failures.
    """
    base_prompt = """You are a literal OCR engine. Do not summarize. Do not interpret. Do not use any internal reasoning or thought process.

STRICT EXTRACTION RULES:
1. Extract 'from' as the exact company name printed at the top of the document.
2. Extract 'documentNo' exactly as printed on the invoice.
3. Extract 'documentDate' exactly as printed on the invoice.
4. For 'summaryDescription', transcribe the text in the 'Description' column letter-for-letter. Do not rephrase or summarize.
5. For 'finalPayableAmount', extract the exact number from the 'Total' or 'Amount Due' field. Output as a plain number (e.g., 540.00).

IMPORTANT: If there are USER SPECIFIC INSTRUCTIONS below, ALWAYS follow those instructions instead of the default rules for any field specified.

Copy text exactly as it appears pixel-by-pixel.

OUTPUT: Return ONLY a valid JSON object. Do not include markdown blocks like ```json. Do not include trailing commas. Output raw JSON ONLY starting with { character."""
    
    if custom_prompt:
        print(f"[DEBUG] Custom prompt received: {custom_prompt}", flush=True)
        base_prompt += f"\n\n=== USER SPECIFIC INSTRUCTIONS (YOU MUST FOLLOW THESE EXACTLY) ===\n{custom_prompt}\n\n=== END OF USER INSTRUCTIONS ===\nRemember: You MUST follow the user instructions above exactly. Do not ignore them."
    else:
        print(f"[DEBUG] No custom prompt provided", flush=True)

    payload = {
        "model": "local-model",
        "messages": [
            {
                "role": "system",
                "content": "You are an OCR assistant. Always follow user instructions exactly. Do not ignore or modify user instructions."
            },
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
        "Content-Type": "application/json"
    }

    last_error = ""
    
    # Retry loop for transient failures
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(LM_STUDIO_URL, json=payload, headers=headers, timeout=120)
            
            # If successful, break out of retry loop
            if resp.status_code == 200:
                break
                
            # If server error (5xx), retry
            if resp.status_code >= 500:
                last_error = f"LM Studio server error: {resp.status_code}"
                continue
                
            # If client error (4xx), don't retry - it's a permanent failure
            return {}, str(resp.status_code), f"LM Studio API error: {resp.text}"
            
        except requests.exceptions.RequestException as e:
            last_error = str(e)
            if attempt < max_retries:
                continue
            return {}, "503", f"Could not connect to LM Studio at {LM_STUDIO_URL}. Is PC 2 turned on? Error: {last_error}"
    
    # If we exited loop without 200 response (after retries)
    if resp.status_code != 200:
        return {}, str(resp.status_code), f"LM Studio API error after {max_retries + 1} attempts: {last_error}"

    # Parse JSON text from LLM safely
    import sys
    try:
        resp_json = resp.json()
        extracted_text = resp_json['choices'][0]['message']['content'].strip()
        
        print(f"[DEBUG] Raw AI response: {extracted_text[:500]}...", flush=True)
            
        if not extracted_text:
            return {}, "500", "The AI returned an empty response. You might still be using a TEXT-ONLY model (like qwen3.5-9b). You MUST load a VISION model (like 'qwen2-vl-7b-instruct.gguf' or 'llava-1.5-7b') in LM Studio!"
            
        if extracted_text.startswith("```json"):
            extracted_text = extracted_text[7:]
        elif extracted_text.startswith("```"):
            extracted_text = extracted_text[3:]
        if extracted_text.endswith("```"):
            extracted_text = extracted_text[:-3]
            
        clean_text = extracted_text.strip()
        
        print(f"[DEBUG] Cleaned AI response: {clean_text[:200]}...", flush=True)
        
        # Check for empty response before parsing
        if not clean_text:
            return {}, "500", "AI returned empty response. Ensure LM Studio is running with a Vision model."
        
        # Regex to strip trailing commas that cause python's strict json parser to crash
        clean_text = re.sub(r',\s*}', '}', clean_text)
        clean_text = re.sub(r',\s*\]', ']', clean_text)
        
        # Use raw_decode to parse ONLY the FIRST valid JSON object and ignore trailing duplicates
        start_idx = clean_text.find('{')
        if start_idx != -1:
            clean_text = clean_text[start_idx:]
        else:
            return {}, "500", f"AI response does not contain valid JSON. Raw: {clean_text[:100]}"
        
        if not clean_text.strip():
            return {}, "500", "AI returned empty text after cleaning."
            
        decoder = json.JSONDecoder()
        data, _ = decoder.raw_decode(clean_text)
    except (json.JSONDecodeError, KeyError) as e:
        return {}, "500", f"LLM returned invalid JSON. Error: {str(e)}\n\nRaw Text from AI:\n{extracted_text}"

    # Format specifically for the response requirements
    try:
        final_amount = data.get("finalPayableAmount", 0.0)
        print(f"[DEBUG] AI returned finalPayableAmount: {final_amount} (type: {type(final_amount)})", flush=True)
        if isinstance(final_amount, str):
            # Remove currency symbols and codes (MYR, USD, $, etc.) and commas
            final_amount = re.sub(r'[A-Z]{3}', '', final_amount)  # Remove currency codes like MYR, USD
            final_amount = final_amount.replace("$", "").replace(",", "").strip()
            final_amount = float(final_amount) if final_amount else 0.0
            print(f"[DEBUG] After string conversion: {final_amount} (type: {type(final_amount)})", flush=True)
    except ValueError:
        final_amount = 0.0

    result_obj = {
        "from": str(data.get("from", "")),
        "documentNo": str(data.get("documentNo", "")),
        "documentDate": str(data.get("documentDate", "")),
        "summaryDescription": str(data.get("summaryDescription", "")),
        "finalPayableAmount": float(final_amount)
    }
    
    print(f"[DEBUG] Parsed result_obj: {result_obj}", flush=True)

    return result_obj, None, None


def extract_document_data_multi_page(base64_images: list[str], custom_prompt: str = None) -> tuple[dict, str, str]:
    """
    Process multiple pages one by one and combine results.
    Returns: (merged_result_dict, error_code, error_message)
    """
    if not base64_images:
        return {}, "400", "No images to process"
    
    print(f"[DEBUG] Starting multi-page extraction with {len(base64_images)} pages", flush=True)
    
    if len(base64_images) == 1:
        return extract_document_data(base64_images[0], custom_prompt)
    
    page_results = []
    
    for idx, base64_img in enumerate(base64_images):
        print(f"[DEBUG] Processing page {idx + 1}/{len(base64_images)}", flush=True)
        result, err_code, error = extract_document_data(base64_img, custom_prompt)
        
        if error:
            return {}, err_code, f"Error on page {idx + 1}: {error}"
        
        print(f"[DEBUG] Page {idx + 1} result: {result}", flush=True)
        page_results.append(result)
    
    print(f"[DEBUG] All page results: {page_results}", flush=True)
    
    # Merge results from all pages
    # Strategy: FIRST non-empty value for most fields, EXCEPT finalPayableAmount (use LAST non-zero)
    merged = {
        "from": "",
        "documentNo": "",
        "documentDate": "",
        "summaryDescription": "",
        "finalPayableAmount": 0.0
    }
    
    for idx, page in enumerate(page_results):
        print(f"[DEBUG] Merging page {idx + 1}: {page}", flush=True)
        
        # Use FIRST non-empty value for these fields
        if not merged["from"] and page.get("from"):
            merged["from"] = page["from"]
        if not merged["documentNo"] and page.get("documentNo") and page["documentNo"] not in ["None", "null", ""]:
            merged["documentNo"] = page["documentNo"]
        if not merged["documentDate"] and page.get("documentDate") and page["documentDate"] not in ["None", "null", ""]:
            merged["documentDate"] = page["documentDate"]
        if not merged["summaryDescription"] and page.get("summaryDescription"):
            merged["summaryDescription"] = page["summaryDescription"]
        
        # Use LAST non-zero value for amount (total is usually on last page)
        if page.get("finalPayableAmount") and page.get("finalPayableAmount") != 0:
            merged["finalPayableAmount"] = page["finalPayableAmount"]
    
    print(f"[DEBUG] Final merged result: {merged}", flush=True)
    
    return merged, None, None