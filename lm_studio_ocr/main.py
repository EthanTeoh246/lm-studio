from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List

# Import from local flat files
from schemas import APIResponse
import ocr_engine

app = FastAPI(
    title="LM Studio OCR API",
    description="A microservice to extract structured JSON data from PDFs/Images via Local LLM.",
    version="1.0.0"
)

# Enable CORS for future frontends/projects to easily connect to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# ----------------- UI ROUTE -----------------
@app.get("/", tags=["UI"])
async def serve_ui():
    """Serves the frontend testing UI"""
    return FileResponse("templates/index.html")

# ----------------- API ROUTE: SINGLE FILE -----------------
@app.post("/api/v1/extract", response_model=APIResponse, tags=["OCR API"])
async def extract_document_api(
    file: UploadFile = File(...),  # Consider adding max_size for file limit
    custom_prompt: Optional[str] = Form(None)
):
    """
    Upload a single PDF or Image, and receive structured JSON extracted via the Local LLM.
    """
    try:
        # Check file size (max 10MB)
        contents = await file.read()
        if len(contents) > 10_000_000:
            return APIResponse(errCode="413", error="File too large. Maximum size is 10MB.", result=None)
        
        mime_type = file.content_type
        
        # 1. Convert File to Base64 Images (one per page)
        base64_images, error_msg = ocr_engine.process_file_to_base64_list(contents, mime_type)
        if error_msg:
            return APIResponse(errCode="400", error=error_msg, result=None)

        # 2. Extract Data via LLM (processes each page)
        result_obj, err_code, llm_error = ocr_engine.extract_document_data_multi_page(base64_images, custom_prompt)
        if llm_error:
            return APIResponse(errCode=err_code, error=llm_error, result=None)

        # 3. Return Success
        return APIResponse(errCode=None, error=None, result=result_obj)

    except Exception as e:
        return APIResponse(errCode="500", error=f"Internal Server Error: {str(e)}", result=None)

# ----------------- API ROUTE: MULTIPLE FILES -----------------
@app.post("/api/v1/extract/batch", tags=["OCR API"])
async def extract_multiple_documents(
    files: List[UploadFile] = File(...),
    custom_prompt: Optional[str] = Form(None)
):
    """
    Upload multiple PDFs or Images, and receive an array of structured JSON results.
    """
    results = []
    errors = []
    
    for idx, file in enumerate(files):
        try:
            contents = await file.read()
            
            # Check individual file size (max 10MB)
            if len(contents) > 10_000_000:
                errors.append({"file": file.filename, "error": "File too large. Maximum size is 10MB."})
                continue
                
            mime_type = file.content_type
            
            # Convert File to Base64 Images (one per page)
            base64_images, error_msg = ocr_engine.process_file_to_base64_list(contents, mime_type)
            if error_msg:
                errors.append({"file": file.filename, "error": error_msg})
                continue

            # Extract Data via LLM (processes each page)
            result_obj, err_code, llm_error = ocr_engine.extract_document_data_multi_page(base64_images, custom_prompt)
            if llm_error:
                errors.append({"file": file.filename, "error": llm_error})
                continue
            
            # Add filename to result
            result_obj["_filename"] = file.filename
            results.append(result_obj)
            
        except Exception as e:
            errors.append({"file": file.filename, "error": str(e)})
    
    return APIResponse(
        errCode=None if not errors else "206",
        error=None,
        result={
            "totalFiles": len(files),
            "successful": len(results),
            "failed": len(errors),
            "items": results,
            "errors": errors
        }
    )