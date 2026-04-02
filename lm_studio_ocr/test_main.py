"""
Unit Tests for LM Studio OCR API

Run tests with: pytest test_main.py -v
"""

import pytest
import json
import base64
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient

# Import the app and modules
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import app
from ocr_engine import process_file_to_base64, process_file_to_base64_list, extract_document_data


client = TestClient(app)


# ==================== API Endpoint Tests ====================

class TestAPIEndpoints:
    """Test FastAPI endpoints"""
    
    def test_root_returns_html(self):
        """Test that root endpoint returns the HTML UI"""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type")
    
    def test_extract_endpoint_no_file(self):
        """Test extract endpoint with no file returns 422"""
        response = client.post("/api/v1/extract")
        assert response.status_code == 422  # FastAPI validation error
    
    def test_extract_batch_endpoint_no_files(self):
        """Test batch endpoint with no files returns 422"""
        response = client.post("/api/v1/extract/batch")
        assert response.status_code == 422


# ==================== OCR Engine Tests ====================

class TestOCREngine:
    """Test OCR engine functions"""
    
    def test_process_file_to_base64_with_invalid_type(self):
        """Test that unsupported file types return error"""
        result, error = process_file_to_base64(b"some content", "application/unknown")
        assert result == ""  # Returns empty string for no images
        assert "Unsupported" in error
    
    def test_process_file_to_base64_with_empty_pdf(self):
        """Test that empty PDF returns error"""
        # Create minimal PDF
        pdf_content = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n%\x00\x00\x00\x00\x00\x00"
        result, error = process_file_to_base64(pdf_content, "application/pdf")
        assert result == "" or error != ""  # Should return error for invalid PDF
    
    def test_extract_document_data_with_empty_base64(self):
        """Test extract with empty base64 returns error"""
        result, err_code, error = extract_document_data("")
        assert result == {}
        assert err_code in ["400", "500", "503"]
        assert "empty" in error.lower() or "vision" in error.lower() or "connect" in error.lower()
    
    @patch('ocr_engine.requests.post')
    def test_extract_document_data_connection_error(self, mock_post):
        """Test connection error to LM Studio returns 503"""
        import requests
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")
        
        result, err_code, error = extract_document_data("fake_base64_image")
        
        assert result == {}
        assert err_code == "503"
        assert "Could not connect" in error
    
    @patch('ocr_engine.requests.post')
    def test_extract_document_data_invalid_json_response(self, mock_post):
        """Test invalid JSON from LLM returns error"""
        # Mock successful HTTP response but invalid JSON from AI
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "This is not valid JSON {"
                    }
                }
            ]
        }
        mock_post.return_value = mock_response
        
        result, err_code, error = extract_document_data("fake_base64")
        
        assert result == {}
        assert err_code == "500"
        assert "invalid json" in error.lower()
    
    @patch('ocr_engine.requests.post')
    def test_extract_document_data_successful_extraction(self, mock_post):
        """Test successful JSON extraction from AI"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"from": "Test Company", "documentNo": "INV001", "documentDate": "2024-01-01", "summaryDescription": "Test item", "finalPayableAmount": 100.50}'
                    }
                }
            ]
        }
        mock_post.return_value = mock_response
        
        result, err_code, error = extract_document_data("fake_base64")
        
        assert err_code is None
        assert error is None
        assert result["from"] == "Test Company"
        assert result["documentNo"] == "INV001"
        assert result["documentDate"] == "2024-01-01"
        assert result["summaryDescription"] == "Test item"
        assert result["finalPayableAmount"] == 100.50
    
    @patch('ocr_engine.requests.post')
    def test_extract_document_data_with_dollar_sign(self, mock_post):
        """Test that dollar signs are stripped from amount"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"from": "Test", "documentNo": "1", "documentDate": "01/01/2024", "summaryDescription": "test", "finalPayableAmount": "$1,234.56"}'
                    }
                }
            ]
        }
        mock_post.return_value = mock_response
        
        result, err_code, error = extract_document_data("fake_base64")
        
        assert result["finalPayableAmount"] == 1234.56
    
    @patch('ocr_engine.requests.post')
    def test_extract_document_data_with_markdown(self, mock_post):
        """Test that markdown code blocks are stripped"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "```json\n{\"from\": \"Test\", \"documentNo\": \"1\", \"documentDate\": \"2024-01-01\", \"summaryDescription\": \"test\", \"finalPayableAmount\": 100}\n```"
                    }
                }
            ]
        }
        mock_post.return_value = mock_response
        
        result, err_code, error = extract_document_data("fake_base64")
        
        assert err_code is None
        assert result["from"] == "Test"
    
    @patch('ocr_engine.requests.post')
    def test_extract_document_data_with_trailing_comma(self, mock_post):
        """Test that trailing commas are handled"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"from": "Test", "documentNo": "1", "documentDate": "2024-01-01", "summaryDescription": "test", "finalPayableAmount": 100, }'
                    }
                }
            ]
        }
        mock_post.return_value = mock_response
        
        result, err_code, error = extract_document_data("fake_base64")
        
        assert err_code is None
        assert result["finalPayableAmount"] == 100


# ==================== Schema Tests ====================

class TestSchemas:
    """Test Pydantic schemas"""
    
    def test_api_response_schema(self):
        """Test APIResponse schema validation"""
        from schemas import APIResponse
        
        # Test success response
        response = APIResponse(
            errCode=None,
            error=None,
            result={"from": "Test", "documentNo": "123"}
        )
        assert response.errCode is None
        assert response.result["from"] == "Test"
        
        # Test error response
        error_response = APIResponse(
            errCode="500",
            error="Test error",
            result=None
        )
        assert error_response.errCode == "500"
        assert error_response.error == "Test error"


# ==================== Helper Tests ====================

class TestHelpers:
    """Test helper functions"""
    
    def test_base64_encoding(self):
        """Test base64 encoding works correctly"""
        test_string = "Hello, World!"
        encoded = base64.b64encode(test_string.encode()).decode()
        decoded = base64.b64decode(encoded.encode()).decode()
        assert decoded == test_string
    
    def test_json_parsing(self):
        """Test JSON parsing with various formats"""
        # Valid JSON
        data = json.loads('{"key": "value"}')
        assert data["key"] == "value"
        
        # JSON with trailing comma (should fail with standard parser)
        with pytest.raises(json.JSONDecodeError):
            json.loads('{"key": "value",}')
    
    def test_regex_trailing_comma_removal(self):
        """Test regex removes trailing commas"""
        import re
        
        text = '{"key": "value", }'
        cleaned = re.sub(r',\s*}', '}', text)
        cleaned = re.sub(r',\s*\]', ']', cleaned)
        assert cleaned == '{"key": "value"}'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
