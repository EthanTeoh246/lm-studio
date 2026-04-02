# Running Tests

## Install Test Dependencies

```bash
pip install -r requirements.txt
```

## Run All Tests

```bash
pytest test_main.py -v
```

## Run Specific Test Classes

```bash
# Test only API endpoints
pytest test_main.py::TestAPIEndpoints -v

# Test only OCR engine
pytest test_main.py::TestOCREngine -v

# Test only schemas
pytest test_main.py::TestSchemas -v
```

## Test Coverage

The test suite covers:

### API Endpoint Tests
- Root endpoint returns HTML
- Extract endpoint validation (no file)
- Batch endpoint validation (no files)

### OCR Engine Tests
- Unsupported file type handling
- Empty PDF handling
- Connection error handling
- Invalid JSON response handling
- Successful extraction
- Dollar sign amount parsing
- Markdown code block stripping
- Trailing comma handling

### Schema Tests
- APIResponse validation
- Error response format

### Helper Tests
- Base64 encoding/decoding
- JSON parsing
- Regex trailing comma removal
