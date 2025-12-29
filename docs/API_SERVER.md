# DocStrange API Server

Run DocStrange as a REST API server with GPU acceleration, designed for integration with external applications like Agent Steve.

## Installation

```bash
pip install -e ".[api]"
```

This installs: FastAPI, uvicorn, python-multipart, pydantic

## Usage

### Start API Server

```bash
docstrange-api
```

Default configuration:
- Port: 8000
- Workers: 4
- GPU mode: enabled (if available)

### Custom Configuration

```bash
docstrange-api --port 9000 --workers 2 --no-gpu
```

- REST API for external applications
- Interactive docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

### Test with UI

For testing with a web interface, use the existing Flask UI:

```bash
docstrange-web
```

## Command Options

```
--host HOST             Bind host (default: 0.0.0.0)
--port PORT             Port number (default: 8000)
--workers NUM           Worker processes (default: 4)
--no-gpu                Use cloud mode instead of GPU
```

## Python Client

```python
from docstrange.api_client import DocStrangeClient

async with DocStrangeClient("http://localhost:8000") as client:
    # Health check
    health = await client.health_check()

    # Simple extraction
    result = await client.extract_simple("document.pdf")
    print(result["markdown"])

    # Extract specific fields
    result = await client.extract(
        "invoice.pdf",
        specified_fields=["invoice_number", "total", "date"]
    )

    # Extract insurance documents (pre-configured)
    result = await client.extract_insurance_document("clearance.pdf")
    print(result["data"])
```

## API Endpoints

### `GET /health`
Health check with GPU status

### `POST /api/extract/simple`
Simple extraction returning markdown and JSON
- **file**: Document file (multipart/form-data)

### `POST /api/extract`
Advanced extraction with field specification
- **file**: Document file (multipart/form-data)
- **specified_fields**: Comma-separated field names (optional)
- **json_schema**: JSON schema string (optional)
- **output_format**: json|markdown|csv|html|all

## Examples

See [examples/api_server_example.py](../examples/api_server_example.py)

## Integration with Agent Steve

```python
from docstrange.api_client import DocStrangeClient
import os

class DocumentProcessor:
    def __init__(self):
        self.client = DocStrangeClient(
            base_url=os.getenv("DOCSTRANGE_API_URL", "http://localhost:8000")
        )

    async def process_pdf(self, pdf_path: str):
        result = await self.client.extract_insurance_document(pdf_path)
        if result["success"]:
            return result["data"]

    async def cleanup(self):
        await self.client.close()
```

## Deployment

### Docker

```dockerfile
FROM python:3.10
COPY . /app
WORKDIR /app
RUN pip install -e ".[api]"
EXPOSE 8000
CMD ["docstrange-api", "--mode", "api"]
```

### Systemd

```ini
[Service]
ExecStart=/usr/local/bin/docstrange-api --mode api --port 8000 --workers 4
Restart=always
```

### Environment

```bash
export DOCSTRANGE_GPU_MODE=true  # or false for cloud mode
```

## Performance

### Worker Configuration

Match workers to GPU memory:
- 8GB GPU: 1-2 workers
- 16GB GPU: 2-4 workers
- 24GB+ GPU: 4-8 workers

```bash
docstrange-api --mode api --workers 2
```

### Monitor GPU

```bash
nvidia-smi -l 1
```

## Testing

```bash
# Health check
curl http://localhost:8000/health

# Extract document
curl -X POST "http://localhost:8000/api/extract/simple" \
  -F "file=@document.pdf"
```

## Troubleshooting

**GPU not detected?**
```bash
nvidia-smi  # Check GPU
python3 -c "import torch; print(torch.cuda.is_available())"
```

**Out of memory?**
```bash
docstrange-api --mode api --workers 1
```

**Port in use?**
```bash
lsof -i :8000
docstrange-api --mode api --port 9000
```
