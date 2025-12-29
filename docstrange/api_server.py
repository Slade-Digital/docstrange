"""
FastAPI server for DocStrange document extraction.
Designed to run on vGPU and serve external applications like Agent Steve.

For testing with a UI, use the existing web interface:
  docstrange-web
"""
import argparse
import json
import os
import subprocess
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .extractor import DocumentExtractor

# Initialize FastAPI app
app = FastAPI(
    title="DocStrange Extraction API",
    description="GPU-accelerated document extraction service",
    version="1.1.8"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def check_gpu_availability() -> bool:
    """Check if GPU is available for processing."""
    try:
        return torch.cuda.is_available()
    except ImportError:
        return False


# Pydantic models
class ExtractionRequest(BaseModel):
    """Request model for extraction with specific fields"""
    specified_fields: Optional[List[str]] = None
    json_schema: Optional[Dict[str, str]] = None
    output_format: str = "json"


class ExtractionResponse(BaseModel):
    """Response model for extraction results"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    markdown: Optional[str] = None
    json_data: Optional[str] = None
    csv_data: Optional[str] = None
    html_data: Optional[str] = None
    error: Optional[str] = None
    processing_time_ms: Optional[float] = None
    filename: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    gpu_available: bool
    mode: str
    timestamp: str


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize extractor on startup"""
    gpu_mode = os.getenv("DOCSTRANGE_GPU_MODE", "true").lower() == "true"

    if gpu_mode and not check_gpu_availability():
        print("⚠️ GPU mode requested but GPU not available. Falling back to cloud mode.")
        gpu_mode = False

    app.state.extractor = DocumentExtractor(gpu=gpu_mode)
    mode_str = "GPU" if gpu_mode else "Cloud"
    print(f"✅ DocumentExtractor initialized with {mode_str} mode")


@app.get("/", response_model=Dict[str, str])
async def root() -> Dict[str, str]:
    """Root endpoint with API info"""
    return {
        "service": "DocStrange Extraction API",
        "version": "1.1.8",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint for monitoring"""
    gpu_available = check_gpu_availability()
    extractor = getattr(app.state, "extractor", None)

    return HealthResponse(
        status="healthy" if extractor is not None else "unhealthy",
        gpu_available=gpu_available,
        mode="gpu" if (extractor and extractor.gpu) else "cloud",
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/api/extract", response_model=ExtractionResponse)
async def extract_document(
    file: UploadFile = File(...),
    specified_fields: Optional[str] = Form(None),
    json_schema: Optional[str] = Form(None),
    output_format: str = Form("json")
) -> ExtractionResponse:
    """
    Extract data from uploaded document.

    Args:
        file: The document file (PDF, DOCX, XLSX, PPTX, images, etc.)
        specified_fields: Comma-separated list of fields to extract (optional)
        json_schema: JSON schema string for structured extraction (optional)
        output_format: Desired output format (json, markdown, csv, html, all)

    Returns:
        ExtractionResponse with extracted data
    """
    extractor = getattr(app.state, "extractor", None)
    if extractor is None:
        raise HTTPException(
            status_code=503,
            detail="DocStrange service is not available"
        )

    start_time = datetime.now()
    tmp_path = None

    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        print(f"Processing file: {file.filename}")

        # Save uploaded file to temp location
        file_ext = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Extract with DocStrange
        result = extractor.extract(tmp_path)

        # Parse optional parameters
        fields_list = None
        if specified_fields:
            fields_list = [f.strip() for f in specified_fields.split(",")]

        schema_dict = None
        if json_schema:
            schema_dict = json.loads(json_schema)

        # Extract structured data
        structured_data = None
        if fields_list or schema_dict:
            structured_data = result.extract_data(
                specified_fields=fields_list,
                json_schema=schema_dict
            )

        # Get different output formats
        response_data = {
            "success": True,
            "filename": file.filename,
            "processing_time_ms": (datetime.now() - start_time).total_seconds() * 1000
        }

        if output_format == "json" or structured_data:
            response_data["data"] = structured_data
            response_data["json_data"] = json.dumps(result.extract_data())

        if output_format == "markdown" or output_format == "all":
            response_data["markdown"] = result.extract_markdown()

        if output_format == "csv" or output_format == "all":
            response_data["csv_data"] = result.extract_csv()

        if output_format == "html" or output_format == "all":
            response_data["html_data"] = result.extract_html()

        processing_time = response_data["processing_time_ms"]
        print(f"✅ Successfully processed {file.filename} in {processing_time:.2f}ms")

        return ExtractionResponse(**response_data)

    except Exception as e: # pylint: disable=broad-except
        error_msg = "Extraction failed: " + str(e)
        print(f"❌ {error_msg}")

        return ExtractionResponse(
            success=False,
            error=error_msg,
            filename=file.filename if file else None,
            processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000
        )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception as e: # pylint: disable=broad-except
                print(f"⚠️ Failed to cleanup temp file {tmp_path}: {e}")


@app.post("/api/extract/simple")
async def extract_simple(file: UploadFile = File(...)) -> JSONResponse:
    """
    Simplified extraction endpoint - returns JSON and Markdown only.
    Designed for quick integration with external applications.

    Args:
        file: The document file

    Returns:
        Simple JSON response with extracted data
    """
    extractor = getattr(app.state, "extractor", None)
    if extractor is None:
        raise HTTPException(
            status_code=503,
            detail="DocStrange service is not available"
        )

    tmp_path = None

    try:
        file_ext = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        result = extractor.extract(tmp_path)

        return JSONResponse({
            "success": True,
            "markdown": result.extract_markdown(),
            "json": json.dumps(result.extract_data()),
            "filename": file.filename
        })

    except Exception as e: # pylint: disable=broad-except
        print(f"❌ Simple extraction failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "filename": file.filename if file else None
            }
        )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception: # pylint: disable=broad-except
                pass


def run_api_server(host: str = "0.0.0.0", port: int = 8000, workers: int = 4) -> None:
    """Run FastAPI server in production mode"""
    import uvicorn

    print(f"🚀 Starting DocStrange API server at http://{host}:{port}")
    print(f"   Workers: {workers}")
    print(f"   GPU Mode: {os.getenv('DOCSTRANGE_GPU_MODE', 'true')}")

    # Check GPU
    if check_gpu_availability():
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                print(f"   GPU detected: {result.stdout.strip()}")
        except FileNotFoundError:
            print("   ⚠️ nvidia-smi not found")
    else:
        print("   ⚠️ No GPU detected. Service will run in cloud mode.")

    uvicorn.run(
        "docstrange.api_server:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info"
    )


def main() -> None:
    """Main entry point for API server"""
    parser = argparse.ArgumentParser(description="DocStrange API Server")
    parser.add_argument(
            "--host",
            default="0.0.0.0",
            help="Host to bind to"
        )
    parser.add_argument(
            "--port",
            type=int,
            default=8000,
            help="Port to bind to (default: 8000)"
        )
    parser.add_argument(
            "--workers",
            type=int,
            default=4,
            help="Number of worker processes"
        )
    parser.add_argument(
            "--no-gpu",
            action="store_true",
            help="Disable GPU mode, use cloud mode"
        )

    args = parser.parse_args()

    # Set GPU mode environment variable
    if args.no_gpu:
        os.environ["DOCSTRANGE_GPU_MODE"] = "false"
    else:
        os.environ["DOCSTRANGE_GPU_MODE"] = "true"

    run_api_server(host=args.host, port=args.port, workers=args.workers)


if __name__ == "__main__":
    main()
