"""
Python client for DocStrange API server.
Use this in external applications like Agent Steve to interact with DocStrange API.
"""
import json
from typing import Optional, List, Dict, Any
from pathlib import Path
import httpx

class DocStrangeClient:
    """
    Async client for DocStrange API.

    Example:
        async with DocStrangeClient(base_url="http://vgpu-server:8000") as client:
            result = await client.extract_simple("document.pdf")
            print(result["markdown"])
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 300.0,
        api_key: Optional[str] = None
    ):
        """
        Initialize DocStrange API client.

        Args:
            base_url: Base URL of DocStrange API service
            timeout: Request timeout in seconds (default: 5 minutes)
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        headers = {}
        if api_key:
            headers["X-API-Key"] = api_key

        self.client = httpx.AsyncClient(timeout=timeout, headers=headers)

    async def health_check(self) -> Dict[str, Any]:
        """
        Check if DocStrange API is healthy and GPU is available.

        Returns:
            Dict with status, gpu_available, mode, timestamp

        Example:
            health = await client.health_check()
            if health["gpu_available"]:
                print("GPU is available!")
        """
        response = await self.client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    async def extract_simple(self, file_path: str) -> Dict[str, Any]:
        """
        Simple extraction - returns markdown and JSON.

        Args:
            file_path: Path to document file

        Returns:
            Dict with 'success', 'markdown', 'json', 'filename' keys

        Example:
            result = await client.extract_simple("invoice.pdf")
            if result["success"]:
                print(result["markdown"])
        """
        with open(file_path, "rb") as f:
            files = {"file": (Path(file_path).name, f, "application/octet-stream")}
            response = await self.client.post(
                f"{self.base_url}/api/extract/simple",
                files=files
            )
            response.raise_for_status()
            return response.json()

    async def extract(
        self,
        file_path: str,
        specified_fields: Optional[List[str]] = None,
        json_schema: Optional[Dict[str, str]] = None,
        output_format: str = "json"
    ) -> Dict[str, Any]:
        """
        Full extraction with field specification and schema support.

        Args:
            file_path: Path to document file
            specified_fields: List of specific fields to extract
            json_schema: JSON schema for structured extraction
            output_format: Output format (json, markdown, csv, html, all)

        Returns:
            Dict with extraction results

        Example:
            # Extract specific fields
            result = await client.extract(
                "insurance.pdf",
                specified_fields=["patient_name", "policy_number", "date"]
            )

            # Extract with schema
            result = await client.extract(
                "invoice.pdf",
                json_schema={
                    "invoice_number": "string",
                    "total_amount": "number",
                    "date": "date"
                }
            )
        """
        data = {"output_format": output_format}

        if specified_fields:
            data["specified_fields"] = ",".join(specified_fields)

        if json_schema:
            data["json_schema"] = json.dumps(json_schema)

        with open(file_path, "rb") as f:
            files = {"file": (Path(file_path).name, f, "application/octet-stream")}
            response = await self.client.post(
                f"{self.base_url}/api/extract",
                files=files,
                data=data
            )
            response.raise_for_status()
            return response.json()

    async def extract_insurance_document(self, file_path: str) -> Dict[str, Any]:
        """
        Convenience method for extracting insurance clearance documents.
        Pre-configured with common insurance fields.

        Args:
            file_path: Path to insurance PDF

        Returns:
            Dict with extracted insurance data

        Example:
            result = await client.extract_insurance_document("clearance.pdf")
            if result["success"]:
                data = result["data"]
                print(f"Patient: {data.get('patient_name')}")
                print(f"Policy: {data.get('policy_number')}")
        """
        insurance_fields = [
            "patient_name",
            "patient_dob",
            "patient_id",
            "insurance_provider",
            "insurance_company",
            "policy_number",
            "group_number",
            "member_id",
            "clearance_date",
            "authorization_date",
            "authorization_number",
            "effective_date",
            "expiration_date",
            "diagnosis_codes",
            "procedure_codes",
            "provider_name",
            "provider_npi",
            "status",
            "notes"
        ]

        return await self.extract(
            file_path,
            specified_fields=insurance_fields,
            output_format="json"
        )

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
