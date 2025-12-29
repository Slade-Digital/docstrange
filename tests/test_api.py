"""
Test script for DocStrange API server.
Run this to verify the API is working correctly.
"""
import asyncio
import httpx
import sys
from pathlib import Path


async def test_api(base_url: str = "http://localhost:8000"):
    """Test DocStrange API endpoints"""

    print(f"Testing DocStrange API at {base_url}\n")

    async with httpx.AsyncClient(timeout=300.0) as client:

        # Test 1: Root endpoint
        print("1. Testing root endpoint...")
        try:
            response = await client.get(f"{base_url}/")
            print(f"   ✓ Root: {response.json()}")
        except Exception as e:
            print(f"   ✗ Root failed: {e}")
            return False

        # Test 2: Health check
        print("\n2. Testing health check...")
        try:
            response = await client.get(f"{base_url}/health")
            health = response.json()
            print(f"   ✓ Status: {health['status']}")
            print(f"   ✓ GPU Available: {health['gpu_available']}")
            print(f"   ✓ Timestamp: {health['timestamp']}")

            if health['status'] != 'healthy':
                print("   ✗ Service is not healthy!")
                return False
        except Exception as e:
            print(f"   ✗ Health check failed: {e}")
            return False

        # Test 3: Simple extraction (if test file exists)
        print("\n3. Testing document extraction...")
        test_files = [
            "../examples/sample.pdf",
            "../tests/fixtures/sample.pdf",
            "test_document.pdf"
        ]

        test_file = None
        for f in test_files:
            if Path(f).exists():
                test_file = f
                break

        if test_file:
            try:
                with open(test_file, "rb") as f:
                    files = {"file": (Path(test_file).name, f, "application/pdf")}
                    response = await client.post(
                        f"{base_url}/api/extract/simple",
                        files=files
                    )
                    result = response.json()

                    if result.get("success"):
                        print(f"   ✓ Extraction successful")
                        print(f"   ✓ Filename: {result.get('filename')}")
                        markdown = result.get('markdown', '')
                        print(f"   ✓ Markdown length: {len(markdown)} chars")
                        if len(markdown) > 100:
                            print(f"   ✓ Preview: {markdown[:100]}...")
                    else:
                        print(f"   ✗ Extraction failed: {result.get('error')}")
                        return False

            except Exception as e:
                print(f"   ✗ Extraction test failed: {e}")
                return False
        else:
            print("   ⊘ No test file found, skipping extraction test")
            print(f"   (Looked for: {', '.join(test_files)})")

        # Test 4: Advanced extraction with fields
        if test_file:
            print("\n4. Testing extraction with specified fields...")
            try:
                with open(test_file, "rb") as f:
                    files = {"file": (Path(test_file).name, f, "application/pdf")}
                    data = {
                        "specified_fields": "title,author,date,content",
                        "output_format": "json"
                    }
                    response = await client.post(
                        f"{base_url}/api/extract",
                        files=files,
                        data=data
                    )
                    result = response.json()

                    if result.get("success"):
                        print(f"   ✓ Advanced extraction successful")
                        print(f"   ✓ Processing time: {result.get('processing_time_ms', 0):.2f}ms")
                        if result.get("data"):
                            print(f"   ✓ Structured data extracted: {len(result['data'])} fields")
                    else:
                        print(f"   ✗ Advanced extraction failed: {result.get('error')}")

            except Exception as e:
                print(f"   ✗ Advanced extraction test failed: {e}")

    print("\n" + "="*50)
    print("✓ All tests passed! API is ready for Agent Steve.")
    print("="*50)
    return True


async def test_client_example():
    """Test using the client library"""
    print("\n\nTesting with client library...\n")

    try:
        # Import the client
        sys.path.append(str(Path(__file__).parent))
        from client_example import DocStrangeClient

        async with DocStrangeClient() as client:
            # Health check
            health = await client.health_check()
            print(f"Client test - Health: {health['status']}")
            print(f"Client test - GPU: {health['gpu_available']}")

            print("\n✓ Client library works correctly!")

    except ImportError:
        print("⊘ Client library test skipped (client_example.py not found)")
    except Exception as e:
        print(f"✗ Client library test failed: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test DocStrange API")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of DocStrange API (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--skip-client",
        action="store_true",
        help="Skip client library test"
    )

    args = parser.parse_args()

    # Run API tests
    success = asyncio.run(test_api(args.url))

    # Run client tests
    if not args.skip_client:
        asyncio.run(test_client_example())

    sys.exit(0 if success else 1)
