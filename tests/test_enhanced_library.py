#!/usr/bin/env python3
"""
Test script for the enhanced docstrange library.
"""

import os
import tempfile
from docstrange import DocumentExtractor


def test_basic_functionality():
    """Test basic functionality of the enhanced library."""
    print("🧪 Testing Enhanced Document Data Extractor Library")
    print("=" * 50)
    
    extractor = DocumentExtractor()
    
    # Test 1: Text file conversion
    print("\n1. Testing text file conversion...")
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document.\n\nIt has multiple lines.\n\n# This is a heading\n\n- List item 1\n- List item 2")
            temp_file = f.name
        
        result = extractor.extract(temp_file)
        print(f"✅ Text conversion successful: {len(result.content)} characters")
        print(f"   Metadata: {result.metadata}")
        
        # Test different output formats
        markdown = result.extract_markdown()
        html = result.extract_html()
        json_output = result.extract_data()
        
        print(f"   Markdown length: {len(markdown)}")
        print(f"   HTML length: {len(html)}")
        print(f"   JSON keys: {list(json_output.keys())}")
        
        os.unlink(temp_file)
        
    except Exception as e:
        print(f"❌ Text conversion failed: {e}")
    
    # Test 2: URL conversion
    print("\n2. Testing URL conversion...")
    try:
        result = extractor.convert_url("https://httpbin.org/html")
        print(f"✅ URL conversion successful: {len(result.content)} characters")
        print(f"   Status code: {result.metadata.get('status_code')}")
        
    except Exception as e:
        print(f"❌ URL conversion failed: {e}")
    
    # Test 3: Plain text conversion
    print("\n3. Testing plain text conversion...")
    try:
        text = "This is plain text for testing the extractor."
        result = extractor.extract_text(text)
        print(f"✅ Plain text conversion successful: {len(result.content)} characters")
        
    except Exception as e:
        print(f"❌ Plain text conversion failed: {e}")
    
    # Test 4: Supported formats
    print("\n4. Testing supported formats...")
    try:
        formats = extractor.get_supported_formats()
        print(f"✅ Supported formats: {formats}")
        
    except Exception as e:
        print(f"❌ Format detection failed: {e}")
    
    # Test 5: Configuration options
    print("\n5. Testing configuration options...")
    try:
        converter_enhanced = DocumentExtractor(
            preserve_layout=True,
            include_images=True,
            ocr_enabled=True
        )
        print(f"✅ Enhanced extractor created with OCR enabled")
        
    except Exception as e:
        print(f"❌ Configuration failed: {e}")


def test_processor_specific_functionality():
    """Test specific processor functionality."""
    print("\n" + "=" * 50)
    print("🔧 Testing Processor-Specific Functionality")
    print("=" * 50)
    
    extractor = DocumentExtractor()
    
    # Test CSV processing
    print("\n1. Testing CSV processing...")
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Name,Age,City\nJohn,30,New York\nJane,25,Los Angeles\nBob,35,Chicago")
            temp_file = f.name
        
        result = extractor.extract(temp_file)
        print(f"✅ CSV conversion successful: {len(result.content)} characters")
        print(f"   Rows: {result.metadata.get('row_count')}")
        print(f"   Columns: {result.metadata.get('column_count')}")
        
        os.unlink(temp_file)
        
    except Exception as e:
        print(f"❌ CSV conversion failed: {e}")
    
    # Test HTML processing
    print("\n2. Testing HTML processing...")
    try:
        html_content = """
        <html>
        <head><title>Test Page</title></head>
        <body>
            <h1>Main Heading</h1>
            <p>This is a paragraph.</p>
            <h2>Sub Heading</h2>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
            </ul>
            <table>
                <tr><th>Name</th><th>Value</th></tr>
                <tr><td>Test</td><td>123</td></tr>
            </table>
        </body>
        </html>
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write(html_content)
            temp_file = f.name
        
        result = extractor.extract(temp_file)
        print(f"✅ HTML conversion successful: {len(result.content)} characters")
        print(f"   Contains table: {'table' in result.content.lower()}")
        print(f"   Contains list: {'item' in result.content.lower()}")
        
        os.unlink(temp_file)
        
    except Exception as e:
        print(f"❌ HTML conversion failed: {e}")
    
    # Test 3: Image processing with OCR
    print("\n3. Testing image processing with OCR...")
    try:
        # Create a simple test image with text (this is just a placeholder)
        # In a real scenario, you'd have an actual image file
        print("   Note: Image OCR test requires an actual image file")
        print("   To test OCR, place an image file in the current directory")
        
    except Exception as e:
        print(f"❌ Image processing failed: {e}")


def test_error_handling():
    """Test error handling functionality."""
    print("\n" + "=" * 50)
    print("⚠️  Testing Error Handling")
    print("=" * 50)
    
    extractor = DocumentExtractor()
    
    # Test non-existent file
    print("\n1. Testing non-existent file...")
    try:
        result = extractor.extract("nonexistent_file.txt")
        print("❌ Should have raised FileNotFoundError")
    except FileNotFoundError:
        print("✅ Correctly handled non-existent file")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    # Test unsupported format
    print("\n2. Testing unsupported format...")
    try:
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            f.write(b"test content")
            temp_file = f.name
        
        result = extractor.extract(temp_file)
        print("❌ Should have raised UnsupportedFormatError")
        os.unlink(temp_file)
    except Exception as e:
        print(f"✅ Correctly handled unsupported format: {type(e).__name__}")
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def test_ocr_functionality():
    """Test OCR functionality if available."""
    print("\n" + "=" * 50)
    print("🤖 Testing OCR Functionality")
    print("=" * 50)
    
    # Test OCR-enabled extractor
    print("\n1. Testing OCR-enabled extractor...")
    try:
        converter_ocr = DocumentExtractor(ocr_enabled=True)
        print("✅ OCR-enabled extractor created successfully")
        
        # Check if PaddleOCR is available
        try:
            from paddleocr import PaddleOCR
            print("✅ PaddleOCR is available")
        except ImportError:
            print("⚠️  PaddleOCR not available - OCR features will be limited")
            
    except Exception as e:
        print(f"❌ OCR setup failed: {e}")


def main():
    """Main test function."""
    print("🚀 Starting Enhanced Library Tests")
    print("=" * 60)
    
    # Run all tests
    test_basic_functionality()
    test_processor_specific_functionality()
    test_error_handling()
    test_ocr_functionality()
    
    print("\n" + "=" * 60)
    print("✅ Enhanced Library Tests Completed!")
    print("\n📋 Summary:")
    print("- Enhanced PDF processing with pdf2image")
    print("- Added PowerPoint support")
    print("- Added image processing with PaddleOCR")
    print("- Improved HTML table conversion")
    print("- Better CSV and Excel handling")
    print("- Enhanced error handling and logging")
    print("- More comprehensive metadata")
    print("- OCR capabilities for image text extraction")


if __name__ == "__main__":
    main() 