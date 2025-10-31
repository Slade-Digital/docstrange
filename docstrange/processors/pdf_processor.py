"""PDF file processor with OCR support for scanned PDFs."""

import os
import logging
import tempfile
from typing import Dict, Any, List, Tuple

from .base import BaseProcessor
from .image_processor import ImageProcessor
from ..result import ConversionResult
from ..exceptions import ConversionError, FileNotFoundError
from ..config import InternalConfig
from ..pipeline.ocr_service import OCRServiceFactory, NeuralOCRService

# Configure logging
logger = logging.getLogger(__name__)


class PDFProcessor(BaseProcessor):
    """Processor for PDF files using PDF-to-image conversion with OCR."""
    
    def __init__(self, preserve_layout: bool = True, include_images: bool = False, ocr_enabled: bool = True, use_markdownify: bool = None):
        super().__init__(preserve_layout, include_images, ocr_enabled, use_markdownify)
        # Create a shared OCR service instance for all pages
        shared_ocr_service = NeuralOCRService()
        self._image_processor = ImageProcessor(
            preserve_layout=preserve_layout,
            include_images=include_images,
            ocr_enabled=ocr_enabled,
            use_markdownify=use_markdownify,
            ocr_service=shared_ocr_service
        )
    
    def can_process(self, file_path: str) -> bool:
        """Check if this processor can handle the given file.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if this processor can handle the file
        """
        if not os.path.exists(file_path):
            return False
        
        # Check file extension - ensure file_path is a string
        file_path_str = str(file_path)
        _, ext = os.path.splitext(file_path_str.lower())
        return ext == '.pdf'
    
    def process(self, file_path: str) -> ConversionResult:
        """Process PDF file with OCR capabilities.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            ConversionResult with extracted content
        """
        try:
            from ..config import InternalConfig
            pdf_to_image_enabled = InternalConfig.pdf_to_image_enabled
        except (ImportError, AttributeError):
            # Fallback if config is not available
            pdf_to_image_enabled = True
            logger.warning("InternalConfig not available, defaulting to pdf_to_image_enabled = True")
        
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"PDF file not found: {file_path}")
            
            logger.info(f"Processing PDF file: {file_path}")
            logger.info(f"pdf_to_image_enabled = {pdf_to_image_enabled}")
            
            # Always use OCR-based processing (pdf2image + OCR)
            logger.info("Using OCR-based PDF processing with pdf2image")
            return self._process_with_ocr(file_path)
            
        except Exception as e:
            logger.error(f"Failed to process PDF file {file_path}: {e}")
            raise ConversionError(f"PDF processing failed: {e}")
    
    def _process_with_ocr(self, file_path: str) -> ConversionResult:
        """Process PDF using OCR after converting pages to images."""
        try:
            from pdf2image import convert_from_path
            from ..config import InternalConfig
            
            # Get DPI from config
            dpi = getattr(InternalConfig, 'pdf_image_dpi', 300)
            
            # Convert PDF pages to images using pdf2image
            images = convert_from_path(file_path, dpi=dpi)
            page_count = len(images)
            all_content = []
            
            for page_num, image in enumerate(images):
                # Save to temporary file for OCR processing
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    image.save(tmp.name, 'PNG')
                    temp_image_path = tmp.name
                
                try:
                    # Process the page image
                    page_result = self._image_processor.process(temp_image_path)
                    page_content = page_result.content
                    
                    if page_content.strip():
                        all_content.append(f"## Page {page_num + 1}\n\n{page_content}")
                    
                finally:
                    # Clean up temporary file
                    os.unlink(temp_image_path)
            
            content = "\n\n".join(all_content) if all_content else "No content extracted from PDF"
            
            return ConversionResult(
                content=content,
                metadata={
                    'file_path': file_path,
                    'file_type': 'pdf',
                    'pages': page_count,
                    'extraction_method': 'ocr'
                }
            )
            
        except ImportError:
            logger.error("pdf2image not available. Please install it: pip install pdf2image")
            raise ConversionError("pdf2image is required for PDF processing")
        except Exception as e:
            logger.error(f"OCR-based PDF processing failed: {e}")
            raise ConversionError(f"OCR-based PDF processing failed: {e}")
    
    def _convert_page_to_image(self, pdf_path: str, page_num: int) -> str:
        """Convert a PDF page to an image file.
        
        Args:
            pdf_path: Path to the PDF file
            page_num: Page number (0-based)
            
        Returns:
            Path to the temporary image file
        """
        try:
            from pdf2image import convert_from_path
            from ..config import InternalConfig
            
            # Use configuration for image quality
            dpi = getattr(InternalConfig, 'pdf_image_dpi', 300)
            
            # Convert specific page to image
            images = convert_from_path(pdf_path, dpi=dpi, first_page=page_num + 1, last_page=page_num + 1)
            
            if not images:
                logger.error(f"Failed to extract page {page_num + 1} to image")
                return None
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                images[0].save(tmp_file.name, 'PNG')
                logger.debug(f"Page {page_num + 1} converted to image: {tmp_file.name}")
                return tmp_file.name
                
        except Exception as e:
            logger.error(f"Failed to extract page {page_num + 1} to image: {e}")
            return None
    
    def _extract_ocr_text_from_result(self, result: ConversionResult) -> str:
        """Extract OCR text from ImageProcessor result.
        
        Args:
            result: ConversionResult from ImageProcessor
            
        Returns:
            Extracted OCR text
        """
        try:
            content = result.content
            
            # Look for OCR section in the content
            if "## Extracted Text (OCR)" in content:
                # Extract text after the OCR header
                parts = content.split("## Extracted Text (OCR)")
                if len(parts) > 1:
                    ocr_section = parts[1]
                    # Remove any remaining headers and clean up
                    lines = ocr_section.split('\n')
                    text_lines = []
                    in_ocr_text = False
                    
                    for line in lines:
                        if line.strip() == "":
                            continue
                        elif line.startswith("##"):
                            # Stop at next header
                            break
                        else:
                            text_lines.append(line)
                    
                    return '\n'.join(text_lines).strip()
            
            # If no OCR section found, return the full content
            return content
            
        except Exception as e:
            logger.error(f"Failed to extract OCR text from result: {e}")
            return ""
    
    def _format_page_content(self, text: str, page_num: int) -> str:
        """Format page content as markdown with enhanced structure.
        
        Args:
            text: Extracted text
            page_num: Page number
            
        Returns:
            Formatted markdown content
        """
        if not text.strip():
            return f"\n## Page {page_num}\n\n*This page appears to be empty or contains no extractable text.*\n"
        
        # The text from nanonets-ocr already has proper markdown structure
        # Just add page header
        content_parts = [f"## Page {page_num}"]
        content_parts.append("")
        content_parts.append(text)
        content_parts.append("")
        
        return '\n'.join(content_parts)
    
    @staticmethod
    def predownload_ocr_models():
        """Pre-download OCR models by running a dummy prediction."""
        try:
            # Use ImageProcessor's predownload method
            ImageProcessor.predownload_ocr_models()
        except Exception as e:
            print(f"Failed to pre-download OCR models: {e}") 