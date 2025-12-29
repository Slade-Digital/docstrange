# DocStrange Web Interface

A beautiful, modern web interface for the DocStrange document extraction library, inspired by the data-extraction-apis project design.

## Features

- **Modern UI**: Clean, responsive design with drag-and-drop file upload
- **Multiple Formats**: Support for PDF, Word, Excel, PowerPoint, images, and more
- **Output Options**: Convert to Markdown, HTML, JSON, CSV, or Flat JSON
- **Real-time Processing**: Live extraction with progress indicators
- **Download Results**: Save extracted content in various formats
- **Mobile Friendly**: Responsive design that works on all devices

## Quick Start

### 1. Install Dependencies

```bash
pip install docstrange[web]
```

### 2. Start the Web Interface

```bash
docstrange web
```

### 3. Open Your Browser

Navigate to: http://localhost:8000

## Usage

### File Upload

1. **Drag & Drop**: Simply drag your file onto the upload area
2. **Click to Browse**: Click the upload area to select a file from your computer
3. **Supported Formats**: PDF, Word (.docx, .doc), Excel (.xlsx, .xls), PowerPoint (.pptx, .ppt), HTML, CSV, Text, Images (PNG, JPG, TIFF, BMP)

### Output Format Selection

Choose from multiple output formats:

- **Markdown**: Clean, structured markdown text
- **HTML**: Formatted HTML with styling
- **JSON**: Structured JSON data
- **CSV**: Table data in CSV format
- **Flat JSON**: Simplified JSON structure

### Results View

After processing, you can:

- **Preview**: View formatted content in the preview tab
- **Raw Output**: See the raw extracted text
- **Download**: Save results as text or JSON files

## API Endpoints

The web interface also provides REST API endpoints:

### Health Check
```
GET /api/health
```

### Get Supported Formats
```
GET /api/supported-formats
```

### Extract Document
```
POST /api/extract
Content-Type: multipart/form-data

Parameters:
- file: The document file to extract
- output_format: markdown, html, json, csv, flat-json
```

## Configuration

### Environment Variables

- `FLASK_ENV`: Set to `development` for debug mode
- `MAX_CONTENT_LENGTH`: Maximum file size (default: 100MB)

### Customization

The web interface uses a modular design system:

- **CSS Variables**: Easy theming via CSS custom properties
- **Responsive Design**: Mobile-first approach
- **Component-based**: Reusable UI components

## Development

### Running in Development Mode

```bash
# Install development dependencies
pip install -e .

# Start with debug mode
python -m docstrange.web_app
```

### File Structure

```
docstrange/
├── web_app.py          # Flask application
├── templates/
│   └── index.html      # Main HTML template
└── static/
    ├── styles.css      # Design system CSS
    └── script.js       # Frontend JavaScript
```

### Testing

```bash
# Run the test script
python test_web_interface.py
```

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Use a different port
   docstrange web --port 8080
   ```

2. **File Upload Fails**
   - Check file size (max 100MB)
   - Verify file format is supported
   - Ensure proper file permissions

3. **Extraction Errors**
   - Check console logs for detailed error messages
   - Verify document is not corrupted
   - Try different output formats

### Logs

The web interface logs to the console. Check for:
- File upload events
- Processing status
- Error messages
- API request details

## Contributing

To contribute to the web interface:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This web interface is part of the DocStrange project and is licensed under the MIT License. 