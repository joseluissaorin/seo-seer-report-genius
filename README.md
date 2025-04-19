
# SEO Seer: Advanced SEO Analysis Tool

SEO Seer is a powerful web application that transforms Google Search Console data into comprehensive SEO insights using AI-powered analysis. Get enterprise-level SEO analytics at a fraction of the cost.

## Key Features

### 🔍 Advanced Analysis
- **AI-Powered Insights**: Leverages Gemini AI for intelligent recommendations
- **Comprehensive Reports**: Get detailed PDF reports with actionable insights
- **Multi-Device Analysis**: Understand performance across desktop, mobile, and tablet
- **Geographic Insights**: Analyze performance by country and region

### 📊 Data Processing
- **CSV Upload**: Easy import of Google Search Console exports
- **Automatic Language Detection**: Supports both English and Spanish GSC exports
- **Real-time Processing**: See analysis progress as your data is processed
- **Privacy-First**: All data is processed locally and securely

### 📈 SEO Analytics
- **Keyword Research**: Discover new keyword opportunities
- **Competitor Analysis**: Understand your market position
- **Device Performance**: Track mobile vs desktop trends
- **Geographic Analysis**: Identify regional opportunities
- **Temporal Trends**: Analyze performance over time

### 📑 Report Generation
- **PDF Reports**: Downloadable, comprehensive analysis
- **Data Visualization**: Clear charts and graphs
- **Actionable Insights**: Step-by-step recommendations
- **Custom Branding**: Professional, branded reports

## Quick Start

1. **Installation**:
   ```bash
   git clone https://github.com/yourusername/seo-seer.git
   cd seo-seer
   npm install
   ```

2. **Start Development Servers**:
   ```bash
   # Frontend
   npm run dev

   # Backend (in new terminal)
   cd api
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   python run_server.py
   ```

3. Open http://localhost:8080 in your browser

## Documentation

For detailed setup instructions, including reverse proxy configuration and advanced usage, see our [Setup Documentation](docs/SETUP.md).

## Requirements

- Node.js 18+
- Python 3.10+
- Gemini API key (for AI analysis)

## Technology Stack

### Frontend
- React with TypeScript
- Vite for build tooling
- TailwindCSS for styling
- shadcn/ui components
- Recharts for data visualization

### Backend
- FastAPI
- Pandas for data processing
- Matplotlib for visualization
- Google Generative AI (Gemini)
- ReportLab for PDF generation

## Contributing

We welcome contributions! Please read our contributing guidelines and submit pull requests to our repository.

## Security

- API keys are never stored, only used for processing
- All data processing happens locally
- No data is sent to external servers except for AI analysis

## License

MIT License - see LICENSE file for details

## Support

For support, please open an issue in the GitHub repository or contact our support team.
