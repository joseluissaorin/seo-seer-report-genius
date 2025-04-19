# SEO Seer: Advanced SEO Analysis Tool

SEO Seer is a powerful web application that transforms Google Search Console data into comprehensive SEO insights using AI-powered analysis. Get enterprise-level SEO analytics at a fraction of the cost.

## Key Features

### 🔍 Advanced Analysis
- **SEO Health Score**: Comprehensive scoring system for overall SEO performance
- **Mobile Optimization**: Detailed mobile-friendliness analysis and recommendations
- **Keyword Analysis**: Advanced keyword research and cannibalization detection
- **SERP Features**: Track featured snippets, knowledge panels, and rich results
- **Competitor Intelligence**: Compare your content against competitors
- **Backlink Analysis**: Monitor your backlink profile and domain authority

### 📊 Data Processing
- **CSV Upload**: Easy import of Google Search Console exports
- **Automatic Language Detection**: Supports both English and Spanish GSC exports
- **Real-time Processing**: See analysis progress as your data is processed
- **Privacy-First**: All data is processed locally and securely

### 📈 SEO Analytics
- **Content Gap Analysis**: Identify missing content opportunities
- **Interactive Visualizations**: Dynamic charts and performance metrics
- **Statistical Analysis**: Advanced data modeling and trend prediction
- **Geographic Insights**: Regional performance breakdown
- **Device Analysis**: Cross-device performance tracking

### 📑 Report Generation
- **PDF Reports**: Downloadable, comprehensive analysis
- **SERP Preview**: Visual search result previews
- **Action Items**: Prioritized recommendations
- **Custom Branding**: Professional, branded reports

## Quick Start

1. **Installation**:
   ```bash
   git clone https://github.com/joseluissaorin/seo-seer-report-genius.git
   cd seo-seer-report-genius
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

3. Open http://localhost:4567 in your browser

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

We welcome contributions! Please read our contributing guidelines and submit pull requests to our [GitHub repository](https://github.com/joseluissaorin/seo-seer-report-genius).

## Security

- API keys are never stored, only used for processing
- All data processing happens locally
- No data is sent to external servers except for AI analysis

## License

[MIT License](LICENSE) - Copyright (c) 2025 José Luis Saorín Ferrer

## Support

For support, please open an issue in the [GitHub repository](https://github.com/joseluissaorin/seo-seer-report-genius) or contact our support team.
