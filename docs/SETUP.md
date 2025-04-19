# SEO Seer: Complete Setup Documentation

## Table of Contents
1. System Requirements
2. Project Structure Overview
3. Frontend Setup
4. Backend Setup
5. Reverse Proxy Configuration (Nginx/Caddy)
6. Development Workflow
7. Troubleshooting Common Issues
8. Advanced Configuration Options

## 1. System Requirements

Before setting up SEO Seer, ensure your system meets these prerequisites:

- **Node.js**: v18.x or higher
- **Python**: v3.10 or higher
- **Package Managers**: npm (for frontend) and pip (for backend)
- **Disk Space**: At least 1GB free for dependencies and project files
- **Operating System**: Windows, macOS, or Linux
- **Memory**: Minimum 4GB RAM recommended for analysis operations

## 2. Project Structure Overview

```
seo-seer/
│
├── / (Root - Frontend React application)
│   ├── src/
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── pages/
│   │   └── utils/
│   ├── public/
│   ├── vite.config.ts
│   └── package.json
│
└── api/ (Backend Python application)
    ├── main.py
    ├── utils/
    │   ├── seo_health.py
    │   ├── keyword_research.py
    │   ├── content_gap_analyzer.py
    │   ├── serp_feature_analyzer.py
    │   └── backlink_analyzer.py
    ├── templates/
    ├── requirements.txt
    └── ...
```

## 3. Frontend Setup

### Installing Dependencies
```bash
cd seo-seer
npm install
```

### Environment Configuration
Create a `.env` file in the root directory:
```
VITE_API_URL=http://localhost:4568
```

### Starting the Development Server
```bash
npm run dev
```

The frontend will be available at http://localhost:4567

## 4. Backend Setup

### Creating a Python Virtual Environment
```bash
cd api
python -m venv venv

# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Installing Python Dependencies
```bash
pip install -r requirements.txt
```

### Starting the Backend Server
```bash
python run_server.py
```

The FastAPI backend will be available at http://localhost:4568

### Frontend Development Server
```bash
npm run dev
```

The frontend will be available at http://localhost:4567

### API Key Configuration
1. Get a Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Enter it in the SEO Seer interface when prompted

## 5. Reverse Proxy Configuration

### Using Nginx
```nginx
server {
    listen 80;
    server_name seo-seer.local;

    # Frontend
    location / {
        proxy_pass http://localhost:4567;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Backend API
    location /api/ {
        proxy_pass http://localhost:4568/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Using Caddy (Recommended)
Add the following to your Caddyfile:

```caddy
seo-seer.yourdomain.com {
    # Frontend application
    reverse_proxy localhost:4567

    # API endpoints
    handle_path /api/* {
        reverse_proxy localhost:4568
    }

    # Optional: Configure maximum upload size for CSV files
    request_body {
        max_size 10MB
    }
}
```

## 6. Development Workflow

1. Start both servers:
   ```bash
   # Terminal 1 (Frontend)
   npm run dev

   # Terminal 2 (Backend)
   cd api && source venv/bin/activate && python run_server.py
   ```

2. Access the application at http://localhost:4567
3. Upload GSC data files for analysis
4. Enter your Gemini API key when prompted
5. View generated reports and insights

## 7. Troubleshooting

### Common Issues and Solutions

1. **Port Conflicts**
   - Frontend port can be changed in `vite.config.ts`
   - Backend port can be changed in `run_server.py`

2. **File Upload Issues**
   - Check file size limits
   - Verify CSV format matches GSC export format
   - Ensure proper file permissions

3. **Analysis Errors**
   - Verify API key validity
   - Check CSV data format
   - Ensure sufficient system resources

4. **CORS Issues**
   - Configure backend CORS settings
   - Check frontend API URL configuration
   - Verify proxy settings if using reverse proxy

## 8. Advanced Features Configuration

### SEO Health Score
- Customize scoring weights in `seo_health.py`
- Adjust threshold values for different metrics
- Add custom scoring factors

### Mobile Optimization
- Configure mobile-friendliness criteria
- Adjust scoring thresholds
- Customize recommendation logic

### Keyword Analysis
- Set cannibalization detection thresholds
- Configure content gap analysis parameters
- Adjust keyword clustering settings

### SERP Features
- Customize feature detection rules
- Configure position tracking
- Adjust feature scoring weights

### Backlink Analysis
- Set domain authority thresholds
- Configure backlink quality metrics
- Adjust link scoring parameters

For more detailed configuration options, please refer to the specific feature documentation in our wiki.
