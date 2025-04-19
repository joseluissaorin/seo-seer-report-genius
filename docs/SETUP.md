
# SEO Seer: Complete Setup Documentation

## Table of Contents
1. System Requirements
2. Project Structure Overview
3. Frontend Setup
4. Backend Setup
5. Reverse Proxy Configuration
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

## 2. Project Structure Overview

SEO Seer follows a client-server architecture:

```
seo-seer/
│
├── / (Root - Frontend React application)
│   ├── src/
│   ├── public/
│   ├── vite.config.ts
│   ├── package.json
│   └── ...
│
└── api/ (Backend Python application)
    ├── main.py
    ├── utils/
    ├── templates/
    ├── requirements.txt
    └── ...
```

## 3. Frontend Setup

### Installing Dependencies

```bash
# Navigate to the project root directory
cd seo-seer
npm install
```

### Starting the Development Server

```bash
npm run dev
```

The frontend will be available at http://localhost:8080

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

Or with uvicorn:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The FastAPI backend will be available at http://localhost:8000

## 5. Reverse Proxy Configuration

### Using Nginx

```nginx
server {
    listen 80;
    server_name seo-seer.local;

    # Frontend
    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Backend API
    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Using Docker Compose

```yaml
version: '3'

services:
  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "8080:8080"

  backend:
    build:
      context: ./api
      dockerfile: Dockerfile.backend
    ports:
      - "8000:8000"

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/conf.d/default.conf
    depends_on:
      - frontend
      - backend
```

## 6. Development Workflow

### Running Both Frontend and Backend

1. Terminal 1 (Frontend):
   ```bash
   npm run dev
   ```

2. Terminal 2 (Backend):
   ```bash
   cd api
   source venv/bin/activate
   python run_server.py
   ```

### API Key Setup

1. Get a Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Enter it in the SEO Seer interface when prompted

## 7. Troubleshooting

### CORS Issues
- Use the provided reverse proxy setup
- Ensure backend CORS headers are configured
- For development, use browser extensions to disable CORS

### Port Conflicts
Frontend (vite.config.ts):
```typescript
server: {
  port: 3000,  // Changed from 8080
},
```

Backend:
```bash
uvicorn main:app --reload --port 8001
```

## 8. Advanced Configuration

### Environment Variables

Frontend (.env):
```
VITE_API_URL=http://localhost:8000
```

Backend:
```bash
export MAX_UPLOAD_SIZE=50000000
export DEBUG=True
```

For more details on advanced usage and customization, please refer to our documentation.
