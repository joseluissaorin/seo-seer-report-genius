
import uvicorn

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=4568, reload=True)
    print("SEO Seer API running at http://localhost:4568")
