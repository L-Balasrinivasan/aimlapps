import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.routers.router import router as ai_router

app = FastAPI(title="AI/ML Demo ", debug=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def _home_for_api_check():
    return {"about": "app for QnA with website", "swagger_url": "http://127.0.0.1:8000/docs"}

app.include_router(ai_router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True)
