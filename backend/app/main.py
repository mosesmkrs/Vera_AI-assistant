from fastapi import FastAPI

from app.routers.admin import router as admin_router
from app.routers.student import router as student_router

app = FastAPI(title="Vera AI Knowledge Assistant", version="0.1.0")
app.include_router(student_router)
app.include_router(admin_router)


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    return {"status": "ok"}

