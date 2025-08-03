from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
import os

from .database import engine
from . import models
from .routers import auth, faculty, students, phd_students, attendance

app = FastAPI()

app.add_middleware(
    CORSMiddleware,  # type: ignore
    allow_origins=["http://localhost:5173"],  # Frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)

models.Base.metadata.create_all(bind=engine)

app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(faculty.router, prefix="/faculty", tags=["faculty"])
app.include_router(students.router, prefix="/students", tags=["students"])
app.include_router(phd_students.router, prefix="/phd_students", tags=["phd_students"])
app.include_router(attendance.router, prefix="/attendance", tags=["attendance"])
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


@app.on_event("startup")
def startup_event():
    # Create directories if they don’t exist
    os.makedirs("uploads/entry_logs", exist_ok=True)
    os.makedirs("uploads/exit_logs", exist_ok=True)
    os.makedirs("uploads/processed_entry_logs", exist_ok=True)
    os.makedirs("uploads/processed_exit_logs", exist_ok=True)


@app.get("/")
def home():
    return {"Message": "Welcome to the Automated Attendance System"}
