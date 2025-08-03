
# 📸 Automated Attendance System using Face Recognition

A real-world, production-grade project developed as part of the B.Tech curriculum at **IIIT Guwahati**, this system automates classroom attendance using deep learning-powered face recognition and a secure, scalable backend.

> 🎓 Built by [Vadisetti Pranay Satvik Reddy (2201221)](mailto:vadisetti.reddy22b@iiitg.ac.in) and [N. Divyagnan Reddy (2201130)]  
> 🧑‍🏫 Guided by **Dr. Upasana Talukdar**, CSE Dept., IIIT Guwahati

---

## 🚀 Demo Video

🎥 [**Watch Demo**](https://drive.google.com/file/d/1W748n_sldVcs-_pvPRicKnC7_rwRDAft/view?usp=sharing)

---

## 🛠️ Features

### ✅ Face Recognition Pipeline (Frontend + ML)
- Multi-angle face registration using `face-api.js` (Front, Left, Right)
- Face detection via **RetinaFace**
- Fine-tuned **ArcFace (ResNet100)** model trained on real student data
- Robust to lighting, pose, and mild occlusion
- Majority voting & frame tracking for robust recognition

### ✅ Secure & Scalable Backend (FastAPI + PostgreSQL)
- Role-based authentication (Student / Faculty / PhD / Admin)
- JWT + OTP-based Multi-Factor Authentication (MFA)
- Dynamic class rescheduling & cancellation
- Attendance correction request workflow
- Entry/exit tracking enforcing ≥ 36-minute presence rule
- Audit logs, file uploads, and CSV/PDF report generation
- Modular design with Pydantic schemas & Swagger docs

---

## 🧑‍🎓 Tech Stack

| Layer        | Tools / Frameworks                               |
|--------------|--------------------------------------------------|
| Frontend     | React, React Router, Axios, face-api.js          |
| Backend      | FastAPI, Pydantic, SQLAlchemy, Celery, Redis     |
| ML Models    | ArcFace (ResNet100), RetinaFace, Albumentations |
| Database     | PostgreSQL (timezone-aware)                      |
| Deployment   | Cloudflare Tunnel, Vercel                        |

---

## 🧪 Model Weights

📦 Download the latest fine-tuned ArcFace model (ResNet100 backbone):  
🔗 [**Download Model Weights**](https://drive.google.com/drive/folders/1qLyMC2dih9s3SWJqx0diGsCrKujk8pzE?usp=sharing)

---

## 📷 Face Registration Workflow

- 📸 Students capture 30 images (10 front + 10 left + 10 right)
- 📦 Images are sent to backend after quality checks (lighting, orientation)
- ✅ Manual verification by admin before training
- 📊 Augmented via Albumentations to create >8000-image dataset
- 🧠 Fine-tuned on ArcFace with 100% test accuracy on curated set

---

## 📊 Recognition Pipeline

1. Face detection from video feed (RetinaFace)
2. Track across frames with custom FaceTracker (IoU + cosine similarity)
3. Majority-vote identity assignment after stable tracking
4. Identity logged as present only if entering from side with confidence ≥ threshold
5. Attendance recorded with timestamp & source details

---

## 🔐 Authentication Flow

- OTP Verification (email-based)
- Secure password hashing (bcrypt)
- JWT with role-based access control
- Frontend access filtered by user role (student, faculty, PhD)

---

## 📅 Dynamic Scheduling

- Faculty can reschedule/cancel classes with live update to attendance logic
- Timetable + `DynamicScheduleUpdate` used to validate real class status
- Integrated conflict detection (time/location overlaps)

---

## 📝 Attendance Correction Workflow

- Students submit correction requests with evidence (e.g., medical certificate)
- Faculty/PhD can approve/reject with audit trail
- Status reflected in attendance reports (before/after correction)

---

## 📁 Database Schema Highlights

- `users`, `attendance_report`, `student_faces`, `dynamic_schedule_update`
- `otp_verification`, `attendance_correction`, `entry_exit_time`
- Relational integrity ensured via foreign keys, unique constraints, cascade deletes

---

## 📈 Reports & Analytics

- Downloadable CSV/PDF reports (per section or individual)
- Faculty dashboard for section-wide stats
- Track attendance trends, anomalies, and corrections

---

## ⚠️ Known Limitations

| Limitation                            | Status        |
|--------------------------------------|---------------|
| Spoof Detection (e.g., photo attack) | ❌ Not yet     |
| Real-time 15s recognition latency    | ❌ In progress |
| Scalable multi-class model           | ❌ Per-class   |

---

## 🔮 Future Directions

- ✅ Add YOLO-based anti-spoofing (2D vs 3D face detection)
- ✅ Mobile app (React Native)
- ✅ Admin dashboard (bulk user management, visual analytics)
- ✅ Real-time alerts (low attendance, schedule conflicts)
- ✅ Incremental learning support for new student faces

---

## 🙏 Acknowledgments

This project would not have been possible without:
- InsightFace, face-api.js, and other open-source contributors
- Guidance from **Dr. Upasana Talukdar**
- The student volunteers who provided training data
- The incredible teamwork behind every module

---

## 📚 References

- [InsightFace GitHub](https://github.com/deepinsight/insightface)  
- [ArcFace Paper (2018)](https://arxiv.org/abs/1801.07698)  
- [Albumentations Library](https://github.com/albumentations-team/albumentations)  
- [face-api.js](https://github.com/justadudewhohacks/face-api.js)  
- [FastAPI Docs](https://fastapi.tiangolo.com/)  
- [PostgreSQL](https://www.postgresql.org/)

---
