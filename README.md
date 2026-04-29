# 🎯 FaceAttend AI — Face Recognition Attendance System

A production-grade AI-powered attendance system built with Python, Streamlit, OpenCV, and the `face_recognition` library.

---

## 🚀 Quick Start

### 1. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y python3-pip cmake build-essential libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev
```

**macOS (Homebrew):**
```bash
brew install cmake
```

**Windows:**
- Install [CMake](https://cmake.org/download/)
- Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- Install [dlib](http://dlib.net/) separately if needed

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate       # Linux/macOS
# OR
venv\Scripts\activate          # Windows
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

> ⚠️ `face_recognition` depends on `dlib`. If it fails to install, try:
> ```bash
> pip install dlib
> pip install face-recognition
> ```

### 4. Run the Application
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📁 Project Structure

```
face_attendance_system/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── attendance.db           # SQLite database (auto-created)
└── face_encodings.pkl      # Face encodings store (auto-created)
```

---

## 🔥 Features

| Module | Description |
|--------|-------------|
| 📊 Dashboard | Real-time stats — students, records, subjects |
| 👤 Register Student | Webcam-based face capture (20 samples) with duplicate prevention |
| 📚 Subject Management | Add/manage subjects dynamically |
| 📷 Mark Attendance | Live face recognition with auto attendance marking |
| 📋 Reports | Subject-wise attendance % with color coding |
| 📤 Export Excel | Styled Excel download per subject |
| 🔄 Reset System | Full wipe of all data and encodings |

---

## 🧱 Database Schema

```sql
-- Students
CREATE TABLE students (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    roll TEXT UNIQUE NOT NULL
);

-- Attendance
CREATE TABLE attendance (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    name    TEXT,
    roll    TEXT,
    subject TEXT,
    date    TEXT,
    time    TEXT
);

-- Subjects
CREATE TABLE subjects (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_name TEXT UNIQUE NOT NULL
);
```

---

## 🎯 How It Works

### Registration
1. Enter student name and roll number
2. Camera opens — position your face in frame
3. System captures 20 facial encodings
4. Encodings stored in `face_encodings.pkl`
5. Student info saved to SQLite

### Attendance
1. Select a subject
2. Camera opens
3. Faces are detected and compared against stored encodings (tolerance: 0.5)
4. Matched student → attendance recorded (once per day per subject)
5. 5-second cooldown prevents duplicate records within a session

### Reports
- Calculates: `Percentage = (classes_attended / total_classes) × 100`
- Color coded: 🟢 ≥75% | 🟡 ≥50% | 🔴 <50%

---

## ⚙️ Configuration

You can tune these in `app.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENCODINGS_FILE` | `face_encodings.pkl` | Path to face encodings |
| `DB_PATH` | `attendance.db` | SQLite database path |
| Tolerance | `0.5` | Face match tolerance (lower = stricter) |
| Samples | `20` | Face samples per registration |
| Cooldown | `5 sec` | Duplicate mark prevention window |

---

## 🛠️ Troubleshooting

**Camera not opening:**
- Make sure no other app is using the webcam
- Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` for external cameras

**face_recognition install fails:**
- Install dlib manually: `pip install dlib`
- On Apple Silicon: `pip install dlib --no-cache-dir`

**Poor recognition accuracy:**
- Ensure good lighting during registration
- Register at different angles
- Lower tolerance to `0.45` for stricter matching
