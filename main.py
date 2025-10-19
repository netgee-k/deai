"""
DE.AI - Your Writing Style Guardian
FastAPI main application with Admin Dashboard
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, BackgroundTasks, HTTPException, Depends, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import json
import asyncio
import os
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import uuid
import hashlib
import secrets

# Import analyzers with error handling
try:
    from analyzers.ai_smell_detector import AISmellDetector
    from analyzers.style_analyzer import StyleAnalyzer
    from analyzers.perplexity_analyzer import PerplexityAnalyzer
    from analyzers.real_time_analyzer import RealTimeAnalyzer
    from analyzers.writing_dna import WritingDNA
    from analyzers.real_ai_analyzer import RealAIAnalyzer
    from analyzers.style_transfer import StyleTransfer
    
    # Initialize analyzers
    ai_detector = AISmellDetector()
    style_analyzer = StyleAnalyzer()
    perplexity_analyzer = PerplexityAnalyzer()
    real_time_analyzer = RealTimeAnalyzer()
    writing_dna = WritingDNA()
    real_ai_analyzer = RealAIAnalyzer()
    style_transfer = StyleTransfer()
    
    ANALYZERS_LOADED = True
except ImportError as e:
    print(f"⚠️  Some analyzers failed to load: {e}")
    print("⚠️  Running in limited mode - creating placeholder analyzers")
    ANALYZERS_LOADED = False
    
    # Create placeholder classes for missing analyzers
    class PlaceholderAnalyzer:
        def detect(self, text):
            return [{"type": "placeholder", "reason": "Analyzer not loaded", "severity": "low"}]
        
        def analyze(self, text):
            return [{"type": "placeholder", "reason": "Analyzer not loaded", "severity": "low"}]
        
        def analyze_against_profile(self, text, user_id):
            return [{"type": "placeholder", "reason": "Analyzer not loaded", "severity": "low"}]
        
        def calculate_style_match(self, text, user_id):
            return 50.0
        
        def train_on_documents(self, documents, user_id):
            print(f"Placeholder training for user {user_id} with {len(documents)} documents")
        
        def quick_analyze(self, text):
            return {"issues": [], "score": 50.0}
        
        def create_fingerprint(self, documents, user_id):
            return {
                "vocabulary_richness": 0.5,
                "sentence_rhythm": {"avg_length": 15.0, "length_variance": 10.0},
                "readability_level": 60.0,
                "signature_phrases": [],
                "emotional_tone": {"neutral": 1.0},
                "writing_patterns": {"formality_score": 0.5}
            }
        
        def calculate_real_perplexity(self, text):
            return 50.0
        
        def analyze_with_ai(self, text, perplexity):
            return {"issues": [], "perplexity": perplexity}
        
        def get_ai_likelihood(self, perplexity):
            return "Unknown"
        
        def suggest_improvements(self, text, writing_dna, metrics):
            return []
    
    # Initialize placeholder analyzers
    ai_detector = PlaceholderAnalyzer()
    style_analyzer = PlaceholderAnalyzer()
    perplexity_analyzer = PlaceholderAnalyzer()
    real_time_analyzer = PlaceholderAnalyzer()
    writing_dna = PlaceholderAnalyzer()
    real_ai_analyzer = PlaceholderAnalyzer()
    style_transfer = PlaceholderAnalyzer()

app = FastAPI(
    title="DE.AI",
    description="A tool that helps your writing sound LESS like AI and MORE like YOU",
    version="4.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database configuration
DATABASE_URL = "data/deai_admin.db"

def get_db():
    """Get database connection"""
    conn = sqlite3.connect(DATABASE_URL)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database with required tables"""
    conn = get_db()
    cursor = conn.cursor()
    
    # Analysis logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id TEXT UNIQUE NOT NULL,
            user_id TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            text_length INTEGER NOT NULL,
            score REAL NOT NULL,
            style_match REAL NOT NULL,
            real_perplexity REAL,
            issues_count INTEGER NOT NULL,
            processing_time REAL NOT NULL,
            analysis_types TEXT
        )
    ''')
    
    # User profiles table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id TEXT PRIMARY KEY,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_activity DATETIME DEFAULT CURRENT_TIMESTAMP,
            analysis_count INTEGER DEFAULT 0,
            avg_score REAL DEFAULT 0,
            avg_style_match REAL DEFAULT 0,
            total_text_length INTEGER DEFAULT 0
        )
    ''')
    
    # Writing sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS writing_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            analysis_id TEXT,
            session_start DATETIME DEFAULT CURRENT_TIMESTAMP,
            session_end DATETIME,
            activity_type TEXT NOT NULL,
            duration_seconds INTEGER
        )
    ''')
    
    # Admin users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS admin_users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def hash_password(password: str) -> str:
    """Hash a password for storage"""
    return hashlib.sha256(password.encode()).hexdigest()

def create_admin_user():
    """Create default admin user if not exists"""
    conn = get_db()
    cursor = conn.cursor()
    
    # Check if admin user exists
    cursor.execute("SELECT * FROM admin_users WHERE username = ?", ("admin",))
    if not cursor.fetchone():
        # Create default admin user (password: admin123)
        password_hash = hash_password("admin123")
        cursor.execute(
            "INSERT INTO admin_users (username, password_hash) VALUES (?, ?)",
            ("admin", password_hash)
        )
        conn.commit()
        print("✅ Default admin user created: admin / admin123")
    
    conn.close()

def verify_admin(username: str, password: str) -> bool:
    """Verify admin credentials"""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT password_hash FROM admin_users WHERE username = ?",
        (username,)
    )
    result = cursor.fetchone()
    conn.close()
    
    if result and result['password_hash'] == hash_password(password):
        return True
    return False

def get_current_admin(auth: str = Depends(lambda: "")):
    """Dependency to get current admin from auth token"""
    # Simple auth for demo - in production use JWT tokens
    if auth != "admin-secret-token":
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {"username": "admin"}

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_db()
    create_admin_user()

# Connection manager for WebSocket
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

class AnalysisRequest(BaseModel):
    text: str
    analysis_types: List[str] = ["all"]
    user_id: Optional[str] = "default"

class AnalysisResponse(BaseModel):
    score: float
    issues: List[Dict]
    suggestions: List[str]
    style_match: float
    highlights: List[Dict]
    writing_dna: Optional[Dict] = None
    real_perplexity: Optional[float] = None
    style_transfer_suggestions: Optional[List[Dict]] = None
    analysis_id: Optional[str] = None

class StyleTrainingRequest(BaseModel):
    documents: List[str]
    user_id: str

class WritingDNARequest(BaseModel):
    user_id: str

class BatchAnalysisRequest(BaseModel):
    folder_path: str
    user_id: str = "default"

class AdminStatsRequest(BaseModel):
    time_range: str = "7d"  # 7d, 30d, 90d

class AdminLoginRequest(BaseModel):
    username: str
    password: str

# Database Models
class AnalysisLog:
    @staticmethod
    def create(db: sqlite3.Connection, analysis_id: str, user_id: str, text_length: int,
               score: float, style_match: float, real_perplexity: Optional[float],
               issues_count: int, processing_time: float, analysis_types: str):
        cursor = db.cursor()
        cursor.execute('''
            INSERT INTO analysis_logs 
            (analysis_id, user_id, text_length, score, style_match, real_perplexity, 
             issues_count, processing_time, analysis_types)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (analysis_id, user_id, text_length, score, style_match, real_perplexity,
              issues_count, processing_time, analysis_types))
        db.commit()
    
    @staticmethod
    def get_recent(db: sqlite3.Connection, limit: int = 10) -> List[Dict]:
        cursor = db.cursor()
        cursor.execute('''
            SELECT * FROM analysis_logs 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        return [dict(row) for row in cursor.fetchall()]
    
    @staticmethod
    def count_all(db: sqlite3.Connection) -> int:
        cursor = db.cursor()
        cursor.execute('SELECT COUNT(*) FROM analysis_logs')
        return cursor.fetchone()[0]
    
    @staticmethod
    def count_since(db: sqlite3.Connection, since: datetime) -> int:
        cursor = db.cursor()
        cursor.execute('SELECT COUNT(*) FROM analysis_logs WHERE timestamp > ?', (since,))
        return cursor.fetchone()[0]
    
    @staticmethod
    def get_daily_counts(db: sqlite3.Connection, start_date: datetime) -> List[Dict]:
        cursor = db.cursor()
        cursor.execute('''
            SELECT DATE(timestamp) as date, COUNT(*) as count
            FROM analysis_logs 
            WHERE timestamp > ?
            GROUP BY DATE(timestamp)
            ORDER BY date
        ''', (start_date,))
        return [dict(row) for row in cursor.fetchall()]
    
    @staticmethod
    def get_score_distribution(db: sqlite3.Connection, start_date: datetime) -> List[Dict]:
        cursor = db.cursor()
        cursor.execute('''
            SELECT 
                CASE 
                    WHEN score < 20 THEN '0-20'
                    WHEN score < 40 THEN '20-40'
                    WHEN score < 60 THEN '40-60'
                    WHEN score < 80 THEN '60-80'
                    ELSE '80-100'
                END as score_range,
                COUNT(*) as count
            FROM analysis_logs 
            WHERE timestamp > ?
            GROUP BY score_range
            ORDER BY score_range
        ''', (start_date,))
        return [dict(row) for row in cursor.fetchall()]
    
    @staticmethod
    def get_paginated(db: sqlite3.Connection, page: int = 1, limit: int = 20, user_id: Optional[str] = None) -> List[Dict]:
        cursor = db.cursor()
        offset = (page - 1) * limit
        
        if user_id:
            cursor.execute('''
                SELECT * FROM analysis_logs 
                WHERE user_id = ?
                ORDER BY timestamp DESC 
                LIMIT ? OFFSET ?
            ''', (user_id, limit, offset))
        else:
            cursor.execute('''
                SELECT * FROM analysis_logs 
                ORDER BY timestamp DESC 
                LIMIT ? OFFSET ?
            ''', (limit, offset))
        
        return [dict(row) for row in cursor.fetchall()]
    
    @staticmethod
    def get_by_user(db: sqlite3.Connection, user_id: str, limit: int = 50) -> List[Dict]:
        cursor = db.cursor()
        cursor.execute('''
            SELECT * FROM analysis_logs 
            WHERE user_id = ?
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (user_id, limit))
        return [dict(row) for row in cursor.fetchall()]
    
    @staticmethod
    def get_ai_detection_stats(db: sqlite3.Connection, start_date: datetime) -> Dict:
        cursor = db.cursor()
        
        # Count analyses with low scores (likely AI)
        cursor.execute('''
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN score < 50 THEN 1 ELSE 0 END) as likely_ai,
                SUM(CASE WHEN score >= 50 THEN 1 ELSE 0 END) as likely_human
            FROM analysis_logs 
            WHERE timestamp > ?
        ''', (start_date,))
        
        result = cursor.fetchone()
        return dict(result) if result else {}

class UserProfile:
    @staticmethod
    def get(db: sqlite3.Connection, user_id: str) -> Optional[Dict]:
        cursor = db.cursor()
        cursor.execute('SELECT * FROM user_profiles WHERE user_id = ?', (user_id,))
        result = cursor.fetchone()
        return dict(result) if result else None
    
    @staticmethod
    def create(db: sqlite3.Connection, user_id: str, avg_score: float, avg_style_match: float, analysis_count: int = 1):
        cursor = db.cursor()
        cursor.execute('''
            INSERT INTO user_profiles 
            (user_id, avg_score, avg_style_match, analysis_count, total_text_length)
            VALUES (?, ?, ?, ?, 0)
        ''', (user_id, avg_score, avg_style_match, analysis_count))
        db.commit()
    
    @staticmethod
    def update_stats(db: sqlite3.Connection, user_id: str, new_score: float, new_style_match: float):
        cursor = db.cursor()
        
        # Get current stats
        cursor.execute('SELECT * FROM user_profiles WHERE user_id = ?', (user_id,))
        current = cursor.fetchone()
        
        if current:
            # Update averages
            current_count = current['analysis_count']
            new_count = current_count + 1
            
            new_avg_score = (current['avg_score'] * current_count + new_score) / new_count
            new_avg_style_match = (current['avg_style_match'] * current_count + new_style_match) / new_count
            
            cursor.execute('''
                UPDATE user_profiles 
                SET avg_score = ?, avg_style_match = ?, analysis_count = ?, last_activity = CURRENT_TIMESTAMP
                WHERE user_id = ?
            ''', (new_avg_score, new_avg_style_match, new_count, user_id))
            db.commit()
    
    @staticmethod
    def count_all(db: sqlite3.Connection) -> int:
        cursor = db.cursor()
        cursor.execute('SELECT COUNT(*) FROM user_profiles')
        return cursor.fetchone()[0]
    
    @staticmethod
    def get_most_active(db: sqlite3.Connection, limit: int = 10) -> List[Dict]:
        cursor = db.cursor()
        cursor.execute('''
            SELECT * FROM user_profiles 
            ORDER BY analysis_count DESC 
            LIMIT ?
        ''', (limit,))
        return [dict(row) for row in cursor.fetchall()]
    
    @staticmethod
    def get_paginated(db: sqlite3.Connection, page: int = 1, limit: int = 20) -> List[Dict]:
        cursor = db.cursor()
        offset = (page - 1) * limit
        
        cursor.execute('''
            SELECT * FROM user_profiles 
            ORDER BY last_activity DESC 
            LIMIT ? OFFSET ?
        ''', (limit, offset))
        
        return [dict(row) for row in cursor.fetchall()]
    
    @staticmethod
    def get_growth_stats(db: sqlite3.Connection, start_date: datetime) -> List[Dict]:
        cursor = db.cursor()
        cursor.execute('''
            SELECT DATE(created_at) as date, COUNT(*) as new_users
            FROM user_profiles 
            WHERE created_at > ?
            GROUP BY DATE(created_at)
            ORDER BY date
        ''', (start_date,))
        return [dict(row) for row in cursor.fetchall()]
    
    @staticmethod
    def delete(db: sqlite3.Connection, user_id: str):
        cursor = db.cursor()
        cursor.execute('DELETE FROM user_profiles WHERE user_id = ?', (user_id,))
        db.commit()

class WritingSession:
    @staticmethod
    def log_activity(db: sqlite3.Connection, user_id: str, analysis_id: Optional[str] = None, activity_type: str = "analysis"):
        cursor = db.cursor()
        cursor.execute('''
            INSERT INTO writing_sessions 
            (user_id, analysis_id, activity_type)
            VALUES (?, ?, ?)
        ''', (user_id, analysis_id, activity_type))
        db.commit()
    
    @staticmethod
    def get_user_sessions(db: sqlite3.Connection, user_id: str, limit: int = 20) -> List[Dict]:
        cursor = db.cursor()
        cursor.execute('''
            SELECT * FROM writing_sessions 
            WHERE user_id = ?
            ORDER BY session_start DESC 
            LIMIT ?
        ''', (user_id, limit))
        return [dict(row) for row in cursor.fetchall()]

# =============================================================================
# MAIN APPLICATION ENDPOINTS
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the main DE.AI interface"""
    try:
        with open("templates/dashboard.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        # Fallback basic dashboard
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>DE.AI - Your Writing Style Guardian</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                textarea { width: 100%; height: 200px; margin: 10px 0; }
                button { padding: 10px 20px; margin: 5px; }
            </style>
        </head>
        <body>
            <h1>DE.AI - Your Writing Style Guardian</h1>
            <p>Make your writing sound LESS like AI and MORE like YOU</p>
            <textarea id="text-input" placeholder="Enter your text here..."></textarea>
            <br>
            <button onclick="analyzeText()">Analyze Text</button>
            <div id="results"></div>
            <script>
                async function analyzeText() {
                    const text = document.getElementById('text-input').value;
                    const response = await fetch('/analyze', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({text: text, analysis_types: ['all']})
                    });
                    const result = await response.json();
                    document.getElementById('results').innerHTML = `
                        <h3>Score: ${result.score}/100</h3>
                        <p>Style Match: ${result.style_match}%</p>
                        <ul>
                            ${result.issues.map(issue => `<li>${issue.type}: ${issue.reason}</li>`).join('')}
                        </ul>
                    `;
                }
            </script>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(request: AnalysisRequest):
    """Analyze text for AI patterns and style match"""
    
    analysis_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    issues = []
    suggestions = []
    writing_dna_data = None
    real_perplexity = None
    style_transfer_suggestions = []
    
    # Run all analyzers
    if "all" in request.analysis_types or "ai_smell" in request.analysis_types:
        ai_issues = ai_detector.detect(request.text)
        issues.extend(ai_issues)
    
    if "all" in request.analysis_types or "style" in request.analysis_types:
        style_issues = style_analyzer.analyze_against_profile(request.text, request.user_id)
        issues.extend(style_issues)
    
    if "all" in request.analysis_types or "perplexity" in request.analysis_types:
        perplexity_issues = perplexity_analyzer.analyze(request.text)
        issues.extend(perplexity_issues)
    
    # Real AI perplexity analysis
    if "all" in request.analysis_types or "real_ai" in request.analysis_types:
        try:
            real_perplexity = real_ai_analyzer.calculate_real_perplexity(request.text)
            perplexity_analysis = real_ai_analyzer.analyze_with_ai(request.text, real_perplexity)
            issues.extend(perplexity_analysis.get("issues", []))
        except Exception as e:
            print(f"Real AI analysis failed: {e}")
            # Fall back to basic perplexity
            real_perplexity = 50.0
    
    # Get Writing DNA if available (using placeholder if real one not loaded)
    try:
        if hasattr(writing_dna, 'fingerprints') and request.user_id in writing_dna.fingerprints:
            writing_dna_data = writing_dna.fingerprints[request.user_id]
        elif hasattr(writing_dna, 'create_fingerprint'):
            # Use placeholder
            writing_dna_data = writing_dna.create_fingerprint([], request.user_id)
    except Exception as e:
        print(f"Writing DNA access failed: {e}")
    
    # Add signature phrase detection if DNA data available
    if writing_dna_data:
        dna_issues = detect_signature_phrase_matches(request.text, writing_dna_data)
        issues.extend(dna_issues)
        
        # Generate style transfer suggestions
        try:
            style_transfer_suggestions = style_transfer.suggest_improvements(
                request.text, writing_dna_data, {
                    'vocabulary_richness': real_perplexity or 50,
                    'issues': issues
                }
            )
        except Exception as e:
            print(f"Style transfer failed: {e}")
    
    # Calculate overall score (0-100, higher = more human)
    score = calculate_humanity_score(issues, request.text, real_perplexity)
    
    # Calculate style match percentage
    try:
        style_match = style_analyzer.calculate_style_match(request.text, request.user_id)
    except Exception as e:
        print(f"Style match calculation failed: {e}")
        style_match = 50.0
    
    # Generate highlights for UI
    highlights = generate_highlights(issues, request.text)
    
    # Generate personalized suggestions
    personalized_suggestions = generate_personalized_suggestions(
        issues, writing_dna_data, real_perplexity, style_transfer_suggestions
    )
    
    # Log analysis to database
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    try:
        db = get_db()
        AnalysisLog.create(
            db=db,
            analysis_id=analysis_id,
            user_id=request.user_id,
            text_length=len(request.text),
            score=score,
            style_match=style_match,
            real_perplexity=real_perplexity,
            issues_count=len(issues),
            processing_time=processing_time,
            analysis_types=json.dumps(request.analysis_types)
        )
        
        # Update or create user profile
        user_profile = UserProfile.get(db, request.user_id)
        if user_profile:
            UserProfile.update_stats(
                db=db,
                user_id=request.user_id,
                new_score=score,
                new_style_match=style_match
            )
        else:
            UserProfile.create(
                db=db,
                user_id=request.user_id,
                avg_score=score,
                avg_style_match=style_match,
                analysis_count=1
            )
        
        # Log writing session
        WritingSession.log_activity(
            db=db,
            user_id=request.user_id,
            analysis_id=analysis_id,
            activity_type="analysis"
        )
        
        db.close()
    except Exception as e:
        print(f"Database operation failed: {e}")
    
    return AnalysisResponse(
        score=score,
        issues=issues,
        suggestions=personalized_suggestions,
        style_match=style_match,
        highlights=highlights,
        writing_dna=writing_dna_data,
        real_perplexity=real_perplexity,
        style_transfer_suggestions=style_transfer_suggestions,
        analysis_id=analysis_id
    )

@app.post("/analyze-folder")
async def analyze_folder(request: BatchAnalysisRequest):
    """Analyze all documents in a folder and compare against user's style"""
    return {"error": "Folder analysis not implemented in this version"}

@app.websocket("/ws/analyze")
async def websocket_analysis(websocket: WebSocket):
    """Real-time analysis via WebSocket with live highlighting"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "text_update":
                # Quick real-time analysis
                analysis = real_time_analyzer.quick_analyze(message["text"])
                
                # Add Real AI analysis if text is substantial
                if len(message["text"]) > 100:
                    try:
                        real_perplexity = real_ai_analyzer.calculate_real_perplexity(message["text"])
                        analysis["real_perplexity"] = real_perplexity
                        analysis["ai_likelihood"] = real_ai_analyzer.get_ai_likelihood(real_perplexity)
                    except Exception as e:
                        analysis["real_ai_error"] = str(e)
                
                await websocket.send_json({
                    "type": "analysis_update",
                    "analysis": analysis,
                    "highlights": generate_realtime_highlights(analysis.get("issues", []), message["text"])
                })
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/train-style")
async def train_style_model(request: StyleTrainingRequest, background_tasks: BackgroundTasks):
    """Train the style model on user's documents"""
    background_tasks.add_task(
        style_analyzer.train_on_documents,
        request.documents,
        request.user_id
    )
    
    return {"status": "training_started", "message": "Style model training in background"}

@app.post("/create-writing-dna")
async def create_writing_dna(request: StyleTrainingRequest):
    """Create a detailed writing fingerprint"""
    try:
        fingerprint = writing_dna.create_fingerprint(request.documents, request.user_id)
        return {
            "status": "fingerprint_created", 
            "fingerprint": fingerprint,
            "message": f"Writing DNA profile created for {request.user_id}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating writing DNA: {str(e)}")

@app.get("/writing-report/{user_id}")
async def get_writing_report(user_id: str):
    """Get comprehensive writing style report"""
    try:
        # Try to get fingerprint from real writing_dna
        if hasattr(writing_dna, 'fingerprints') and user_id in writing_dna.fingerprints:
            fingerprint = writing_dna.fingerprints[user_id]
        else:
            # Use placeholder
            fingerprint = writing_dna.create_fingerprint([], user_id)
        
        # Calculate writing personality
        personality = calculate_writing_personality(fingerprint)
        
        return {
            "user_id": user_id,
            "vocabulary_richness": f"{fingerprint.get('vocabulary_richness', 0.5):.1%}",
            "avg_sentence_length": f"{fingerprint.get('sentence_rhythm', {}).get('avg_length', 15):.1f} words",
            "sentence_variety": "High" if fingerprint.get('sentence_rhythm', {}).get('length_variance', 10) > 15 else "Moderate" if fingerprint.get('sentence_rhythm', {}).get('length_variance', 10) > 8 else "Low",
            "readability_score": f"{fingerprint.get('readability_level', 60):.1f}",
            "signature_phrases": fingerprint.get('signature_phrases', [])[:5],
            "emotional_tone": max(fingerprint.get('emotional_tone', {'neutral': 1.0}).items(), key=lambda x: x[1])[0],
            "writing_personality": personality,
        }
    except Exception as e:
        return {"error": f"No writing DNA fingerprint found: {str(e)}"}

@app.post("/upload-training-docs")
async def upload_training_documents(files: List[UploadFile] = File(...), user_id: str = "default"):
    """Upload documents for style training and DNA analysis"""
    saved_files = []
    document_contents = []
    
    for file in files:
        # Save file
        file_path = f"data/training_{user_id}_{file.filename}"
        os.makedirs("data", exist_ok=True)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        saved_files.append(file_path)
        
        # Extract text content for analysis
        try:
            text_content = content.decode('utf-8')
            document_contents.append(text_content)
        except:
            # If binary file, skip text extraction
            pass
    
    # Train both style analyzer and writing DNA
    if document_contents:
        style_analyzer.train_on_documents(document_contents, user_id)
        writing_dna.create_fingerprint(document_contents, user_id)
    
    return {
        "saved_files": saved_files, 
        "message": f"Documents uploaded and analyzed for user {user_id}",
        "documents_processed": len(document_contents)
    }

@app.get("/user-profiles")
async def list_user_profiles():
    """List all available user profiles"""
    try:
        if hasattr(style_analyzer, 'user_profiles'):
            style_profiles = list(style_analyzer.user_profiles.keys())
        else:
            style_profiles = []
        
        if hasattr(writing_dna, 'fingerprints'):
            dna_profiles = list(writing_dna.fingerprints.keys())
        else:
            dna_profiles = []
        
        return {
            "style_profiles": style_profiles,
            "dna_profiles": dna_profiles,
            "total_users": len(set(style_profiles + dna_profiles))
        }
    except Exception as e:
        return {"style_profiles": [], "dna_profiles": [], "total_users": 0, "error": str(e)}

@app.delete("/user-profile/{user_id}")
async def delete_user_profile(user_id: str):
    """Delete a user's profile"""
    try:
        if hasattr(style_analyzer, 'user_profiles') and user_id in style_analyzer.user_profiles:
            del style_analyzer.user_profiles[user_id]
        if hasattr(writing_dna, 'fingerprints') and user_id in writing_dna.fingerprints:
            del writing_dna.fingerprints[user_id]
        
        if hasattr(style_analyzer, 'save_profiles'):
            style_analyzer.save_profiles()
        
        return {"message": f"Profile for {user_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting profile: {str(e)}")

# =============================================================================
# ADMIN ENDPOINTS (Simplified for now)
# =============================================================================

@app.post("/admin/login")
async def admin_login(request: AdminLoginRequest):
    """Admin login endpoint"""
    if verify_admin(request.username, request.password):
        return {"success": True, "token": "admin-secret-token", "message": "Login successful"}
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/admin/")
async def admin_dashboard():
    """Admin dashboard overview"""
    return {"message": "Admin dashboard - use proper authentication"}

@app.get("/admin/analytics")
async def get_analytics(time_range: str = "7d"):
    """Get analytics data for charts"""
    return {"message": "Analytics endpoint - use proper authentication"}

@app.get("/admin/users")
async def get_users(page: int = 1, limit: int = 20):
    """Get paginated user list"""
    return {"message": "Users endpoint - use proper authentication"}

@app.get("/admin/analyses")
async def get_analyses(page: int = 1, limit: int = 20, user_id: Optional[str] = None):
    """Get paginated analysis logs"""
    return {"message": "Analyses endpoint - use proper authentication"}

@app.get("/admin/user/{user_id}")
async def get_user_detail(user_id: str):
    """Get detailed user information"""
    return {"message": f"User detail for {user_id} - use proper authentication"}

@app.delete("/admin/user/{user_id}")
async def delete_user(user_id: str):
    """Delete a user and all their data"""
    return {"message": f"Delete user {user_id} - use proper authentication"}

@app.get("/admin/export/analyses")
async def export_analyses(format: str = "json"):
    """Export analysis data"""
    return {"message": "Export endpoint - use proper authentication"}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_humanity_score(issues: List[Dict], text: str, real_perplexity: Optional[float] = None) -> float:
    """Calculate overall humanity score (0-100) with Writing DNA bonus and real perplexity"""
    base_score = 100.0
    
    # Deduct points for each issue
    for issue in issues:
        if issue.get("severity") == "high":
            base_score -= 5
        elif issue.get("severity") == "medium":
            base_score -= 2
        else:
            base_score -= 1
    
    # Bonus for signature phrases (makes writing more human)
    signature_bonus = sum(1 for issue in issues if issue.get("type") == "signature_phrase") * 2
    base_score += signature_bonus
    
    # Adjust based on real perplexity if available
    if real_perplexity:
        if real_perplexity < 20:  # Very predictable (AI-like)
            base_score *= 0.7
        elif real_perplexity > 80:  # Very unpredictable (human-like)
            base_score *= 1.1
    
    # Ensure score stays within bounds
    return max(0.0, min(100.0, base_score))

def generate_highlights(issues: List[Dict], text: str) -> List[Dict]:
    """Generate highlighted sections for the UI with DNA-aware coloring"""
    highlights = []
    for issue in issues:
        if "position" in issue:
            color = get_highlight_color(issue["type"])
            highlights.append({
                "start": issue["position"],
                "end": issue["position"] + len(issue.get("text", "")),
                "type": issue["type"],
                "message": issue["reason"],
                "suggestion": issue.get("suggestion", ""),
                "color": color,
                "severity": issue.get("severity", "low")
            })
    return highlights

def get_highlight_color(issue_type: str) -> str:
    """Get color for different issue types"""
    color_map = {
        "formal_word": "#ff6b6b",
        "redundant_phrase": "#ff6b6b",
        "ai_cliche": "#ff6b6b",
        "passive_voice": "#ff922b",
        "sentence_rhythm": "#ffd93d",
        "vocabulary_mismatch": "#ffd93d",
        "signature_phrase": "#51cf66",
        "low_creativity": "#339af0",
        "high_creativity": "#339af0",
        "low_perplexity": "#e64980",
        "high_perplexity": "#339af0",
        "placeholder": "#868e96"
    }
    return color_map.get(issue_type, "#868e96")

def generate_personalized_suggestions(issues: List[Dict], writing_dna: Optional[Dict], 
                                    real_perplexity: Optional[float], 
                                    style_transfer_suggestions: List[Dict]) -> List[str]:
    """Generate suggestions based on writing DNA and real AI analysis"""
    suggestions = []
    
    # Basic suggestions
    ai_issues = [issue for issue in issues if issue["type"] in ["formal_word", "redundant_phrase", "ai_cliche"]]
    if ai_issues:
        suggestions.append(f"Found {len(ai_issues)} AI-like patterns. Try using more natural language.")
    
    style_issues = [issue for issue in issues if issue["type"] in ["sentence_rhythm", "vocabulary_mismatch"]]
    if style_issues:
        suggestions.append("Consider adjusting to match your usual writing style.")
    
    # Real AI analysis suggestions
    if real_perplexity:
        if real_perplexity < 30:
            suggestions.append("Text is very predictable (AI-like). Add more creative variations.")
        elif real_perplexity > 70:
            suggestions.append("Text shows good creativity and unpredictability (human-like).")
    
    # Style transfer suggestions
    for transfer_suggestion in style_transfer_suggestions[:3]:
        suggestions.append(transfer_suggestion.get("message", ""))
    
    # Always include score
    score = calculate_humanity_score(issues, "", real_perplexity)
    suggestions.append(f"Overall humanity score: {score:.1f}/100")
    
    return suggestions

def detect_signature_phrase_matches(text: str, writing_dna: Dict) -> List[Dict]:
    """Detect when text uses signature phrases from writing DNA"""
    issues = []
    text_lower = text.lower()
    
    for phrase in writing_dna.get('signature_phrases', [])[:5]:
        if phrase in text_lower:
            position = text_lower.find(phrase)
            issues.append({
                "type": "signature_phrase",
                "text": phrase,
                "reason": "This matches your signature writing style",
                "suggestion": "Great! This sounds authentically like you.",
                "severity": "low",
                "position": position
            })
    
    return issues

def generate_realtime_highlights(issues: List[Dict], text: str) -> List[Dict]:
    """Generate highlights for real-time analysis"""
    return generate_highlights(issues, text)

def calculate_writing_personality(fingerprint: Dict) -> str:
    """Calculate writing personality based on DNA fingerprint"""
    traits = []
    
    # Vocabulary richness
    if fingerprint.get('vocabulary_richness', 0.5) > 0.4:
        traits.append("Eloquent")
    elif fingerprint.get('vocabulary_richness', 0.5) < 0.2:
        traits.append("Direct")
    else:
        traits.append("Balanced")
    
    # Sentence rhythm
    variance = fingerprint.get('sentence_rhythm', {}).get('length_variance', 10)
    if variance > 20:
        traits.append("Dynamic")
    elif variance < 5:
        traits.append("Consistent")
    else:
        traits.append("Fluid")
    
    return " • ".join(traits) if traits else "Unique"

# Ensure data directory exists
Path("data").mkdir(exist_ok=True)
Path("templates").mkdir(exist_ok=True)

# Mount static files if directory exists
if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Add route listing endpoint that handles WebSocket routes
@app.get("/routes")
async def list_routes():
    """List all available routes"""
    routes = []
    for route in app.routes:
        route_info = {
            "path": route.path,
            "name": getattr(route, "name", "Unnamed"),
            "methods": getattr(route, "methods", ["WEBSOCKET"])  # WebSocket routes don't have methods
        }
        routes.append(route_info)
    return routes

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting DE.AI Server...")
    print("📊 Available endpoints:")
    for route in app.routes:
        methods = getattr(route, "methods", ["WEBSOCKET"])
        print(f"   {', '.join(methods):<15} {route.path}")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)