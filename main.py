"""Neural Study OS - Main Backend
Local AI-powered study system with RAG, spaced repetition, and cognitive mapping.
Optimized for RTX 3050 / Ryzen 5500H
"""

import os
import json
import sqlite3
import base64
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma_db"
DB_PATH = DATA_DIR / "study.db"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")

for d in [DATA_DIR, UPLOAD_DIR, CHROMA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================================
# APP INITIALIZATION
# ============================================================================

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# ============================================================================
# DATABASE SETUP
# ============================================================================

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("""CREATE TABLE IF NOT EXISTS study_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT DEFAULT 'default',
        topic TEXT,
        subject TEXT,
        duration_minutes INTEGER,
        description TEXT,
        mood TEXT,
        difficulty_rating INTEGER,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        completed INTEGER DEFAULT 0
    )""")
    
    c.execute("""CREATE TABLE IF NOT EXISTS cognitive_profile (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT DEFAULT 'default',
        learning_style TEXT,
        peak_hours TEXT,
        avg_session_duration INTEGER,
        preferred_content_type TEXT,
        weak_topics TEXT,
        strong_topics TEXT,
        retention_rate REAL,
        last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")
    
    c.execute("""CREATE TABLE IF NOT EXISTS spaced_repetition (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT DEFAULT 'default',
        topic TEXT,
        content TEXT,
        difficulty REAL DEFAULT 2.5,
        interval_days INTEGER DEFAULT 1,
        repetitions INTEGER DEFAULT 0,
        next_review DATETIME,
        last_reviewed DATETIME,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")
    
    c.execute("""CREATE TABLE IF NOT EXISTS knowledge_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_type TEXT,
        source_name TEXT,
        content TEXT,
        embedding_id TEXT,
        metadata TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")
    
    c.execute("""CREATE TABLE IF NOT EXISTS exam_mappings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        exam_name TEXT,
        topic TEXT,
        chapter TEXT,
        priority TEXT,
        status TEXT DEFAULT 'not_started',
        progress REAL DEFAULT 0.0,
        last_studied DATETIME
    )""")
    
    c.execute("""CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT DEFAULT 'default',
        role TEXT,
        content TEXT,
        topic TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")
    
    c.execute("""CREATE TABLE IF NOT EXISTS diagnostic_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT DEFAULT 'default',
        topic TEXT,
        questions JSON,
        answers JSON,
        score REAL,
        level TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")
    
    conn.commit()
    conn.close()
    print("Database initialized.")

# ============================================================================
# OLLAMA LOCAL AI ENGINE
# ============================================================================

def query_ollama(prompt: str, model: str = None, system: str = None) -> str:
    """Query local Ollama LLM with fallback to online if needed."""
    model = model or LLM_MODEL
    
    try:
        url = f"{OLLAMA_URL}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system or "",
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 2048
            }
        }
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        print(f"Ollama error: {e}")
        return f"Local AI unavailable. Please ensure Ollama is running with: ollama pull {model}"


def query_ollama_chat(messages: List[Dict]) -> str:
    """Chat with Ollama using conversation history."""
    try:
        url = f"{OLLAMA_URL}/api/chat"
        payload = {
            "model": LLM_MODEL,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.7}
        }
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "")
    except Exception as e:
        return f"Ollama not available. Run: ollama pull {LLM_MODEL}"


# ============================================================================
# KNOWLEDGE PROCESSOR - PDF & YOUTUBE
# ============================================================================

try:
    import fitz  # PyMuPDF
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

try:
    import yt_dlp
    HAS_YTDLP = True
except ImportError:
    HAS_YTDLP = False


def process_pdf(file_path: str, source_name: str) -> List[Dict]:
    """Extract text chunks from PDF for RAG indexing."""
    if not HAS_PDF:
        return []
    
    chunks = []
    doc = fitz.open(file_path)
    
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        # Split into chunks of ~500 chars
        for i, chunk_start in enumerate(range(0, len(text), 500)):
            chunk = text[chunk_start:chunk_start + 500]
            if chunk.strip():
                chunks.append({
                    "source_type": "pdf",
                    "source_name": source_name,
                    "page": page_num + 1,
                    "chunk_idx": i,
                    "content": chunk
                })
    
    doc.close()
    return chunks


def process_youtube(url: str, save_dir: str = None) -> Optional[Dict]:
    """Download YouTube video audio and transcribe with Whisper."""
    if not HAS_YTDLP:
        return None
    
    save_dir = save_dir or str(UPLOAD_DIR / "youtube")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": f"{save_dir}/%(id)s.%(ext)s",
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }],
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            audio_path = f"{save_dir}/{info['id']}.wav"
        
        # Transcribe with Whisper
        import whisper
        model = whisper.load_model(WHISPER_MODEL)
        result = model.transcribe(audio_path, verbose=False)
        
        return {
            "title": info.get("title", "Unknown"),
            "duration": info.get("duration", 0),
            "transcript": result["text"],
            "segments": result.get("segments", []),
            "audio_path": audio_path
        }
    except Exception as e:
        print(f"YouTube processing error: {e}")
        return None


# ============================================================================
# COGNITIVE PROFILE ENGINE - Learns About You
# ============================================================================

class CognitiveProfile:
    """Builds and updates a cognitive model of the student."""
    
    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._ensure_profile()
    
    def _ensure_profile(self):
        c = self.conn.cursor()
        c.execute("SELECT * FROM cognitive_profile WHERE user_id = ?", (self.user_id,))
        if not c.fetchone():
            c.execute("""INSERT INTO cognitive_profile 
                        (user_id, learning_style, peak_hours, avg_session_duration, 
                         preferred_content_type, weak_topics, strong_topics, retention_rate)
                        VALUES (?, 'mixed', '9-11,15-17,21-23', 45, 'visual', 
                                '[]', '[]', 0.75)""", (self.user_id,))
            self.conn.commit()
    
    def log_session(self, topic: str, subject: str, duration: int, 
                    description: str = "", mood: str = "neutral", difficulty: int = 3):
        """Log a study session and update cognitive profile."""
        c = self.conn.cursor()
        c.execute("""INSERT INTO study_sessions 
                    (user_id, topic, subject, duration_minutes, description, mood, difficulty_rating)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                 (self.user_id, topic, subject, duration, description, mood, difficulty))
        self.conn.commit()
        self._update_profile()
        return self._generate_feedback(description, difficulty, mood)
    
    def _update_profile(self):
        """Recalculate profile based on session history."""
        c = self.conn.cursor()
        
        # Average session duration
        c.execute("SELECT AVG(duration_minutes) FROM study_sessions WHERE user_id = ?", 
                 (self.user_id,))
        avg_dur = c.fetchone()[0] or 45
        
        # Peak hours analysis
        c.execute("""SELECT strftime('%H', timestamp) as hour, COUNT(*) as cnt
                     FROM study_sessions WHERE user_id = ? GROUP BY hour ORDER BY cnt DESC LIMIT 3""",
                 (self.user_id,))
        peak_hours = [r[0] for r in c.fetchall()]
        
        # Weak topics (high difficulty, low completion)
        c.execute("""SELECT topic, AVG(difficulty_rating) as avg_diff, COUNT(*) as cnt
                     FROM study_sessions WHERE user_id = ? AND difficulty_rating >= 4
                     GROUP BY topic HAVING cnt >= 2 ORDER BY avg_diff DESC LIMIT 5""",
                 (self.user_id,))
        weak = [r[0] for r in c.fetchall()]
        
        # Strong topics
        c.execute("""SELECT topic, AVG(difficulty_rating) as avg_diff, COUNT(*) as cnt
                     FROM study_sessions WHERE user_id = ? AND difficulty_rating <= 2
                     GROUP BY topic HAVING cnt >= 2 ORDER BY avg_diff ASC LIMIT 5""",
                 (self.user_id,))
        strong = [r[0] for r in c.fetchall()]
        
        c.execute("""UPDATE cognitive_profile SET 
                    avg_session_duration = ?, peak_hours = ?, weak_topics = ?, 
                    strong_topics = ?, last_updated = CURRENT_TIMESTAMP
                    WHERE user_id = ?""",
                 (int(avg_dur), json.dumps(peak_hours), json.dumps(weak), 
                  json.dumps(strong), self.user_id))
        self.conn.commit()
    
    def _generate_feedback(self, description: str, difficulty: int, mood: str) -> Dict:
        """Generate AI-powered feedback based on session."""
        feedback = {
            "session_logged": True,
            "suggestions": [],
            "next_steps": [],
            "struggle_detected": False
        }
        
        desc_lower = description.lower()
        
        # Struggle detection
        struggle_keywords = ["confused", "stuck", "difficult", "hard", "don't understand", 
                            "struggle", "can't", "hard to", "not getting"]
        if any(kw in desc_lower for kw in struggle_keywords):
            feedback["struggle_detected"] = True
            prompt = f"""Student struggled with a study session. They wrote: "{description}"
            Difficulty rating: {difficulty}/5. Mood: {mood}.
            Provide 3 specific suggestions:
            1. What prerequisite topic to review
            2. What study technique to use (Feynman, active recall, interleaving, etc.)
            3. What specific next action to take
            Keep each suggestion under 30 words. Return as JSON."""
            ai_response = query_ollama(prompt)
            try:
                feedback["struggle_suggestions"] = json.loads(ai_response)
            except:
                feedback["struggle_suggestions"] = {"suggestions": ai_response[:500]}
        
        # Technique recommendations
        if difficulty >= 4:
            feedback["suggestions"].append(
                "Try the Feynman Technique: explain the concept in simple terms as if teaching it.")
            feedback["suggestions"].append(
                f"Review prerequisites for '{description[:50]}' - you may have a knowledge gap.")
        elif difficulty <= 2:
            feedback["suggestions"].append(
                "You're grasping this well! Try interleaved practice with a harder topic.")
        
        # Spaced repetition scheduling
        feedback["next_steps"].append("Schedule a review in 1 day (Ebbinghaus curve).")
        feedback["next_steps"].append("Create 3-5 flashcards for key concepts.")
        
        return feedback
    
    def get_profile(self) -> Dict:
        c = self.conn.cursor()
        c.execute("SELECT * FROM cognitive_profile WHERE user_id = ?", (self.user_id,))
        row = c.fetchone()
        if row:
            return dict(row)
        return {}
    
    def get_sessions(self, days: int = 30) -> List[Dict]:
        c = self.conn.cursor()
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        c.execute("""SELECT * FROM study_sessions 
                    WHERE user_id = ? AND date(timestamp) >= ?
                    ORDER BY timestamp DESC""", (self.user_id, cutoff))
        return [dict(row) for row in c.fetchall()]
    
    def get_study_streak(self) -> Dict:
        c = self.conn.cursor()
        c.execute("""SELECT date(timestamp) as study_date, COUNT(*) as sessions
                    FROM study_sessions WHERE user_id = ?
                    GROUP BY date(timestamp) ORDER BY study_date DESC""", (self.user_id,))
        dates = [r[0] for r in c.fetchall()]
        
        streak = 0
        today = datetime.now().date()
        for i, d in enumerate(dates):
            expected = today - timedelta(days=i)
            if d == expected.strftime("%Y-%m-%d"):
                streak += 1
            else:
                break
        
        return {"current_streak": streak, "total_sessions": len(dates)}
    
    def get_subject_distribution(self) -> Dict:
        c = self.conn.cursor()
        c.execute("""SELECT subject, SUM(duration_minutes) as total_mins, COUNT(*) as sessions
                    FROM study_sessions WHERE user_id = ?
                    GROUP BY subject ORDER BY total_mins DESC""", (self.user_id,))
        return {r[0]: {"minutes": r[1], "sessions": r[2]} for r in c.fetchall()}
    
    def close(self):
        self.conn.close()

# ============================================================================
# RAG ENGINE - Retrieval Augmented Generation
# ============================================================================

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False

try:
    import chromadb
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False


class RAGEngine:
    """RAG system using local embeddings and ChromaDB."""
    
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.model = None
        self.client = None
        self.collection = None
        self._init_chroma()
    
    def _init_chroma(self):
        if not HAS_CHROMA:
            return
        try:
            self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            self.collection = self.client.get_or_create_collection("study_knowledge")
        except Exception as e:
            print(f"ChromaDB init error: {e}")
    
    def _get_embedding_model(self):
        if not HAS_ST or self.model is None:
            try:
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
            except:
                pass
        return self.model
    
    def add_documents(self, chunks: List[Dict]):
        """Add document chunks to knowledge base."""
        if not self.collection:
            # Fallback: store in SQLite only
            c = self.conn.cursor()
            for chunk in chunks:
                c.execute("""INSERT INTO knowledge_chunks 
                            (source_type, source_name, content, metadata)
                            VALUES (?, ?, ?, ?)""",
                         (chunk["source_type"], chunk["source_name"], 
                          chunk["content"], json.dumps(chunk)))
            self.conn.commit()
            return
        
        model = self._get_embedding_model()
        if not model:
            # No embeddings, store raw
            c = self.conn.cursor()
            for chunk in chunks:
                c.execute("""INSERT INTO knowledge_chunks 
                            (source_type, source_name, content, metadata)
                            VALUES (?, ?, ?, ?)""",
                         (chunk["source_type"], chunk["source_name"], 
                          chunk["content"], json.dumps(chunk)))
            self.conn.commit()
            return
        
        # Add to ChromaDB
        texts = [c["content"] for c in chunks]
        ids = [f"{c['source_name']}_{c.get('page',0)}_{c.get('chunk_idx',0)}" 
               for c in chunks]
        metadatas = [{"source_type": c["source_type"], 
                      "source_name": c["source_name"],
                      "page": c.get("page", 0)} for c in chunks]
        
        embeddings = model.encode(texts).tolist()
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        # Also store in SQLite
        c = self.conn.cursor()
        for i, chunk in enumerate(chunks):
            c.execute("""INSERT INTO knowledge_chunks 
                        (source_type, source_name, content, embedding_id, metadata)
                        VALUES (?, ?, ?, ?, ?)""",
                     (chunk["source_type"], chunk["source_name"], 
                      chunk["content"], ids[i], json.dumps(chunk)))
        self.conn.commit()
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for relevant chunks."""
        # First try ChromaDB
        if self.collection:
            try:
                model = self._get_embedding_model()
                if model:
                    query_embedding = model.encode([query]).tolist()
                    results = self.collection.query(
                        query_embeddings=query_embedding,
                        n_results=n_results,
                        include=["documents", "metadatas"]
                    )
                    docs = results["documents"][0] if results["documents"] else []
                    metas = results["metadatas"][0] if results["metadatas"] else []
                    return [{"content": d, "metadata": m} for d, m in zip(docs, metas)]
            except Exception as e:
                print(f"ChromaDB search error: {e}")
        
        # Fallback: SQLite keyword search
        c = self.conn.cursor()
        c.execute("""SELECT content, metadata FROM knowledge_chunks 
                    WHERE content LIKE ? OR content LIKE ?
                    ORDER BY length(content) ASC LIMIT ?""",
                 (f"%{query}%", f"%{query.lower()}%", n_results))
        return [{"content": r[0], "metadata": json.loads(r[1]) if r[1] else {}} 
                for r in c.fetchall()]
    
    def chat_with_context(self, user_query: str, context: str = "") -> str:
        """Chat with AI using retrieved context (RAG)."""
        # Retrieve relevant documents
        docs = self.search(user_query)
        context_text = "\n\n".join([d["content"] for d in docs]) if docs else ""
        
        prompt = f"""You are an expert study tutor for an ISC Class 12 student (PCMB + English).
Use the following retrieved context from the student's textbooks and notes to answer.
If the context doesn't contain the answer, use your general knowledge.
Explain concepts clearly with examples. Use LaTeX for math formulas.

Retrieved context:
{context_text[:3000]}

Student's question: {user_query}

Answer:"""
        
        return query_ollama(prompt)
    
    def close(self):
        self.conn.close()


# ============================================================================
# SPACED REPETITION ENGINE (Anki SM-2 Algorithm)
# ============================================================================

class SpacedRepetition:
    """Implements SuperMemo-2 spaced repetition algorithm."""
    
    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
    
    def add_card(self, topic: str, content: str) -> int:
        """Add a new flashcard."""
        c = self.conn.cursor()
        next_review = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        c.execute("""INSERT INTO spaced_repetition 
                    (user_id, topic, content, difficulty, interval_days, next_review)
                    VALUES (?, ?, ?, 2.5, 1, ?)""",
                 (self.user_id, topic, content, next_review))
        self.conn.commit()
        return c.lastrowid
    
    def get_due_cards(self) -> List[Dict]:
        """Get cards due for review today."""
        c = self.conn.cursor()
        today = datetime.now().strftime("%Y-%m-%d")
        c.execute("""SELECT * FROM spaced_repetition 
                    WHERE user_id = ? AND date(next_review) <= ?
                    ORDER BY next_review ASC""", (self.user_id, today))
        return [dict(row) for row in c.fetchall()]
    
    def review_card(self, card_id: int, quality: int) -> Dict:
        """Rate a card (0-5 scale) and schedule next review using SM-2."""
        c = self.conn.cursor()
        c.execute("SELECT * FROM spaced_repetition WHERE id = ? AND user_id = ?",
                 (card_id, self.user_id))
        row = c.fetchone()
        if not row:
            return {"error": "Card not found"}
        
        row = dict(row)
        
        # SM-2 Algorithm
        q = quality  # 0-5 rating
        old_e = row["difficulty"]
        n = row["repetitions"]
        i = row["interval_days"]
        
        # Update easiness
        new_e = old_e + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
        new_e = max(1.3, min(new_e, 2.5))
        
        if q >= 3:
            if n == 0:
                new_i = 1
            elif n == 1:
                new_i = 6
            else:
                new_i = int(i * new_e)
            new_n = n + 1
        else:
            new_n = 0
            new_i = 1
        
        next_review = (datetime.now() + timedelta(days=new_i)).strftime("%Y-%m-%d")
        last_reviewed = datetime.now().strftime("%Y-%m-%d")
        
        c.execute("""UPDATE spaced_repetition SET 
                    difficulty = ?, interval_days = ?, repetitions = ?,
                    next_review = ?, last_reviewed = ?
                    WHERE id = ? AND user_id = ?""",
                 (new_e, new_i, new_n, next_review, last_reviewed, card_id, self.user_id))
        self.conn.commit()
        
        return {
            "success": True,
            "new_interval": new_i,
            "new_easiness": round(new_e, 2),
            "repetitions": new_n,
            "next_review": next_review
        }
    
    def get_stats(self) -> Dict:
        c = self.conn.cursor()
        c.execute("SELECT COUNT(*) FROM spaced_repetition WHERE user_id = ?",
                 (self.user_id,))
        total = c.fetchone()[0]
        
        c.execute("SELECT COUNT(*) FROM spaced_repetition 
                  WHERE user_id = ? AND repetitions > 0",
                 (self.user_id,))
        mastered = c.fetchone()[0]
        
        due = len(self.get_due_cards())
        
        return {
            "total_cards": total,
            "mastered": mastered,
            "due_today": due,
            "mastery_percent": round(mastered / total * 100, 1) if total > 0 else 0
        }
    
    def close(self):
        self.conn.close()


# ============================================================================
# VISUALIZATION ENGINE (Manim + 3Blue1Brown style)
# ============================================================================

class VisualizationEngine:
    """Generates 3Blue1Brown-style mathematical animations."""
    
    TEMPLATE_DIR = BASE_DIR / "templates" / "manim"
    OUTPUT_DIR = DATA_DIR / "videos"
    
    def __init__(self):
        Path(self.TEMPLATE_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    def create_derivative_scene(self) -> str:
        scene = '''from manim import *\n\nclass Derivative(Scene):\n    def construct(self):\n        axes = Axes(x_range=[-3, 3], y_range=[-1, 5])\n        curve = axes.plot(lambda x: x**2, color=BLUE)\n        self.add(axes, curve)\n        self.play(Create(axes), Create(curve))\n        tangent = axes.plot(lambda x: 4*x - 4, color=RED)\n        self.play(Create(tangent))\n        self.wait()\n'''[1:]  # Remove leading newline
        return self._save_and_render("derivative", scene)
    
    def create_wave_scene(self) -> str:
        scene = '''from manim import *\nimport numpy as np\n\nclass WaveMotion(Scene):\n    def construct(self):\n        axes = Axes(x_range=[0, 4], y_range=[-2, 2])\n        wave = always_redraw(lambda: axes.plot(\n            lambda x: np.sin(x), color=GREEN\n        ))\n        self.add(axes, wave)\n        self.play(\n            axes.animate.set(x_range=[0, 8]),\n            run_time=3\n        )\n        self.wait()\n'''[1:]
        return self._save_and_render("wave", scene)
    
    def create_projectile_scene(self) -> str:
        scene = '''from manim import *\n\nclass ProjectileMotion(Scene):\n    def construct(self):\n        axes = Axes(x_range=[0, 10], y_range=[0, 6])\n        trajectory = axes.plot(lambda x: x - 0.1*x**2, color=ORANGE)\n        dot = Dot(axes.c2p(0, 0), color=WHITE)\n        self.add(axes, trajectory, dot)\n        self.play(MoveAlongPath(dot, trajectory), run_time=3)\n        self.wait()\n'''[1:]
        return self._save_and_render("projectile", scene)
    
    def create_torque_scene(self) -> str:
        scene = '''from manim import *\n\nclass TorqueVisualization(Scene):\n    def construct(self):\n        pivot = Dot(ORIGIN, color=YELLOW)\n        lever = Line(ORIGIN, RIGHT*3, color=WHITE)\n        force = Arrow(RIGHT*3, RIGHT*3 + UP*2, color=RED)\n        self.add(pivot, lever, force)\n        self.play(Rotate(lever, PI/4, about_point=ORIGIN))\n        self.wait()\n'''[1:]
        return self._save_and_render("torque", scene)
    
    def _save_and_render(self, name: str, scene_code: str) -> str:
        scene_file = self.TEMPLATE_DIR / f"{name}.py"
        scene_file.write_text(scene_code)
        output_file = self.OUTPUT_DIR / f"{name}.mp4"
        
        # Try to render with manim
        try:
            import subprocess
            result = subprocess.run(
                ["manim", "-qh", str(scene_file), 
                 scene_code.split("class ")[1].split("(")[0]],
                capture_output=True, text=True, timeout=60
            )
            return str(output_file)
        except Exception as e:
            print(f"Manim render error: {e}")
            return f"Scene saved at {scene_file} - render manually with: manim -qh {scene_file} {name}"
    
    def list_templates(self) -> List[str]:
        return [f.stem for f in self.TEMPLATE_DIR.glob("*.py")]


# ============================================================================
# DIAGNOSTIC / QUESTIONNAIRE ENGINE
# ============================================================================

class DiagnosticEngine:
    """Creates Bloom's Taxonomy-based diagnostic questionnaires."""
    
    BLOOM_LEVELS = [
        ("Remember", "Recall facts and basic concepts"),
        ("Understand", "Explain ideas or concepts"),
        ("Apply", "Use information in new situations"),
        ("Analyze", "Draw connections among ideas"),
        ("Evaluate", "Justify a stand or decision"),
        ("Create", "Produce new or original work")
    ]
    
    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    
    def generate_diagnostic(self, topic: str) -> Dict:
        """Generate AI-powered diagnostic quiz for a topic."""
        prompt = f"""Generate a 5-question diagnostic quiz on: {topic}
Follow Bloom's Taxonomy progression (1 question per level):
1. Remember: Simple recall question
2. Understand: Explain a concept  
3. Apply: Solve a problem
4. Analyze: Compare/contrast concepts
5. Evaluate: Justify or critique an approach

Return ONLY valid JSON in this exact format:
{{"topic": "topic_name", "questions": [
  {{"id": 1, "level": "Remember", "question": "...", "options": ["a)", "b)", "c)", "d)"], "answer": "a)"}},
  {{"id": 2, "level": "Understand", "question": "...", "options": [...], "answer": "..."}},
  ... for all 5 levels
]}}"""
        
        response = query_ollama(prompt)
        
        # Parse JSON from response
        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                quiz = json.loads(json_match.group())
            else:
                quiz = {"error": "Could not parse quiz", "raw": response[:500]}
        except:
            quiz = {"error": "Failed to generate quiz", "raw": response[:500]}
        
        # Store in DB
        c = self.conn.cursor()
        c.execute("""INSERT INTO diagnostic_results 
                    (user_id, topic, questions, score, level)
                    VALUES (?, ?, ?, 0, 'pending')""",
                 (self.user_id, topic, json.dumps(quiz)))
        self.conn.commit()
        
        return quiz
    
    def submit_diagnostic(self, topic: str, answers: Dict) -> Dict:
        """Submit answers and get analysis."""
        c = self.conn.cursor()
        c.execute("""SELECT questions FROM diagnostic_results 
                    WHERE user_id = ? AND topic = ?
                    ORDER BY id DESC LIMIT 1""", (self.user_id, topic))
        row = c.fetchone()
        
        if not row:
            return {"error": "No quiz found"}
        
        quiz = json.loads(row[0])
        questions = quiz.get("questions", [])
        
        # Score the quiz
        correct = 0
        level_scores = {}
        for q in questions:
            user_ans = answers.get(str(q["id"]), "")
            is_correct = user_ans.strip() == q["answer"].strip()
            if is_correct:
                correct += 1
            level = q.get("level", "Unknown")
            level_scores[level] = level_scores.get(level, {"correct": 0, "total": 0})
            level_scores[level]["total"] += 1
            level_scores[level]["correct"] += int(is_correct)
        
        score = correct / len(questions) * 100
        
        # Determine level
        if score >= 80:
            level = "Advanced"
        elif score >= 50:
            level = "Intermediate"
        else:
            level = "Beginner"
        
        # Update DB
        c.execute("""UPDATE diagnostic_results SET score = ?, level = ?, answers = ?
                    WHERE user_id = ? AND topic = ?""",
                 (score, level, json.dumps(answers), self.user_id, topic))
        self.conn.commit()
        
        # Generate AI feedback
        feedback_prompt = f"""Student scored {score}% on diagnostic for: {topic}
Level: {level}
Topic breakdown: {json.dumps(level_scores)}

Provide personalized feedback:
1. What they did well
2. Specific weak areas
3. Prerequisites to review (if beginner)
4. Next study steps
5. Recommended study technique
Return as JSON."""
        ai_feedback = query_ollama(feedback_prompt)
        
        return {
            "score": score,
            "level": level,
            "correct": correct,
            "total": len(questions),
            "level_breakdown": level_scores,
            "ai_feedback": ai_feedback
        }
    
    def close(self):
        self.conn.close()


# ============================================================================
# FLASK API ROUTES
# ============================================================================

# Global instances
profile_engine = None
rag_engine = None
sr_engine = None
diag_engine = None
viz_engine = None
chat_history = []


def get_engines():
    global profile_engine, rag_engine, sr_engine, diag_engine, viz_engine
    if profile_engine is None:
        profile_engine = CognitiveProfile()
    if rag_engine is None:
        rag_engine = RAGEngine()
    if sr_engine is None:
        sr_engine = SpacedRepetition()
    if diag_engine is None:
        diag_engine = DiagnosticEngine()
    if viz_engine is None:
        viz_engine = VisualizationEngine()
    return profile_engine, rag_engine, sr_engine, diag_engine, viz_engine


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})


# --- Study Session Routes ---

@app.route("/api/session/log", methods=["POST"])
def log_session():
    data = request.json
    profile, *_ = get_engines()
    feedback = profile.log_session(
        topic=data.get("topic", "General"),
        subject=data.get("subject", "General"),
        duration=data.get("duration", 30),
        description=data.get("description", ""),
        mood=data.get("mood", "neutral"),
        difficulty=data.get("difficulty", 3)
    )
    return jsonify(feedback)


@app.route("/api/sessions", methods=["GET"])
def get_sessions():
    days = request.args.get("days", 30, type=int)
    profile, *_ = get_engines()
    return jsonify({
        "sessions": profile.get_sessions(days),
        "streak": profile.get_study_streak(),
        "distribution": profile.get_subject_distribution()
    })


@app.route("/api/profile", methods=["GET"])
def get_profile():
    profile, *_ = get_engines()
    return jsonify(profile.get_profile())


# --- Chat / RAG Routes ---

@app.route("/api/chat", methods=["POST"])
def chat():
    global chat_history
    data = request.json
    user_msg = data.get("message", "")
    topic = data.get("topic", "General")
    use_rag = data.get("use_rag", True)
    
    rag, *_ = get_engines()
    
    if use_rag:
        response = rag.chat_with_context(user_msg)
    else:
        response = query_ollama(user_msg)
    
    # Save to history
    chat_history.append({"role": "user", "content": user_msg, "topic": topic})
    chat_history.append({"role": "assistant", "content": response, "topic": topic})
    
    # Keep history manageable
    chat_history = chat_history[-20:]
    
    return jsonify({
        "response": response,
        "history": chat_history[-10:]
    })


# --- File Upload Routes ---

@app.route("/api/upload/pdf", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    source_name = data.get("source_name", file.filename)
    
    # Save file
    file_path = UPLOAD_DIR / file.filename
    file.save(str(file_path))
    
    # Process chunks
    chunks = process_pdf(str(file_path), source_name)
    
    _, rag, *_ = get_engines()
    rag.add_documents(chunks)
    
    return jsonify({
        "success": True,
        "chunks_processed": len(chunks),
        "source_name": source_name,
        "message": f"Processed {len(chunks)} chunks from {file.filename}"
    })


@app.route("/api/upload/youtube", methods=["POST"])
def upload_youtube():
    data = request.json
    url = data.get("url")
    
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    
    result = process_youtube(url)
    
    if result:
        # Store transcript in knowledge base
        chunks = [{
            "source_type": "youtube",
            "source_name": result["title"],
            "content": result["transcript"],
            "metadata": {"duration": result["duration"]}
        }]
        _, rag, *_ = get_engines()
        rag.add_documents(chunks)
        
        return jsonify({
            "success": True,
            "title": result["title"],
            "duration": result["duration"],
            "transcript_length": len(result["transcript"])
        })
    else:
        return jsonify({"error": "Failed to process video"}), 500


# --- Spaced Repetition Routes ---

@app.route("/api/sr/cards", methods=["GET"])
def get_cards():
    _, _, sr, *_ = get_engines()
    return jsonify({
        "due": sr.get_due_cards(),
        "stats": sr.get_stats()
    })


@app.route("/api/sr/add", methods=["POST"])
def add_card():
    data = request.json
    _, _, sr, *_ = get_engines()
    card_id = sr.add_card(
        topic=data.get("topic", "General"),
        content=data.get("content", "")
    )
    return jsonify({"card_id": card_id, "success": True})


@app.route("/api/sr/review", methods=["POST"])
def review_card():
    data = request.json
    _, _, sr, *_ = get_engines()
    result = sr.review_card(
        card_id=data.get("card_id"),
        quality=data.get("quality", 3)
    )
    return jsonify(result)


# --- Diagnostic Routes ---

@app.route("/api/diagnostic/generate", methods=["POST"])
def generate_diagnostic():
    data = request.json
    _, _, _, diag, *_ = get_engines()
    quiz = diag.generate_diagnostic(data.get("topic", "General"))
    return jsonify(quiz)


@app.route("/api/diagnostic/submit", methods=["POST"])
def submit_diagnostic():
    data = request.json
    _, _, _, diag, *_ = get_engines()
    result = diag.submit_diagnostic(
        topic=data.get("topic"),
        answers=data.get("answers", {})
    )
    return jsonify(result)


# --- Visualization Routes ---

@app.route("/api/visualize", methods=["POST"])
def visualize():
    data = request.json
    viz_type = data.get("type", "derivative")
    _, _, _, _, viz = get_engines()
    
    if viz_type == "derivative":
        result = viz.create_derivative_scene()
    elif viz_type == "wave":
        result = viz.create_wave_scene()
    elif viz_type == "projectile":
        result = viz.create_projectile_scene()
    elif viz_type == "torque":
        result = viz.create_torque_scene()
    else:
        result = viz.list_templates()
    
    return jsonify({"result": result})


# --- Search Routes ---

@app.route("/api/search", methods=["GET"])
def search_knowledge():
    query = request.args.get("q", "")
    n = request.args.get("n", 5, type=int)
    _, rag, *_ = get_engines()
    results = rag.search(query, n)
    return jsonify({"results": results})


# --- Ollama Management ---

@app.route("/api/ollama/status", methods=["GET"])
def ollama_status():
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = resp.json().get("models", [])
        return jsonify({
            "status": "running",
            "models": [m["name"] for m in models]
        })
    except:
        return jsonify({
            "status": "not_running",
            "message": f"Start Ollama with: ollama pull {LLM_MODEL}",
            "recommended_model": LLM_MODEL
        })


# --- Static Files ---

@app.route("/videos/<path:filename>")
def serve_video(filename):
    return send_from_directory(str(viz_engine.OUTPUT_DIR), filename) if viz_engine else ""


@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(str(UPLOAD_DIR), filename)


# ============================================================================
# SOCKET.IO EVENTS (Real-time updates)
# ============================================================================

@socketio.on("connect")
def handle_connect():
    emit("status", {"message": "Connected to Neural Study OS"})


@socketio.on("study_update")
def handle_study_update(data):
    # Broadcast study session to all connected clients
    broadcast("new_session", data)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  NEURAL STUDY OS - Local AI Study System")
    print("  Optimized for RTX 3050 / Ryzen 5500H")
    print("="*60)
    
    init_db()
    
    print("\nInitializing engines...")
    profile_engine = CognitiveProfile()
    rag_engine = RAGEngine()
    sr_engine = SpacedRepetition()
    diag_engine = DiagnosticEngine()
    viz_engine = VisualizationEngine()
    print("All engines ready.")
    
    print("\nChecking Ollama...")
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        print(f"Ollama running. Available models: {models}")
    except:
        print(f"Ollama not running. Run: ollama pull {LLM_MODEL}")
    
    print("\nStarting server at http://localhost:5000")
    print("Open http://localhost:5000 in your browser.")
    print("="*60 + "\n")
    
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
