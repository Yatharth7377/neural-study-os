# Neural Study OS

**Local AI-Powered Neural Study System** | Optimized for RTX 3050 / Ryzen 5500H

[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

A complete study operating system that runs **100% locally** on your PC. Upload textbooks (PDFs), process YouTube lectures, chat with AI about your materials, track study sessions, use spaced repetition flashcards, take diagnostic quizzes, and visualize math concepts — all in one dark-themed interface.

## Features

### AI Chat (RAG)
- Ask questions about **your uploaded textbooks and notes**
- Local LLM (Ollama + Llama 3.2 3B) — **no API costs, no data leaves your PC**
- Retrieval-Augmented Generation (RAG) with ChromaDB vector search

### Study Session Logging
- Log every study session with topic, duration, mood, difficulty
- **Struggle detection**: AI analyzes your descriptions and suggests next steps
- Tracks peak study hours, subject distribution, session streaks

### Spaced Repetition (SM-2 Algorithm)
- Create flashcards with topic + content
- Review with Again/Good/Easy ratings
- Automatic interval scheduling (1d → 6d → 14d → 30d → 90d)

### Diagnostic Quizzes
- AI generates 5-question Bloom's Taxonomy quizzes per topic
- Levels: Remember → Understand → Apply → Analyze → Evaluate
- AI-powered feedback with prerequisites and study technique suggestions

### Math Visualizations
- Manim integration for 3Blue1Brown-style animations
- Pre-built templates: Derivatives, Wave Motion, Projectile, Torque
- Links to PhET, GeoGebra, Desmos

### PDF & YouTube Processing
- Upload PDFs → auto-chunked → indexed in vector database
- YouTube URL → Whisper transcription → searchable transcript
- All materials become queryable through RAG

### Cognitive Profile
- Learns your study patterns over time
- Identifies weak topics, strong topics, peak hours
- Adapts AI feedback to your learning style

## Quick Start (3 Commands)

```bash
# 1. Clone
git clone https://github.com/Yatharth7377/neural-study-os.git
cd neural-study-os

# 2. Setup (installs all deps + creates folders)
python scripts/setup.py

# 3. Start Ollama (in a separate terminal)
ollama serve
ollama pull llama3.2:3b

# 4. Run
python main.py
# Open http://localhost:5000
```

## Install Ollama

### Windows
```bash
# Download from https://ollama.com/download/windows
# Then in terminal:
ollama pull llama3.2:3b
```

### Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b
```

### macOS
```bash
brew install ollama
ollama pull llama3.2:3b
```

## Desktop App (Electron)

To build a portable `.exe` (Windows):
```bash
npm install
npm run build-win
```

To run as a desktop app:
```bash
npm start
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Chat with AI (RAG or general) |
| `/api/session/log` | POST | Log a study session |
| `/api/sessions` | GET | Get sessions + stats |
| `/api/profile` | GET | Get cognitive profile |
| `/api/upload/pdf` | POST | Upload & process PDF |
| `/api/upload/youtube` | POST | Process YouTube video |
| `/api/sr/cards` | GET | Get flashcards due today |
| `/api/sr/add` | POST | Add a flashcard |
| `/api/sr/review` | POST | Rate a flashcard (SM-2) |
| `/api/diagnostic/generate` | POST | Generate quiz for topic |
| `/api/diagnostic/submit` | POST | Submit quiz answers |
| `/api/visualize` | POST | Generate math animation |
| `/api/search` | GET | Search knowledge base |
| `/api/ollama/status` | GET | Check AI status |

## Project Structure

```
neural-study-os/
├── main.py           # Flask backend + all engines
├── requirements.txt  # Python dependencies
├── package.json      # Electron config + npm scripts
├── scripts/
│   └── setup.py      # One-command installer
├── electron/
│   └── main.js       # Electron app entry point
├── static/
│   └── index.html    # Dark UI frontend
├── data/             # Auto-created at runtime
│   ├── study.db      # SQLite database
│   ├── chroma_db/    # Vector embeddings
│   ├── uploads/      # PDFs, videos
│   └── videos/       # Manim outputs
└── templates/
    └── manim/        # Animation scene templates
```

## Tech Stack

- **Backend**: Flask, Flask-CORS, Flask-SocketIO
- **Local AI**: Ollama (Llama 3.2 3B), Sentence Transformers
- **RAG**: ChromaDB, LangChain
- **PDF**: PyMuPDF (fitz)
- **YouTube**: yt-dlp, OpenAI Whisper
- **Database**: SQLite
- **Frontend**: Vanilla HTML/CSS/JS (dark theme)
- **Desktop**: Electron
- **Visualizations**: Manim (3Blue1Brown)

## System Requirements

- **CPU**: Ryzen 5 5500H or equivalent
- **GPU**: RTX 3050 (2GB+ VRAM) — model uses ~2-3GB
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free (model + dependencies)
- **Python**: 3.9+
- **Node.js**: 18+
- **FFmpeg**: Required for video processing

## How It Learns About You

1. **Session Logging**: Every study session updates your cognitive profile
2. **Pattern Detection**: Finds your peak hours, optimal session length, weak/strong topics
3. **Struggle Analysis**: Detects keywords like "confused" or "stuck" → AI suggests prerequisites
4. **Adaptive Feedback**: Difficulty 1-2 → "Try harder topics" | Difficulty 4-5 → "Review prerequisites"
5. **Retention Tracking**: Uses Ebbinghaus forgetting curve to schedule reviews

## Supported Exams

Upload your syllabus PDFs for:
- ISC Class 12 (PCMB + English)
- JEE Main / Advanced
- NEET
- IAT / CET
- IB / IGCSE
- SAT / AP

The RAG system will answer questions using **your specific textbooks**.

## License

MIT License — free for personal and commercial use.

## Credits

- [Ollama](https://ollama.com) — Local LLM runtime
- [ChromaDB](https://www.trychroma.com) — Vector database
- [Whisper](https://github.com/openai/whisper) — Transcription
- [Manim](https://www.manim.community) — Math animations
- [PhET](https://phet.colorado.edu) — Interactive simulations
