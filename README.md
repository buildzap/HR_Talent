# 🧠 HR Talent Matching and Development AI

A fully local, cost-effective HR Talent AI system that matches employees to suitable projects using AI, detects skill gaps, and recommends training courses.

## 🎯 Features

- **Resume Parsing**: Automatically extracts skills and information from resume text
- **AI-Powered Matching**: Uses SentenceTransformers and ChromaDB for intelligent project-employee matching
- **Skill Gap Analysis**: Identifies missing skills and recommends relevant training courses
- **Career Path Visualization**: Provides career trajectory suggestions based on current skills
- **Fully Local**: No data leaves your machine - complete privacy and security
- **Modern UI**: Beautiful, responsive web interface built with Tailwind CSS

## 🛠️ Tech Stack

- **Backend**: Python + FastAPI
- **Database**: SQLite (lightweight local storage)
- **Vector Search**: ChromaDB for semantic similarity
- **AI/ML**: SentenceTransformers (all-MiniLM-L6-v2) for embeddings
- **Frontend**: HTML + Tailwind CSS + JavaScript
- **Visualization**: Chart.js for skill analysis

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download

```bash
# If using git
git clone <repository-url>
cd HR_Talent

# Or download and extract the project files
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download spaCy Model (Optional)

```bash
python -m spacy download en_core_web_sm
```

## 🚀 Usage

### Start the Backend Server

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### Access the Web Interface

Open your browser and navigate to `http://localhost:8000`

### API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation

## 📋 API Endpoints

### Core Endpoints

- `POST /upload_resume` - Upload and process employee resume
- `POST /add_project` - Add new project to the system
- `POST /add_course` - Add new training course
- `GET /match_projects/{employee_id}` - Get project matches for employee
- `GET /skill_gap/{employee_id}` - Analyze skill gaps and get recommendations
- `GET /career_path/{employee_id}` - Get career path suggestions

### Data Endpoints

- `GET /projects` - List all projects
- `GET /courses` - List all courses
- `GET /health` - Health check

## 💡 How to Use

### 1. Upload Resume

1. Go to the "Upload Resume" tab
2. Enter employee name and paste resume text
3. Optionally add work preferences
4. Click "Process Resume & Find Matches"

### 2. View Project Matches

1. After uploading, switch to "View Matches" tab
2. See ranked project matches with similarity scores
3. View skill gap analysis and missing skills
4. Browse recommended training courses

### 3. Career Path Analysis

1. Go to "Career Path" tab
2. View current skill level assessment
3. See suggested career trajectories
4. Get personalized training recommendations

## 🗂️ Project Structure

```
HR_Talent/
├── main.py                 # FastAPI application
├── db.py                   # Database operations
├── vector_store.py         # ChromaDB integration
├── utils.py                # Resume parsing and utilities
├── match.py                # Matching logic
├── requirements.txt        # Python dependencies
├── frontend/
│   └── index.html         # Web interface
├── data/
│   ├── sample_projects.json
│   └── sample_courses.json
└── README.md
```

## 🔧 Configuration

### Database

The system uses SQLite by default. The database file `hr_talent.db` will be created automatically.

### Vector Store

ChromaDB data is stored in the `./chroma_db` directory. This contains the embeddings for semantic search.

### Sample Data

Sample projects and courses are automatically loaded on first startup from the `data/` directory.

## 🎨 Customization

### Adding New Skills

Edit the `technical_skills` and `soft_skills` dictionaries in `utils.py` to add new skill patterns.

### Modifying Matching Logic

Adjust the scoring weights in `match.py` in the `_calculate_overall_score` method.

### UI Customization

Modify `frontend/index.html` to customize the web interface appearance and functionality.

## 🔒 Privacy & Security

- **Fully Local**: All data processing happens on your machine
- **No Cloud Dependencies**: No external API calls or data transmission
- **Open Source**: Complete transparency in how data is processed
- **SQLite**: Lightweight, file-based database with no network access

## 🐛 Troubleshooting

### Common Issues

1. **Port Already in Use**: Change the port in `main.py` or stop other services using port 8000
2. **Memory Issues**: Reduce the number of concurrent embeddings or use a smaller model
3. **Database Errors**: Delete `hr_talent.db` to reset the database
4. **Vector Store Issues**: Delete the `./chroma_db` directory to reset embeddings

### Performance Tips

- Use SSD storage for better ChromaDB performance
- Ensure sufficient RAM for embedding generation
- Consider using a smaller sentence transformer model for lower resource usage

## 📈 Future Enhancements

- [ ] Support for PDF resume uploads
- [ ] Advanced skill categorization
- [ ] Integration with HR systems
- [ ] Batch processing capabilities
- [ ] Advanced analytics and reporting
- [ ] Multi-language support

## 🤝 Contributing

This is an open-source project. Feel free to:

- Report bugs and issues
- Suggest new features
- Submit pull requests
- Improve documentation

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- [SentenceTransformers](https://www.sbert.net/) for embedding models
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Tailwind CSS](https://tailwindcss.com/) for styling

---

**Built with ❤️ for local, privacy-focused HR technology**
