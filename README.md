# Netflix Insight Studio

An Interactive Entertainment Analytics and Content Intelligence Platform.

## Included
- React + TypeScript + Vite + Tailwind CSS frontend
- FastAPI + Pandas + NumPy + scikit-learn backend
- Jupyter DAV notebook using the supplied dataset
- Content Explorer
- Genre and country analytics
- Release/addition trends
- Data Laboratory
- ML Studio
- TF-IDF + cosine-similarity recommendation engine
- Project report, syllabus mapping and viva preparation

## Dataset
`data/netflix_titles.csv.zip` is the supplied archive. The verified CSV contains **8,807 rows and 12 columns**:
`show_id, type, title, director, cast, country, date_added, release_year, rating, duration, listed_in, description`.

## Windows setup

### 1. Backend
From the project root:
```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r backend\requirements.txt
uvicorn app.main:app --reload --app-dir backend
```
Backend: http://localhost:8000  
API docs: http://localhost:8000/docs

### 2. Frontend
Open a second terminal:
```bat
cd frontend
npm install
npm run dev
```
Frontend: http://localhost:5173

### 3. Notebook
Install Jupyter and notebook dependencies:
```bat
pip install jupyter numpy pandas matplotlib seaborn plotly scikit-learn ipywidgets folium openpyxl
jupyter notebook notebooks/Netflix_DAV_Project.ipynb
```
In VS Code, install the Python and Jupyter extensions and open the notebook directly.

## Analytical rules
- Counts use the phrase **most represented**, not most popular.
- Release year and Netflix addition year are kept separate.
- Genre and country counts represent multi-valued associations.
- No external APIs or fabricated catalog information are used.
- Recommendation is metadata-only TF-IDF/cosine similarity.
- ML models are trained dynamically and scores are not hardcoded.

## Structure
```text
Netflix_Insight_Studio/
├── frontend/
├── backend/
├── notebooks/
├── shared/
├── data/
├── docs/
├── outputs/
└── README.md
```
