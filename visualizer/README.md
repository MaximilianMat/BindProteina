# BindCraft Evolution Visualizer

A web-based 3D visualization platform for exploring and analyzing binder evolution results.

## Features

- **3D Structure Viewer**: Interactive protein visualization using NGL Viewer
- **Evolution Progress**: View designs across multiple evolution rounds
- **Metrics Display**: Show scoring metrics (i_pTM, dG, dSASA, etc.)
- **Animation Support**: Animate evolution progress and conformational changes
- **Interactive Exploration**: Filter, sort, and compare designs

## Setup

### Backend

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
cd visualizer/backend
python server.py --evolution-dir ../evolution_output --port 5000
```

### Frontend

1. Open `frontend/index.html` in a web browser, or serve it with a simple HTTP server:
```bash
cd visualizer/frontend
python -m http.server 8000
```

2. Open `http://localhost:8000` in your browser

## Usage

1. Start the backend server (see Setup above)
2. Open the frontend in a web browser
3. Select an evolution run from the dropdown
4. Click "Load Run" to load the evolution data
5. Click on rounds to see designs
6. Click on designs to view them in the 3D viewer

## API Endpoints

- `GET /api/runs` - List all evolution runs
- `GET /api/runs/<run_name>` - Get details of a specific run
- `GET /api/runs/<run_name>/rounds/<round_num>/designs` - Get designs for a round
- `GET /api/structure/<file_path>` - Get structure information
- `GET /api/pdb/<file_path>` - Serve PDB file
- `GET /api/health` - Health check

## File Structure

The visualizer expects evolution output in the following structure:
```
evolution_output/
└── evolution_YYYYMMDD_HHMM/
    └── round_N/
        └── scored/
            ├── scoring_summary.csv
            ├── af2_predictions/
            └── relaxed/
```

