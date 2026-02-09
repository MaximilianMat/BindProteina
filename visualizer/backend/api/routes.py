"""
API routes for the visualizer backend.
"""

from flask import Flask, jsonify, send_file, request
from flask_cors import CORS
from pathlib import Path
import sys
import os
import pandas as pd
import numpy as np
from typing import Optional
import math

# Add parent directory to path to import parsers
bindcraft_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(bindcraft_root))

from visualizer.backend.parsers.evolution_parser import EvolutionParser
from visualizer.backend.parsers.structure_parser import StructureParser
from visualizer.backend.api.models import EvolutionRunModel, StructureInfoModel

# Initialize Flask app with CORS
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Initialize parser (will be set by server.py)
parser: Optional[EvolutionParser] = None


def init_parser(evolution_output_dir: str = "evolution_output"):
    """Initialize the evolution parser."""
    global parser
    parser = EvolutionParser(evolution_output_dir)


@app.route('/api/runs', methods=['GET'])
def list_runs():
    """List all evolution runs."""
    if parser is None:
        return jsonify({"error": "Parser not initialized"}), 500
    
    try:
        runs = parser.list_evolution_runs()
        return jsonify({"runs": runs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/runs/<path:run_name>', methods=['GET'])
def get_run(run_name: str):
    """Get details of a specific evolution run."""
    if parser is None:
        return jsonify({"error": "Parser not initialized"}), 500
    
    try:
        summary = parser.get_design_summary(run_name)
        # Ensure all values are JSON serializable
        import traceback
        try:
            # Test serialization
            import json as json_lib
            json_lib.dumps(summary)
        except (TypeError, ValueError) as e:
            return jsonify({
                "error": f"Serialization error: {str(e)}"
            }), 500
        return jsonify(summary)
    except Exception as e:
        import traceback
        return jsonify({"error": str(e)}), 500


@app.route('/api/runs/<path:run_name>/rounds/<int:round_num>/designs', methods=['GET'])
def get_round_designs(run_name: str, round_num: int):
    """Get all designs for a specific round."""
    if parser is None:
        return jsonify({"error": "Parser not initialized"}), 500
    
    try:
        # Parse specifically the requested round (lazy parsing)
        round_obj = parser.parse_round(run_name, round_num)
        
        if round_obj is None:
            return jsonify({
                "error": f"Round {round_num} not found in run {run_name}"
            }), 404
        
        designs = []
        for design in round_obj.designs:
            # Clean metrics to ensure JSON serializable (handle any remaining NaN)
            cleaned_metrics = {}
            for key, value in design.metrics.items():
                # Check for NaN/None values
                if value is None:
                    cleaned_metrics[key] = None
                elif isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                    cleaned_metrics[key] = None
                elif pd.isna(value):
                    cleaned_metrics[key] = None
                elif isinstance(value, (np.integer, np.int64)):
                    cleaned_metrics[key] = int(value)
                elif isinstance(value, (np.floating, np.float64)):
                    if math.isnan(value) or math.isinf(value):
                        cleaned_metrics[key] = None
                    else:
                        cleaned_metrics[key] = float(value)
                else:
                    cleaned_metrics[key] = value
            
            # Convert Path objects to strings, ensuring they're relative to BindCraft root
            af2_path = None
            relaxed_path = None
            laproteina_path = None
            
            if design.af2_pdb_path:
                af2_path = str(design.af2_pdb_path)
                # Ensure path uses forward slashes (for URL encoding)
                af2_path = af2_path.replace('\\', '/')
            
            if design.relaxed_pdb_path:
                relaxed_path = str(design.relaxed_pdb_path)
                # Ensure path uses forward slashes (for URL encoding)
                relaxed_path = relaxed_path.replace('\\', '/')
            
            if design.laproteina_pdb_path:
                laproteina_path = str(design.laproteina_pdb_path)
                # Ensure path uses forward slashes (for URL encoding)
                laproteina_path = laproteina_path.replace('\\', '/')
            
            design_dict = {
                "design_name": design.design_name,
                "model": design.model,
                "round_number": design.round_number,
                "status": design.status,
                "metrics": cleaned_metrics,
                "af2_pdb_path": af2_path,
                "relaxed_pdb_path": relaxed_path,
                "laproteina_pdb_path": laproteina_path
            }
            designs.append(design_dict)
        
        return jsonify({"designs": designs})
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route('/api/runs/<path:run_name>/notes', methods=['GET'])
def get_run_notes(run_name: str):
    """Get notes for a specific evolution run."""
    if parser is None:
        return jsonify({"error": "Parser not initialized"}), 500
    
    try:
        notes = parser.get_run_notes(run_name)
        return jsonify({"notes": notes})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/runs/<path:run_name>/notes', methods=['POST'])
def save_run_notes(run_name: str):
    """Save notes for a specific evolution run."""
    if parser is None:
        return jsonify({"error": "Parser not initialized"}), 500
    
    try:
        data = request.get_json()
        if not data or 'notes' not in data:
            return jsonify({"error": "No notes provided"}), 400
            
        success = parser.save_run_notes(run_name, data['notes'])
        if success:
            return jsonify({"status": "success"})
        else:
            return jsonify({"error": "Failed to save notes"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/structure/<path:file_path>', methods=['GET'])
def get_structure_info(file_path: str):
    """Get information about a structure file."""
    try:
        # Decode URL-encoded path
        file_path = file_path.replace('%2F', '/')
        info = StructureParser.get_structure_info(file_path)
        return jsonify(info)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/pdb/<path:file_path>', methods=['GET'])
def serve_pdb(file_path: str):
    """Serve a PDB file."""
    try:
        # Decode URL-encoded path
        file_path = file_path.replace('%2F', '/')
        
        # Get BindCraft root directory
        if parser:
            bindcraft_root = parser.evolution_output_dir.parent.resolve()
        else:
            # Fallback: assume evolution_output is in current directory
            bindcraft_root = Path.cwd()
        
        # Resolve the path relative to BindCraft root
        # Handle both relative paths (with or without ..) and absolute paths
        if Path(file_path).is_absolute():
            pdb_path = Path(file_path)
        else:
            # Resolve relative to BindCraft root
            pdb_path = (bindcraft_root / file_path).resolve()
        
        # Security check: ensure path is within BindCraft directory
        try:
            pdb_path.resolve().relative_to(bindcraft_root.resolve())
        except ValueError:
            return jsonify({
                "error": "Access denied: path outside allowed directory",
                "requested_path": file_path,
                "resolved_path": str(pdb_path),
                "bindcraft_root": str(bindcraft_root)
            }), 403
        
        if not pdb_path.exists():
            return jsonify({
                "error": f"File not found: {pdb_path}",
                "resolved_path": str(pdb_path),
                "original_path": file_path,
                "bindcraft_root": str(bindcraft_root),
                "debug": {
                    "file_path_decoded": file_path,
                    "is_absolute": Path(file_path).is_absolute(),
                    "pdb_path_resolved": str(pdb_path.resolve())
                }
            }), 404
        
        return send_file(str(pdb_path), mimetype='text/plain')
    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc(),
            "file_path": file_path
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok", "parser_initialized": parser is not None})

