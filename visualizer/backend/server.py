"""
Main server file for the visualizer backend.
"""

import argparse
import sys
from pathlib import Path

# Add BindCraft root to path
bindcraft_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(bindcraft_root))

from visualizer.backend.api.routes import app, init_parser

def main():
    parser = argparse.ArgumentParser(description="Evolution Visualizer Backend Server")
    parser.add_argument(
        "--evolution-dir",
        type=str,
        default="evolution_output",
        help="Path to evolution output directory"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to bind to"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode"
    )
    
    args = parser.parse_args()
    
    # Initialize parser
    print(f"Initializing parser with evolution directory: {args.evolution_dir}")
    init_parser(args.evolution_dir)
    
    # Run Flask app
    print(f"Starting server on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()

