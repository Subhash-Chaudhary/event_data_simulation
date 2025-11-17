# Main entry point for Event Camera Simulation
"""
Entry point module that enables running the application with:
python -m event_sim

Provides both command-line interface and interactive GUI modes.
"""

import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('event_sim.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for the application"""
    try:
        from .cli import run_cli
        
        logger.info("Starting Event Camera Simulation")
        logger.info("=" * 50)
        
        # Run the command-line interface
        run_cli()
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()