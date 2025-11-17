# Command-line interface for Event Camera Simulation
"""
Professional command-line interface with argument parsing,
interactive mode, and comprehensive error handling.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import tkinter as tk
from tkinter import filedialog, messagebox

logger = logging.getLogger(__name__)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create comprehensive command-line argument parser"""
    parser = argparse.ArgumentParser(
        description="Event Camera Data Simulation - Convert videos to neuromorphic event streams",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (recommended)
  python -m event_sim
  
  # Process specific video with default settings
  python -m event_sim --video video.mp4
  
  # Custom processing parameters
  python -m event_sim --video video.mp4 --threshold 10 --max-frames 200
  
  # Optimized for small objects (ping pong balls)
  python -m event_sim --video ping_pong.mp4 --threshold 6 --small-objects
  
  # Fast processing mode
  python -m event_sim --video video.mp4 --threshold 25 --fast
        """
    )
    
    # Input options
    input_group = parser.add_argument_group('Input Options')
    input_group.add_argument(
        '--video', '-v',
        type=str,
        help='Path to input video file'
    )
    input_group.add_argument(
        '--config', '-c',
        type=str,
        help='Path to custom configuration file'
    )
    
    # Processing parameters
    processing_group = parser.add_argument_group('Processing Parameters')
    processing_group.add_argument(
        '--threshold', '-t',
        type=int,
        help='Event detection threshold (3-50, default: 15)'
    )
    processing_group.add_argument(
        '--max-frames', '-f',
        type=int,
        help='Maximum frames to process (default: 300)'
    )
    processing_group.add_argument(
        '--time-window',
        type=float,
        help='Time window for event accumulation in ms (default: 30.0)'
    )
    
    # Feature flags
    features_group = parser.add_argument_group('Feature Options')
    features_group.add_argument(
        '--no-multi-scale',
        action='store_true',
        help='Disable multi-scale processing'
    )
    features_group.add_argument(
        '--no-motion-boost',
        action='store_true', 
        help='Disable motion magnitude boosting'
    )
    features_group.add_argument(
        '--no-noise',
        action='store_true',
        help='Disable noise generation'
    )
    
    # Preset modes
    presets_group = parser.add_argument_group('Preset Modes')
    presets_group.add_argument(
        '--small-objects',
        action='store_true',
        help='Optimize for small fast objects (ping pong balls)'
    )
    presets_group.add_argument(
        '--fast',
        action='store_true',
        help='Fast processing mode with higher threshold'
    )
    presets_group.add_argument(
        '--detailed',
        action='store_true',
        help='Detailed analysis mode with lower threshold'
    )
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Output directory for generated files'
    )
    output_group.add_argument(
        '--no-video',
        action='store_true',
        help='Skip event video generation'
    )
    output_group.add_argument(
        '--no-analysis',
        action='store_true',
        help='Skip analysis and heatmap generation'
    )
    
    # Utility options
    utility_group = parser.add_argument_group('Utility Options')
    utility_group.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Force interactive mode even with video argument'
    )
    utility_group.add_argument(
        '--verbose', '-V',
        action='count',
        default=0,
        help='Increase verbosity (-V for INFO, -VV for DEBUG)'
    )
    utility_group.add_argument(
        '--version',
        action='version',
        version='Event Camera Simulation v1.0.0'
    )
    
    return parser


def configure_logging(verbosity_level: int) -> None:
    """Configure logging based on verbosity level"""
    if verbosity_level >= 2:
        level = logging.DEBUG
    elif verbosity_level >= 1:
        level = logging.INFO
    else:
        level = logging.WARNING
    
    logging.getLogger('event_sim').setLevel(level)
    logger.info(f"Logging level set to {logging.getLevelName(level)}")


def apply_preset_configurations(args: argparse.Namespace) -> Dict[str, Any]:
    """Apply preset configurations based on command-line flags"""
    overrides = {}
    
    if args.small_objects:
        logger.info("Applying small objects preset (optimized for ping pong balls)")
        overrides.update({
            'threshold': 6,
            'enable_multi_scale': True,
            'motion_boost': True,
            'time_window_ms': 20.0
        })
    
    elif args.fast:
        logger.info("Applying fast processing preset")
        overrides.update({
            'threshold': 25,
            'enable_multi_scale': False,
            'time_window_ms': 50.0
        })
    
    elif args.detailed:
        logger.info("Applying detailed analysis preset")  
        overrides.update({
            'threshold': 10,
            'enable_multi_scale': True,
            'motion_boost': True,
            'time_window_ms': 15.0
        })
    
    # Apply explicit argument overrides
    if args.threshold is not None:
        overrides['threshold'] = args.threshold
    if args.max_frames is not None:
        overrides['max_frames'] = args.max_frames
    if args.time_window is not None:
        overrides['time_window_ms'] = args.time_window
    
    # Apply feature flags
    if args.no_multi_scale:
        overrides['enable_multi_scale'] = False
    if args.no_motion_boost:
        overrides['motion_boost'] = False
    if args.no_noise:
        overrides['noise_level'] = 0.0
    
    return overrides


def interactive_video_selection() -> Optional[str]:
    """Interactive video file selection using GUI dialog"""
    logger.info("Opening interactive video selection dialog")
    
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        # Supported video formats
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.webm', '.m4v', '.flv']
        video_filter = " ".join(f"*{ext}" for ext in video_extensions)
        
        filetypes = [
            ("All Video Files", video_filter),
            ("MP4 Files", "*.mp4"),
            ("AVI Files", "*.avi"),
            ("MOV Files", "*.mov"),
            ("All Files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Video for Event Camera Simulation",
            filetypes=filetypes,
            initialdir=str(Path.cwd())
        )
        
        root.destroy()
        
        if filename:
            logger.info(f"Video selected: {filename}")
            return filename
        else:
            logger.info("No video file selected")
            return None
            
    except Exception as e:
        logger.error(f"Error in video selection: {e}")
        return None


def validate_video_file(video_path: str) -> dict:
    """Validate video file and get properties"""
    import cv2
    
    file_path = Path(video_path)
    
    # Check file existence
    if not file_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Check file extension
    supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.webm', '.m4v', '.flv']
    if file_path.suffix.lower() not in supported_formats:
        supported = ', '.join(supported_formats)
        raise ValueError(f"Unsupported video format: {file_path.suffix}. Supported formats: {supported}")
    
    # Validate with OpenCV
    cap = cv2.VideoCapture(str(file_path))
    
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Cannot open video file: {video_path}")
    
    try:
        # Extract video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Validate properties
        if any(prop <= 0 for prop in [fps, total_frames, width, height]):
            raise ValueError("Invalid video properties detected")
        
        duration_seconds = total_frames / fps if fps > 0 else 0
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        video_info = {
            'file_path': str(file_path),
            'file_size_mb': file_size_mb,
            'width': width,
            'height': height,
            'fps': fps,
            'total_frames': total_frames,
            'duration_seconds': duration_seconds,
            'resolution': f"{width}x{height}",
            'megapixels': (width * height) / 1_000_000
        }
        
        logger.info(f"Video validated: {video_info['resolution']}, {duration_seconds:.1f}s, {fps:.1f}fps")
        
        return video_info
    
    finally:
        cap.release()


def process_video_with_configuration(video_path: str, config_overrides: Dict[str, Any], args: argparse.Namespace) -> bool:
    """Process video with the given configuration"""
    try:
        from .core.event_detector import EventDetector

        
        logger.info(f"🎬 Processing video: {Path(video_path).name}")
        
        # Validate video file
        video_info = validate_video_file(video_path)
        
        # Initialize event detector
        detector = EventDetector(
            threshold=config_overrides.get('threshold', 15),
            enable_multi_scale=config_overrides.get('enable_multi_scale', True),
            motion_boost=config_overrides.get('motion_boost', True),
            noise_level=config_overrides.get('noise_level', 0.01)
        )
        
        # Process video
        logger.info("🔄 Starting event detection...")
        
        def progress_callback(progress_info):
            """Progress callback for user feedback"""
            print(f"\rProgress: {progress_info['percentage']:.1f}% "
                  f"({progress_info['current_step']}/{progress_info['total_steps']})", end='')
        
        events_df = detector.process_video_stream(
            video_path,
            max_frames=config_overrides.get('max_frames', 300),
            progress_callback=progress_callback
        )
        
        print()  # New line after progress
        
        if len(events_df) == 0:
            logger.warning("No events detected! Try lowering the threshold.")
            return False
        
        # Export results
        output_dir = Path(config_overrides.get('output_dir', 'output'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export CSV
        logger.info("💾 Exporting event data...")
        csv_path = output_dir / 'events.csv'
        events_df.to_csv(csv_path, index=False)
        
        # Create simple visualizations
        if not args.no_video:
            logger.info("🎨 Creating color-coded event video...")
            # Basic video creation (simplified for self-contained version)
            create_event_video(events_df, video_info, output_dir / 'event_video.mp4')
        
        if not args.no_analysis:
            logger.info("📊 Generating analysis...")
            create_analysis_plots(events_df, video_info, output_dir / 'analysis_report.png')
        
        # Print success summary
        _print_processing_summary(events_df, detector.get_processing_statistics(), output_dir)
        
        return True
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return False


def create_event_video(events_df, video_info, output_path):
    """Create basic color-coded event video"""
    import cv2
    import numpy as np
    
    width, height = video_info['width'], video_info['height']
    fps = 30
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Simple temporal windows
    time_span = events_df['time'].max() - events_df['time'].min()
    num_frames = int(time_span / 1000000 * fps)  # Convert microseconds to seconds
    
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Get events in current time window
        start_time = events_df['time'].min() + i * (time_span / num_frames)
        end_time = start_time + (time_span / num_frames)
        
        current_events = events_df[
            (events_df['time'] >= start_time) & 
            (events_df['time'] < end_time)
        ]
        
        # Draw events
        for _, event in current_events.iterrows():
            x, y = int(event['x']), int(event['y'])
            if 0 <= x < width and 0 <= y < height:
                if event['polarity'] > 0:
                    frame[y, x] = [0, 0, 255]  # Red for positive
                else:
                    frame[y, x] = [255, 0, 0]  # Blue for negative
        
        out.write(frame)
    
    out.release()
    logger.info(f"Event video saved: {output_path}")


def create_analysis_plots(events_df, video_info, output_path):
    """Create basic analysis plots"""
    import matplotlib.pyplot as plt
    import numpy as np

    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Event scatter plot
    pos_events = events_df[events_df['polarity'] > 0]
    neg_events = events_df[events_df['polarity'] < 0]
    
    axes[0, 0].scatter(pos_events['x'], pos_events['y'], c='red', s=1, alpha=0.5, label='Positive')
    axes[0, 0].scatter(neg_events['x'], neg_events['y'], c='blue', s=1, alpha=0.5, label='Negative')
    axes[0, 0].set_title('Event Spatial Distribution')
    axes[0, 0].legend()
    axes[0, 0].invert_yaxis()
    
    # Event timeline
    time_bins = np.linspace(events_df['time'].min(), events_df['time'].max(), 50)
    counts, _ = np.histogram(events_df['time'], bins=time_bins)
    axes[0, 1].plot(time_bins[:-1], counts)
    axes[0, 1].set_title('Events Over Time')
    axes[0, 1].set_xlabel('Time (μs)')
    axes[0, 1].set_ylabel('Event Count')
    
    # Polarity distribution
    polarity_counts = events_df['polarity'].value_counts()
    axes[1, 0].bar(['Negative', 'Positive'], [polarity_counts.get(-1, 0), polarity_counts.get(1, 0)], 
                   color=['blue', 'red'])
    axes[1, 0].set_title('Polarity Distribution')
    
    # Magnitude histogram
    axes[1, 1].hist(events_df['magnitude'], bins=30, alpha=0.7)
    axes[1, 1].set_title('Event Magnitude Distribution')
    axes[1, 1].set_xlabel('Magnitude')
    axes[1, 1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Analysis plots saved: {output_path}")


def _print_processing_summary(events_df, processing_stats, output_dir):
    """Print comprehensive processing summary"""
    print("\n" + "="*60)
    print("🎉 EVENT CAMERA SIMULATION COMPLETE!")
    print("="*60)
    
    # Event statistics
    total_events = len(events_df)
    pos_events = len(events_df[events_df['polarity'] > 0])
    neg_events = len(events_df[events_df['polarity'] < 0])
    
    print(f"📊 Results Summary:")
    print(f"   Total events: {total_events:,}")
    print(f"   🔴 Positive (RED): {pos_events:,} ({pos_events/total_events*100:.1f}%)")
    print(f"   🔵 Negative (BLUE): {neg_events:,} ({neg_events/total_events*100:.1f}%)")
    print(f"   Frames processed: {processing_stats['frames_processed']}")
    print(f"   Events per frame: {processing_stats['events_per_frame']:.1f}")
    
    # Output files
    print(f"\n📁 Output Files:")
    output_files = ['events.csv', 'event_video.mp4', 'analysis_report.png']
    for filename in output_files:
        filepath = output_dir / filename
        if filepath.exists():
            file_size = filepath.stat().st_size
            if file_size > 1024*1024:
                size_str = f"{file_size/(1024*1024):.1f}MB"
            else:
                size_str = f"{file_size/1024:.1f}KB"
            print(f"   ✅ {filename}: {filepath} ({size_str})")
    
    print(f"\n🎯 Next Steps:")
    print(f"   1. View the color-coded event video")
    print(f"   2. Analyze the CSV data in Excel/Python")
    print(f"   3. Use data for research or algorithm development")
    print("="*60)


def run_cli():
    """Main CLI runner function"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Configure logging
    configure_logging(args.verbose)
    
    # Apply preset configurations
    config_overrides = apply_preset_configurations(args)
    
    # Determine video file
    video_path = None
    
    if args.video:
        # Use specified video file
        video_path = args.video
        logger.info(f"Using specified video: {video_path}")
    elif args.interactive or not args.video:
        # Interactive mode
        print("🎬 EVENT CAMERA DATA SIMULATION")
        print("=" * 50)
        print("Professional neuromorphic vision simulation system")
        print("🔴 RED = Brightness increase | 🔵 BLUE = Brightness decrease")
        print("=" * 50)
        
        video_path = interactive_video_selection()
        
        if not video_path:
            print("👋 No video selected. Exiting...")
            return
    
    # Validate video file
    if not video_path or not Path(video_path).exists():
        logger.error(f"Video file not found: {video_path}")
        sys.exit(1)
    
    # Override output directory if specified
    if args.output_dir:
        config_overrides['output_dir'] = args.output_dir
    
    # Process the video
    success = process_video_with_configuration(video_path, config_overrides, args)
    
    if success:
        logger.info("Processing completed successfully!")
    else:
        logger.error("Processing failed!")
        sys.exit(1)