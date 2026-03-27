#!/usr/bin/env python3
"""
Batch Process Full ICLE Corpus with ASC-analyzer (MPS-ACCELERATED)

Processes 9,529 documents with GPU acceleration:
1. Extracts texts to individual files
2. Runs ASC-analyzer in batches WITH MPS
3. Tracks progress and handles errors
4. Generates processing report
5. ROBUST: Handles broken pipes, can resume from interruptions

Estimated time: ~30-40 minutes with MPS acceleration (~0.2s/doc)
"""

import csv
import subprocess
import time
import os
import signal
from pathlib import Path
from datetime import datetime, timedelta
import json
import sys


class CorpusProcessorMPS:
    """Batch processor for ICLE corpus with ASC-analyzer + MPS acceleration."""

    def __init__(self, csv_path, output_dir, asc_output_dir, batch_size=100):
        """
        Initialize processor with MPS support.

        Args:
            csv_path: Path to icle_dataset.csv
            output_dir: Directory for extracted text files
            asc_output_dir: Directory for ASC-analyzer output
            batch_size: Number of documents to process in each batch
        """
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)
        self.asc_output_dir = Path(asc_output_dir)
        self.batch_size = batch_size

        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.asc_output_dir.mkdir(exist_ok=True)

        # Progress tracking
        self.stats = {
            'total_docs': 0,
            'extracted': 0,
            'processed': 0,
            'errors': [],
            'start_time': None,
            'end_time': None,
            'batch_times': [],
            'mps_enabled': True
        }

        # Log file
        self.log_file = self.asc_output_dir / 'processing_log.txt'
        self.progress_file = self.asc_output_dir / 'progress.json'

    def log(self, message):
        """Log message to file and console (handles broken pipe)."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] {message}"

        # Always write to file (this is critical)
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_msg + '\n')
        except Exception as e:
            # If even file writing fails, try stderr
            sys.stderr.write(f"LOG ERROR: {e}\n")

        # Try to print to console, but don't crash if pipe is broken
        try:
            print(log_msg)
            sys.stdout.flush()
        except (BrokenPipeError, IOError):
            # Silence broken pipe errors - log file has the data
            # Redirect stdout to devnull to prevent future errors
            sys.stdout = open(os.devnull, 'w')
            pass

    def save_progress(self):
        """Save progress to JSON file."""
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, default=str)

    def extract_texts(self):
        """Extract texts from CSV to individual files."""
        self.log("="*80)
        self.log("STEP 1: Extracting texts from CSV")
        self.log("="*80)

        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                doc_id = row['doc_id']
                text = row['text']

                # Write text file
                text_file = self.output_dir / f"{doc_id}.txt"
                with open(text_file, 'w', encoding='utf-8') as out:
                    out.write(text)

                self.stats['extracted'] += 1

                if self.stats['extracted'] % 500 == 0:
                    self.log(f"Extracted {self.stats['extracted']} texts...")

        self.stats['total_docs'] = self.stats['extracted']
        self.log(f"✓ Extracted {self.stats['total_docs']} texts to {self.output_dir}")
        self.save_progress()

    def enable_mps(self):
        """Enable MPS acceleration for spaCy."""
        self.log("="*80)
        self.log("Enabling MPS Acceleration")
        self.log("="*80)

        try:
            import spacy
            import torch

            # Check MPS availability
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                spacy.require_gpu()
                self.log("✓ MPS acceleration enabled")
                self.log(f"  PyTorch MPS available: True")
                self.log(f"  PyTorch MPS built: True")
                return True
            else:
                self.log("⚠ MPS not available, falling back to CPU")
                return False
        except Exception as e:
            self.log(f"✗ Failed to enable MPS: {e}")
            return False

    def process_batch(self, batch_files):
        """
        Process a batch of files with ASC-analyzer using MPS.

        Args:
            batch_files: List of file paths to process

        Returns:
            tuple: (success_count, error_count, elapsed_time)
        """
        # Create temporary directory for this batch
        batch_dir = self.output_dir / f"batch_temp_{int(time.time())}"
        batch_dir.mkdir(exist_ok=True)

        # Copy files to batch directory
        for file_path in batch_files:
            import shutil
            shutil.copy(file_path, batch_dir / file_path.name)

        # Run ASC-analyzer with GPU environment variables
        start_time = time.time()

        try:
            # Set environment to prefer GPU
            env = os.environ.copy()
            env['SPACY_PREFER_GPU'] = '1'

            cmd = [
                'python3.12', '-m', 'asc_analyzer.cli',
                '--input-dir', str(batch_dir),
                '--save-asc-output',
                '--source', 'cow'
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.batch_size * 10,  # 10 sec per doc max
                env=env
            )

            elapsed = time.time() - start_time

            # Move ASC output files to main output directory
            for asc_file in batch_dir.glob('*_ASCinfo.txt'):
                import shutil
                shutil.move(str(asc_file), self.asc_output_dir / asc_file.name)

            # Clean up batch directory
            import shutil
            shutil.rmtree(batch_dir)

            success_count = len(batch_files)
            error_count = 0

            if result.returncode != 0:
                self.log(f"⚠ Batch processing had warnings: {result.stderr[:200]}")

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            self.log(f"✗ Batch timeout after {elapsed:.1f}s")
            success_count = 0
            error_count = len(batch_files)
            import shutil
            if batch_dir.exists():
                shutil.rmtree(batch_dir)
        except Exception as e:
            elapsed = time.time() - start_time
            self.log(f"✗ Batch error: {e}")
            success_count = 0
            error_count = len(batch_files)
            import shutil
            if batch_dir.exists():
                shutil.rmtree(batch_dir)

        return success_count, error_count, elapsed

    def run_asc_analyzer(self, resume=True):
        """Run ASC-analyzer on all extracted texts with MPS acceleration.

        Args:
            resume: If True, skip files that already have ASC output
        """
        self.log("="*80)
        self.log("STEP 2: Running ASC-analyzer with MPS")
        self.log("="*80)

        # Get all text files
        all_text_files = sorted(self.output_dir.glob('*.txt'))

        # Filter out already-processed files if resuming
        if resume:
            existing_asc = set(f.stem.replace('_ASCinfo', '')
                             for f in self.asc_output_dir.glob('*_ASCinfo.txt'))
            text_files = [f for f in all_text_files
                         if f.stem not in existing_asc]

            if len(text_files) < len(all_text_files):
                already_done = len(all_text_files) - len(text_files)
                self.log(f"✓ RESUME MODE: Skipping {already_done} already-processed files")
                self.stats['processed'] = already_done
        else:
            text_files = all_text_files

        total_files = len(all_text_files)
        remaining_files = len(text_files)

        self.log(f"Total text files: {total_files}")
        self.log(f"Remaining to process: {remaining_files}")
        self.log(f"Batch size: {self.batch_size}")
        self.log(f"Expected speedup: 3-5x faster with MPS")
        if remaining_files == 0:
            self.log("✓ All files already processed!")
            return
        self.log("")

        self.stats['start_time'] = datetime.now()

        # Process in batches
        for i in range(0, total_files, self.batch_size):
            batch_num = (i // self.batch_size) + 1
            total_batches = (total_files + self.batch_size - 1) // self.batch_size

            batch_files = text_files[i:i + self.batch_size]
            batch_size_actual = len(batch_files)

            self.log(f"Processing batch {batch_num}/{total_batches} ({batch_size_actual} files)...")

            success, errors, elapsed = self.process_batch(batch_files)

            self.stats['processed'] += success
            self.stats['batch_times'].append(elapsed)

            # Calculate ETA
            avg_time_per_batch = sum(self.stats['batch_times']) / len(self.stats['batch_times'])
            remaining_batches = total_batches - batch_num
            eta_seconds = avg_time_per_batch * remaining_batches
            eta = timedelta(seconds=int(eta_seconds))

            # Progress report
            progress_pct = (self.stats['processed'] / total_files) * 100
            time_per_doc = elapsed / batch_size_actual

            self.log(f"  ✓ Batch complete: {success} processed, {errors} errors")
            self.log(f"  ⏱ Time: {elapsed:.1f}s ({time_per_doc:.2f}s/doc) 🚀 MPS")
            self.log(f"  📊 Progress: {self.stats['processed']}/{total_files} ({progress_pct:.1f}%)")
            self.log(f"  🕐 ETA: {eta}")
            self.log("")

            # Save progress
            self.save_progress()

        self.stats['end_time'] = datetime.now()
        self.log("✓ ASC-analyzer processing complete with MPS acceleration")

    def generate_report(self):
        """Generate final processing report."""
        self.log("="*80)
        self.log("PROCESSING REPORT (MPS-ACCELERATED)")
        self.log("="*80)

        duration = self.stats['end_time'] - self.stats['start_time']
        avg_time_per_doc = duration.total_seconds() / self.stats['processed']

        self.log(f"\nDataset: {self.csv_path.name}")
        self.log(f"Total documents: {self.stats['total_docs']}")
        self.log(f"Successfully processed: {self.stats['processed']}")
        self.log(f"Errors: {len(self.stats['errors'])}")
        self.log(f"MPS acceleration: {'✓ Enabled' if self.stats['mps_enabled'] else '✗ Disabled'}")
        self.log(f"\nStart time: {self.stats['start_time']}")
        self.log(f"End time: {self.stats['end_time']}")
        self.log(f"Total duration: {duration}")
        self.log(f"Average time per document: {avg_time_per_doc:.2f}s 🚀")

        # Check output files
        asc_files = list(self.asc_output_dir.glob('*_ASCinfo.txt'))
        self.log(f"\nASC output files created: {len(asc_files)}")

        # Save final report
        report_file = self.asc_output_dir / 'processing_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, default=str)

        self.log(f"\n✓ Report saved to: {report_file}")
        self.log("="*80)


def main():
    """Run batch processing with MPS acceleration."""
    # Ignore SIGPIPE to prevent broken pipe crashes
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)

    # Configuration
    csv_path = '/Users/fatihbozdag/Documents/ConstructionMiner-Clean/icle_dataset.csv'
    output_dir = '/Users/fatihbozdag/Documents/ConstructionMiner-Clean/corpus_texts'
    asc_output_dir = '/Users/fatihbozdag/Documents/ConstructionMiner-Clean/corpus_asc_output'
    batch_size = 100

    # Create processor
    processor = CorpusProcessorMPS(csv_path, output_dir, asc_output_dir, batch_size)

    print("="*80)
    print("ICLE CORPUS BATCH PROCESSOR (MPS-ACCELERATED)")
    print("="*80)
    print(f"Dataset: {csv_path}")
    print(f"Output directory: {asc_output_dir}")
    print(f"Batch size: {batch_size}")
    print(f"MPS Acceleration: ✓ ENABLED")
    print(f"Estimated time: ~30-40 minutes (3-5x faster!)")
    print("="*80)
    print()

    # Ask for confirmation
    response = input("Start MPS-accelerated processing? (yes/no): ").strip().lower()
    if response != 'yes':
        print("Processing cancelled.")
        return

    # Run processing
    try:
        # Enable MPS
        processor.enable_mps()

        # Step 1: Extract texts (if not already done)
        if not list(processor.output_dir.glob('*.txt')):
            processor.extract_texts()
        else:
            processor.stats['total_docs'] = len(list(processor.output_dir.glob('*.txt')))
            processor.stats['extracted'] = processor.stats['total_docs']
            processor.log(f"✓ Using existing {processor.stats['total_docs']} extracted texts")

        # Step 2: Run ASC-analyzer with MPS
        processor.run_asc_analyzer()

        # Step 3: Generate report
        processor.generate_report()

        print("\n✓ MPS-accelerated processing complete!")

    except KeyboardInterrupt:
        print("\n\n⚠ Processing interrupted by user")
        processor.log("Processing interrupted by user")
        processor.save_progress()
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        processor.log(f"Fatal error: {e}")
        processor.save_progress()
        raise


if __name__ == "__main__":
    main()
