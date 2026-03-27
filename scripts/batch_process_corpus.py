#!/usr/bin/env python3
"""
Batch Process Full ICLE Corpus with ASC-analyzer

Processes 9,529 documents from icle_dataset.csv:
1. Extracts texts to individual files
2. Runs ASC-analyzer in batches
3. Tracks progress and handles errors
4. Generates processing report

Estimated time: ~13 hours with MPS acceleration (~5 sec/doc)
"""

import csv
import subprocess
import time
from pathlib import Path
from datetime import datetime, timedelta
import json
import sys


class CorpusProcessor:
    """Batch processor for ICLE corpus with ASC-analyzer."""

    def __init__(self, csv_path, output_dir, asc_output_dir, batch_size=100):
        """
        Initialize processor.

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
            'batch_times': []
        }

        # Log file
        self.log_file = self.asc_output_dir / 'processing_log.txt'
        self.progress_file = self.asc_output_dir / 'progress.json'

    def log(self, message):
        """Log message to file and console."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_msg + '\n')

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

    def process_batch(self, batch_files):
        """
        Process a batch of files with ASC-analyzer.

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

        # Run ASC-analyzer
        start_time = time.time()

        try:
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
                timeout=self.batch_size * 10  # 10 sec per doc max
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
            # Clean up
            import shutil
            if batch_dir.exists():
                shutil.rmtree(batch_dir)
        except Exception as e:
            elapsed = time.time() - start_time
            self.log(f"✗ Batch error: {e}")
            success_count = 0
            error_count = len(batch_files)
            # Clean up
            import shutil
            if batch_dir.exists():
                shutil.rmtree(batch_dir)

        return success_count, error_count, elapsed

    def run_asc_analyzer(self):
        """Run ASC-analyzer on all extracted texts in batches."""
        self.log("="*80)
        self.log("STEP 2: Running ASC-analyzer")
        self.log("="*80)

        text_files = sorted(self.output_dir.glob('*.txt'))
        total_files = len(text_files)

        self.log(f"Found {total_files} text files to process")
        self.log(f"Batch size: {self.batch_size}")
        self.log(f"Estimated time: {(total_files * 5) / 3600:.1f} hours")
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
            self.log(f"  ⏱ Time: {elapsed:.1f}s ({time_per_doc:.2f}s/doc)")
            self.log(f"  📊 Progress: {self.stats['processed']}/{total_files} ({progress_pct:.1f}%)")
            self.log(f"  🕐 ETA: {eta}")
            self.log("")

            # Save progress
            self.save_progress()

        self.stats['end_time'] = datetime.now()
        self.log("✓ ASC-analyzer processing complete")

    def generate_report(self):
        """Generate final processing report."""
        self.log("="*80)
        self.log("PROCESSING REPORT")
        self.log("="*80)

        duration = self.stats['end_time'] - self.stats['start_time']
        avg_time_per_doc = duration.total_seconds() / self.stats['processed']

        self.log(f"\nDataset: {self.csv_path.name}")
        self.log(f"Total documents: {self.stats['total_docs']}")
        self.log(f"Successfully processed: {self.stats['processed']}")
        self.log(f"Errors: {len(self.stats['errors'])}")
        self.log(f"\nStart time: {self.stats['start_time']}")
        self.log(f"End time: {self.stats['end_time']}")
        self.log(f"Total duration: {duration}")
        self.log(f"Average time per document: {avg_time_per_doc:.2f}s")

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
    """Run batch processing."""
    # Configuration
    csv_path = '/Users/fatihbozdag/Documents/ConstructionMiner-Clean/icle_dataset.csv'
    output_dir = '/Users/fatihbozdag/Documents/ConstructionMiner-Clean/corpus_texts'
    asc_output_dir = '/Users/fatihbozdag/Documents/ConstructionMiner-Clean/corpus_asc_output'
    batch_size = 100  # Process 100 docs at a time

    # Create processor
    processor = CorpusProcessor(csv_path, output_dir, asc_output_dir, batch_size)

    print("="*80)
    print("ICLE CORPUS BATCH PROCESSOR")
    print("="*80)
    print(f"Dataset: {csv_path}")
    print(f"Output directory: {asc_output_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Estimated time: ~13 hours")
    print("="*80)
    print()

    # Ask for confirmation
    response = input("Start processing? (yes/no): ").strip().lower()
    if response != 'yes':
        print("Processing cancelled.")
        return

    # Run processing
    try:
        # Step 1: Extract texts
        processor.extract_texts()

        # Step 2: Run ASC-analyzer
        processor.run_asc_analyzer()

        # Step 3: Generate report
        processor.generate_report()

        print("\n✓ Processing complete!")

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
