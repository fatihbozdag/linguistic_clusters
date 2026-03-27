#!/usr/bin/env python3
"""
Process missing files through ASC-analyzer.
Processes the 1,931 files that haven't been converted to ASC format yet.
Uses ASC-analyzer CLI to generate _ASCinfo.txt files.
ROBUST: Handles broken pipes and interruptions gracefully.
"""
import subprocess
import shutil
import time
import signal
import sys
import os
from pathlib import Path

# Ignore SIGPIPE to prevent broken pipe crashes
signal.signal(signal.SIGPIPE, signal.SIG_IGN)

def safe_print(msg):
    """Print with broken pipe handling."""
    try:
        print(msg)
        sys.stdout.flush()
    except (BrokenPipeError, IOError):
        sys.stdout = open(os.devnull, 'w')
        pass

# Directories
corpus_texts_dir = Path('/Users/fatihbozdag/Documents/ConstructionMiner-Clean/corpus_texts')
corpus_asc_output_dir = Path('/Users/fatihbozdag/Documents/ConstructionMiner-Clean/corpus_asc_output')
missing_files_list = Path('/tmp/missing_files.txt')

# Batch processing parameters
BATCH_SIZE = 100  # Process 100 files at a time

# Read missing files
with open(missing_files_list, 'r') as f:
    missing_files = [line.strip() for line in f if line.strip()]

safe_print(f"Processing {len(missing_files)} missing files...")
safe_print(f"Output directory: {corpus_asc_output_dir}")
safe_print(f"Batch size: {BATCH_SIZE}")
safe_print("")

# Process in batches
processed = 0
errors = 0
total_batches = (len(missing_files) + BATCH_SIZE - 1) // BATCH_SIZE

for batch_idx in range(0, len(missing_files), BATCH_SIZE):
    batch_num = (batch_idx // BATCH_SIZE) + 1
    batch_files = missing_files[batch_idx:batch_idx + BATCH_SIZE]

    safe_print(f"[Batch {batch_num}/{total_batches}] Processing {len(batch_files)} files...")

    # Create temporary batch directory
    batch_dir = Path(f'/tmp/asc_batch_{int(time.time())}')
    batch_dir.mkdir(exist_ok=True)

    # Copy files to batch directory
    batch_copied = 0
    for file_id in batch_files:
        input_file = corpus_texts_dir / f"{file_id}.txt"
        if input_file.exists():
            shutil.copy(input_file, batch_dir / f"{file_id}.txt")
            batch_copied += 1
        else:
            print(f"  ⚠️  File not found: {file_id}.txt")
            errors += 1

    # Run ASC-analyzer CLI
    try:
        start_time = time.time()

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
            timeout=BATCH_SIZE * 10  # 10 sec per doc max
        )

        elapsed = time.time() - start_time

        # Move ASC output files to main output directory
        moved_count = 0
        for asc_file in batch_dir.glob('*_ASCinfo.txt'):
            shutil.move(str(asc_file), corpus_asc_output_dir / asc_file.name)
            moved_count += 1

        processed += moved_count

        # Clean up batch directory
        shutil.rmtree(batch_dir)

        print(f"  ✓ Processed {moved_count} files in {elapsed:.1f}s ({elapsed/batch_copied:.2f}s/doc)")

        if result.returncode != 0:
            print(f"  ⚠️  Warnings: {result.stderr[:200]}")

    except subprocess.TimeoutExpired:
        print(f"  ✗ Batch timeout after {BATCH_SIZE * 10}s")
        errors += batch_copied
        if batch_dir.exists():
            shutil.rmtree(batch_dir)

    except Exception as e:
        print(f"  ✗ Batch error: {e}")
        errors += batch_copied
        if batch_dir.exists():
            shutil.rmtree(batch_dir)

    # Progress update
    progress_pct = (processed / len(missing_files)) * 100
    safe_print(f"  📊 Progress: {processed}/{len(missing_files)} ({progress_pct:.1f}%)")
    safe_print()

# Final report
safe_print("="*60)
safe_print("✓ Processing Complete!")
safe_print("="*60)
safe_print(f"Processed: {processed}")
safe_print(f"Errors: {errors}")
safe_print(f"Total ASC files: {len(list(corpus_asc_output_dir.glob('*_ASCinfo.txt')))}")
safe_print()
