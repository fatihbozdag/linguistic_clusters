#!/usr/bin/env python3
"""
Process LOCNESS Dataset Through ConstructionMiner Pipeline

Steps:
1. Extract texts from CSV to individual files
2. Run ASC-analyzer to generate ASCinfo files
3. Run full corpus analysis pipeline
"""

import csv
import subprocess
import time
from pathlib import Path
from datetime import datetime
import sys
import shutil

# Configuration
CSV_PATH = '/Users/fatihbozdag/Documents/ConstructionMiner-Clean/locness_dataset.csv'
TEXTS_DIR = '/Users/fatihbozdag/Documents/ConstructionMiner-Clean/locness_texts'
ASC_OUTPUT_DIR = '/Users/fatihbozdag/Documents/ConstructionMiner-Clean/locness_asc_output'
RESULTS_DIR = '/Users/fatihbozdag/Documents/ConstructionMiner-Clean/locness_results'
BATCH_SIZE = 50  # Process 50 files at a time


def log(msg):
    """Log with timestamp."""
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {msg}")


def extract_texts():
    """Extract texts from CSV to individual files."""
    log("="*70)
    log("STEP 1: Extracting texts from CSV")
    log("="*70)

    texts_dir = Path(TEXTS_DIR)
    texts_dir.mkdir(exist_ok=True)

    count = 0
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            doc_id = row['file_name']  # LOCNESS uses file_name column
            text = row['text_field']    # LOCNESS uses text_field column

            # Write text file
            text_file = texts_dir / f"{doc_id}.txt"
            with open(text_file, 'w', encoding='utf-8') as out:
                out.write(text)

            count += 1
            if count % 500 == 0:
                log(f"  Extracted {count} texts...")

    log(f"✓ Extracted {count} texts to {texts_dir}")
    return count


def run_asc_analyzer(batch_start=0):
    """Run ASC-analyzer on all texts."""
    log("="*70)
    log("STEP 2: Running ASC-analyzer")
    log("="*70)

    texts_dir = Path(TEXTS_DIR)
    asc_dir = Path(ASC_OUTPUT_DIR)
    asc_dir.mkdir(exist_ok=True)

    # Get all text files
    text_files = sorted(texts_dir.glob('*.txt'))
    total_files = len(text_files)

    log(f"Found {total_files} text files")
    log(f"Batch size: {BATCH_SIZE}")

    start_time = time.time()
    processed = 0

    # Process in batches
    for i in range(batch_start, total_files, BATCH_SIZE):
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (total_files + BATCH_SIZE - 1) // BATCH_SIZE

        batch_files = text_files[i:i + BATCH_SIZE]

        log(f"\nBatch {batch_num}/{total_batches} ({len(batch_files)} files)...")

        # Create temp batch directory
        batch_dir = texts_dir / f"_batch_temp_{batch_num}"
        batch_dir.mkdir(exist_ok=True)

        # Copy files to batch directory
        for f in batch_files:
            shutil.copy(f, batch_dir / f.name)

        # Run ASC-analyzer
        batch_start_time = time.time()
        try:
            cmd = [
                sys.executable, '-m', 'asc_analyzer.cli',
                '--input-dir', str(batch_dir),
                '--save-asc-output',
                '--source', 'cow'
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=BATCH_SIZE * 30  # 30 sec per doc max
            )

            batch_time = time.time() - batch_start_time

            # Move ASC output files
            for asc_file in batch_dir.glob('*_ASCinfo.txt'):
                shutil.move(str(asc_file), asc_dir / asc_file.name)

            processed += len(batch_files)

            # Progress
            pct = (processed / total_files) * 100
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta_sec = (total_files - processed) / rate if rate > 0 else 0
            eta_min = int(eta_sec / 60)

            log(f"  ✓ Processed: {processed}/{total_files} ({pct:.1f}%), "
                f"Batch time: {batch_time:.1f}s, ETA: {eta_min} min")

        except Exception as e:
            log(f"  ✗ Batch error: {e}")
        finally:
            # Cleanup batch dir
            shutil.rmtree(batch_dir, ignore_errors=True)

    total_time = time.time() - start_time
    log(f"\n✓ ASC-analyzer complete: {processed} files in {total_time/60:.1f} minutes")

    return processed


def run_pipeline():
    """Run the full corpus analysis pipeline."""
    log("="*70)
    log("STEP 3: Running Full Corpus Analysis Pipeline")
    log("="*70)

    # Import pipeline modules
    sys.path.insert(0, str(Path(__file__).parent / 'src'))
    from run_full_corpus_analysis import CorpusAnalysisPipeline

    # Create and run pipeline
    pipeline = CorpusAnalysisPipeline(
        asc_output_dir=ASC_OUTPUT_DIR,
        results_dir=RESULTS_DIR,
        mode='production'
    )

    pipeline.run_pipeline()

    log("✓ Pipeline complete!")


def main():
    """Run full LOCNESS processing."""
    log("\n" + "="*70)
    log("LOCNESS CORPUS PROCESSING - ConstructionMiner")
    log("="*70)
    log(f"Dataset: {CSV_PATH}")
    log(f"Texts dir: {TEXTS_DIR}")
    log(f"ASC output: {ASC_OUTPUT_DIR}")
    log(f"Results: {RESULTS_DIR}")
    log("="*70)

    # Check what's already done
    texts_dir = Path(TEXTS_DIR)
    asc_dir = Path(ASC_OUTPUT_DIR)

    existing_texts = len(list(texts_dir.glob('*.txt'))) if texts_dir.exists() else 0
    existing_asc = len(list(asc_dir.glob('*_ASCinfo.txt'))) if asc_dir.exists() else 0

    log(f"\nExisting files: {existing_texts} texts, {existing_asc} ASC files")

    try:
        # Step 1: Extract texts (if needed)
        if existing_texts < 8000:
            extract_texts()
        else:
            log(f"\n✓ Texts already extracted ({existing_texts} files)")

        # Step 2: Run ASC-analyzer (if needed)
        if existing_asc < 8000:
            run_asc_analyzer(batch_start=existing_asc // BATCH_SIZE * BATCH_SIZE)
        else:
            log(f"\n✓ ASC files already generated ({existing_asc} files)")

        # Step 3: Run pipeline
        run_pipeline()

    except KeyboardInterrupt:
        log("\n\n⚠ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        log(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
