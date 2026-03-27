#!/usr/bin/env python3
"""Monitor ASC processing progress and notify when complete."""
import time
from pathlib import Path
from datetime import datetime

corpus_asc_output_dir = Path('/Users/fatihbozdag/Documents/ConstructionMiner-Clean/corpus_asc_output')
target = 9529

print("Monitoring ASC processing...")
print(f"Target: {target} files")
print()

while True:
    count = len(list(corpus_asc_output_dir.glob('*_ASCinfo.txt')))
    progress_pct = (count / target) * 100
    remaining = target - count

    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {count}/{target} files ({progress_pct:.1f}%) - {remaining} remaining")

    if count >= target:
        print()
        print("="*60)
        print("✓✓✓ COMPLETE! All 9,529 files processed! ✓✓✓")
        print("="*60)
        print()
        print("Ready to run ConstructionMiner pipeline on full dataset!")
        break

    time.sleep(30)
