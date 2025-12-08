"""
Cleanup Script for ExamTT_v2 Old Data Files
============================================
This script removes all pickle files generated with the incorrect 30x25 grid dimensions
for the examtt_v2 benchmark. After running this script, you'll need to regenerate the
data with the correct 8x8 dimensions.

Usage:
    python cleanup_examtt_v2_old_data.py [--dry-run]

Options:
    --dry-run    Show what would be deleted without actually deleting
"""

import os
import sys
import glob
import argparse
from pathlib import Path


def find_examtt_v2_files():
    """Find all examtt_v2 pickle files in the project."""
    patterns = [
        'phase1_output/examtt_v2*.pkl',
        'phase2_output/examtt_v2*.pkl',
        'phase2_output/examtt_v2_*.pkl',
        'phase2_output_cop/examtt_v2*.pkl',
        'phase2_output_cop/examtt_v2/**/*.pkl',
        'phase2_output_lion/examtt_v2*.pkl',
        'phase2_output_lion/examtt_v2/**/*.pkl',
        'phase3_output_cop/examtt_v2*.pkl',
        'phase3_output_cop/examtt_v2/**/*.pkl',
        'phase3_output_lion/examtt_v2*.pkl',
        'phase3_output_lion/examtt_v2/**/*.pkl',
        'solution_variance_output/examtt_v2*/**/*.pkl',
    ]
    
    files_to_delete = []
    for pattern in patterns:
        matching_files = glob.glob(pattern, recursive=True)
        files_to_delete.extend(matching_files)
    
    # Remove duplicates and sort
    files_to_delete = sorted(set(files_to_delete))
    
    return files_to_delete


def format_size(size_bytes):
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def main():
    parser = argparse.ArgumentParser(
        description='Cleanup old examtt_v2 pickle files with incorrect dimensions'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be deleted without actually deleting'
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("ExamTT_v2 Data Cleanup Script")
    print("=" * 70)
    print()
    
    # Find all files
    files_to_delete = find_examtt_v2_files()
    
    if not files_to_delete:
        print("✅ No examtt_v2 pickle files found. Nothing to clean up.")
        return 0
    
    # Calculate total size
    total_size = 0
    file_info = []
    for filepath in files_to_delete:
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            total_size += size
            file_info.append((filepath, size))
    
    # Display what will be deleted
    print(f"Found {len(file_info)} examtt_v2 pickle files:")
    print()
    
    # Group by directory
    by_dir = {}
    for filepath, size in file_info:
        dirname = os.path.dirname(filepath)
        if dirname not in by_dir:
            by_dir[dirname] = []
        by_dir[dirname].append((os.path.basename(filepath), size))
    
    for dirname in sorted(by_dir.keys()):
        files = by_dir[dirname]
        dir_size = sum(size for _, size in files)
        print(f"\n📁 {dirname}/  [{format_size(dir_size)}]")
        for filename, size in sorted(files):
            print(f"   - {filename} ({format_size(size)})")
    
    print()
    print("=" * 70)
    print(f"Total: {len(file_info)} files, {format_size(total_size)}")
    print("=" * 70)
    print()
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No files were deleted")
        print()
        print("To actually delete these files, run:")
        print("    python cleanup_examtt_v2_old_data.py")
        return 0
    
    # Confirm deletion
    print("⚠️  WARNING: This will permanently delete all these files!")
    print()
    response = input("Are you sure you want to delete these files? [y/N]: ")
    
    if response.lower() not in ['y', 'yes']:
        print()
        print("❌ Cancelled. No files were deleted.")
        return 1
    
    print()
    print("Deleting files...")
    
    # Delete files
    deleted_count = 0
    failed_count = 0
    
    for filepath, _ in file_info:
        try:
            os.remove(filepath)
            deleted_count += 1
            print(f"  ✓ Deleted: {filepath}")
        except Exception as e:
            failed_count += 1
            print(f"  ✗ Failed to delete {filepath}: {e}")
    
    # Clean up empty directories
    print()
    print("Cleaning up empty directories...")
    
    dirs_to_check = set(os.path.dirname(f) for f, _ in file_info)
    for dirname in sorted(dirs_to_check, reverse=True):
        try:
            if os.path.exists(dirname) and not os.listdir(dirname):
                os.rmdir(dirname)
                print(f"  ✓ Removed empty directory: {dirname}")
        except Exception as e:
            print(f"  ✗ Could not remove directory {dirname}: {e}")
    
    print()
    print("=" * 70)
    print(f"✅ Cleanup complete!")
    print(f"   Deleted: {deleted_count} files ({format_size(total_size)})")
    if failed_count > 0:
        print(f"   Failed: {failed_count} files")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Regenerate Phase 1 data:")
    print("     python phase1_passive_learning.py --benchmark examtt_v2")
    print()
    print("  2. Run your experiments as usual")
    print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

