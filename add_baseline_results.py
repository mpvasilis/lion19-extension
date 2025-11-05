

import json
import argparse
import pandas as pd


def load_current_results(csv_path='results/hcar_pipeline_results.csv'):
    
    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        print(f"[ERROR] Results file not found: {csv_path}")
        print("Please run 'python generate_results_table.py' first.")
        return None


def update_baseline_metrics(df, benchmark, alq=None, alt=None, paq=None, pat=None):
    

    mask = df['Prob.'] == benchmark
    
    if not mask.any():
        print(f"[ERROR] Benchmark '{benchmark}' not found in results table.")
        print(f"Available benchmarks: {', '.join(df['Prob.'].tolist())}")
        return None

    if alq is not None:
        df.loc[mask, 'ALQ'] = alq
    
    if alt is not None:
        df.loc[mask, 'ALT(s)'] = alt
    
    if paq is not None:
        df.loc[mask, 'PAQ'] = paq
    
    if pat is not None:
        df.loc[mask, 'PAT(s)'] = pat
    
    return df


def save_updated_results(df, output_dir='results'):
    

    csv_path = f"{output_dir}/hcar_pipeline_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"[SAVED] Updated CSV: {csv_path}")

    json_path = f"{output_dir}/hcar_pipeline_results.json"
    df.to_json(json_path, orient='records', indent=2)
    print(f"[SAVED] Updated JSON: {json_path}")

    txt_path = f"{output_dir}/hcar_pipeline_results.txt"
    with open(txt_path, 'w') as f:
        f.write("HCAR Pipeline Results (with Baselines)\n")
        f.write("="*80 + "\n\n")
        f.write(df.to_string(index=False))
    print(f"[SAVED] Updated Text: {txt_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Add baseline results to HCAR pipeline results table'
    )
    
    parser.add_argument(
        '--benchmark',
        type=str,
        required=True,
        help='Benchmark name (e.g., Sudoku, Sudoku-GT, JSudoku)'
    )
    
    parser.add_argument(
        '--alq',
        type=int,
        help='Total queries for pure Active Learning baseline (MQuAcq-2 only)'
    )
    
    parser.add_argument(
        '--alt',
        type=float,
        help='Total time for pure Active Learning baseline (seconds)'
    )
    
    parser.add_argument(
        '--paq',
        type=int,
        help='Total queries for Passive+Active baseline (no refinement)'
    )
    
    parser.add_argument(
        '--pat',
        type=float,
        help='Total time for Passive+Active baseline (seconds)'
    )
    
    args = parser.parse_args()

    if all(arg is None for arg in [args.alq, args.alt, args.paq, args.pat]):
        print("[ERROR] At least one baseline metric must be provided.")
        print("Use --alq, --alt, --paq, or --pat")
        return

    print(f"Loading current results...")
    df = load_current_results()
    
    if df is None:
        return

    print(f"Updating baseline metrics for {args.benchmark}...")
    df = update_baseline_metrics(
        df, 
        args.benchmark,
        alq=args.alq,
        alt=args.alt,
        paq=args.paq,
        pat=args.pat
    )
    
    if df is None:
        return

    save_updated_results(df)

    print(f"\n[SUCCESS] Baseline metrics updated for {args.benchmark}")
    print("\nUpdated row:")
    print(df[df['Prob.'] == args.benchmark].to_string(index=False))


if __name__ == "__main__":
    main()

