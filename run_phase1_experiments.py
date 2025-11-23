

import os
import sys
from phase1_passive_learning import run_phase1

def main():
    
    
    benchmarks = [
        {
            'name': 'sudoku',
            'description': 'Regular 9x9 Sudoku',
            'num_examples': 5,
            'num_overfitted': 10
        },
        {
            'name': 'sudoku_gt',
            'description': 'Sudoku with Greater-Than Constraints',
            'num_examples': 5,
            'num_overfitted': 18  
        },
        {
            'name': 'sudoku_4x4_gt',
            'description': '4x4 Sudoku with Greater-Than Constraints',
            'num_examples': 5,
            'num_overfitted': 10
        },
        {
            'name': 'jsudoku',
            'description': 'Jigsaw Sudoku (9x9 with irregular regions)',
            'num_examples': 5,
            'num_overfitted': 20  
        },
        {
            'name': 'latin_square',
            'description': 'Latin Square (9x9 - rows and columns only)',
            'num_examples': 5,
            'num_overfitted': 10  
        },
        {
            'name': 'graph_coloring_register',
            'description': 'Graph Coloring - Register Allocation (realistic compiler problem)',
            'num_examples': 5,
            'num_overfitted': 10  
        },
        {
            'name': 'examtt_v1',
            'description': 'Exam Timetabling Variant 1 (Small)',
            'num_examples': 5,
            'num_overfitted': 10  
        },
        {
            'name': 'examtt_v2',
            'description': 'Exam Timetabling Variant 2 (Large)',
            'num_examples': 5,
            'num_overfitted': 10  
        }
    ]
    
    output_dir = 'phase1_output'
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("PHASE 1 BATCH RUNNER")
    print("="*80)
    print(f"\nRunning Phase 1 for {len(benchmarks)} benchmarks:")
    for i, bench in enumerate(benchmarks, 1):
        print(f"  {i}. {bench['name']}: {bench['description']}")
    print(f"\nOutput directory: {output_dir}")
    print("="*80)
    
    results = []
    
    for i, bench in enumerate(benchmarks, 1):
        print(f"\n\n{'='*80}")
        print(f"BENCHMARK {i}/{len(benchmarks)}: {bench['name']}")
        print(f"Description: {bench['description']}")
        print(f"{'='*80}\n")
        
        try:
            output_path = run_phase1(
                benchmark_name=bench['name'],
                output_dir=output_dir,
                num_examples=bench['num_examples'],
                num_overfitted=bench['num_overfitted']
            )
            
            if output_path:
                results.append({
                    'benchmark': bench['name'],
                    'status': 'SUCCESS',
                    'output': output_path
                })
                print(f"\n[SUCCESS] {bench['name']} completed!")
            else:
                results.append({
                    'benchmark': bench['name'],
                    'status': 'FAILED',
                    'output': None
                })
                print(f"\n[FAILED] {bench['name']} did not complete!")
        
        except Exception as e:
            results.append({
                'benchmark': bench['name'],
                'status': 'ERROR',
                'output': str(e)
            })
            print(f"\n[ERROR] {bench['name']} raised exception:")
            print(f"  {str(e)}")
            import traceback
            traceback.print_exc()

    print("\n\n" + "="*80)
    print("PHASE 1 BATCH RUNNER - FINAL SUMMARY")
    print("="*80)
    
    success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
    failed_count = sum(1 for r in results if r['status'] == 'FAILED')
    error_count = sum(1 for r in results if r['status'] == 'ERROR')
    
    print(f"\nTotal benchmarks: {len(benchmarks)}")
    print(f"  [+] Successful: {success_count}")
    print(f"  [-] Failed: {failed_count}")
    print(f"  [!] Errors: {error_count}")
    
    print("\nDetailed results:")
    for result in results:
        status_symbol = "[+]" if result['status'] == 'SUCCESS' else "[-]"
        print(f"  {status_symbol} {result['benchmark']}: {result['status']}")
        if result['output']:
            print(f"      => {result['output']}")
    
    print("\n" + "="*80)

    if success_count == len(benchmarks):
        print("\nALL BENCHMARKS COMPLETED SUCCESSFULLY!")
        return 0
    else:
        print(f"\n{failed_count + error_count} BENCHMARK(S) FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())

