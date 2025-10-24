


import os
import sys
import time
import subprocess
import csv
import json
from datetime import datetime
from typing import List, Dict, Any

BENCHMARKS = [
    "sudoku",
    "examtt", 
    "nurse",
    "uefa",
    "vm_allocation"
]

DEFAULT_CONFIG = {
    "timeout": 600,  
    "use_bayesian": True,
    "use_passive_constraints": False,
    "passive_solutions": None,
    "passive_output_dir": "output"
}

class BenchmarkRunner:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or DEFAULT_CONFIG
        self.results = []
        self.start_time = None
        self.log_file = None
        self.benchmarks = BENCHMARKS
        
    def setup_logging(self):
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = "benchmark_logs"
        os.makedirs(log_dir, exist_ok=True)
        
        self.log_file = os.path.join(log_dir, f"benchmark_run_{timestamp}.log")
        print(f"Logging to: {self.log_file}")
    
    def log_message(self, message: str):
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        print(log_entry)
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(log_entry + "\n")
    
    def run_benchmark(self, benchmark_name: str) -> Dict[str, Any]:
        
        self.log_message(f"Starting benchmark: {benchmark_name}")

        cmd = [
            sys.executable, "main.py",
            "--experiment", benchmark_name,
            "--timeout", str(self.config["timeout"])
        ]
        
        if self.config["use_bayesian"]:
            cmd.append("--use_bayesian")
        
        if self.config["use_passive_constraints"]:
            cmd.append("--use_passive_constraints")
            if self.config["passive_solutions"]:
                cmd.extend(["--passive_solutions", str(self.config["passive_solutions"])])
            cmd.extend(["--passive_output_dir", self.config["passive_output_dir"]])
        
        self.log_message(f"Command: {' '.join(cmd)}")

        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config["timeout"] + 60  
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            success = result.returncode == 0
            
            benchmark_result = {
                "benchmark": benchmark_name,
                "success": success,
                "execution_time": execution_time,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timestamp": datetime.now().isoformat()
            }
            
            if success:
                self.log_message(f"✅ {benchmark_name} completed successfully in {execution_time:.2f}s")
            else:
                self.log_message(f"❌ {benchmark_name} failed with return code {result.returncode}")
                self.log_message(f"Error output: {result.stderr}")
            
            return benchmark_result
            
        except subprocess.TimeoutExpired:
            self.log_message(f"⏰ {benchmark_name} timed out after {self.config['timeout']}s")
            return {
                "benchmark": benchmark_name,
                "success": False,
                "execution_time": self.config["timeout"],
                "return_code": -1,
                "stdout": "",
                "stderr": "Timeout expired",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.log_message(f"💥 {benchmark_name} failed with exception: {str(e)}")
            return {
                "benchmark": benchmark_name,
                "success": False,
                "execution_time": 0,
                "return_code": -2,
                "stdout": "",
                "stderr": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def run_all_benchmarks(self) -> List[Dict[str, Any]]:
        
        self.start_time = time.time()
        self.setup_logging()
        
        self.log_message("="*60)
        self.log_message("Starting benchmark runner")
        self.log_message(f"Benchmarks to run: {', '.join(self.benchmarks)}")
        self.log_message(f"Configuration: {json.dumps(self.config, indent=2)}")
        self.log_message("="*60)
        
        results = []
        for i, benchmark in enumerate(self.benchmarks, 1):
            self.log_message(f"Progress: {i}/{len(self.benchmarks)} benchmarks")
            result = self.run_benchmark(benchmark)
            results.append(result)

            if i < len(self.benchmarks):
                time.sleep(2)
        
        self.results = results
        return results
    
    def generate_summary_report(self) -> str:
        
        if not self.results:
            return "No benchmark results available"
        
        total_time = time.time() - self.start_time if self.start_time else 0
        successful = [r for r in self.results if r["success"]]
        failed = [r for r in self.results if not r["success"]]
        
        report = []
        report.append("="*60)
        report.append("BENCHMARK RUNNER SUMMARY REPORT")
        report.append("="*60)
        report.append(f"Total benchmarks run: {len(self.results)}")
        report.append(f"Successful: {len(successful)}")
        report.append(f"Failed: {len(failed)}")
        report.append(f"Total execution time: {total_time:.2f}s")
        report.append("")
        
        if successful:
            report.append("✅ SUCCESSFUL BENCHMARKS:")
            for result in successful:
                report.append(f"  - {result['benchmark']}: {result['execution_time']:.2f}s")
            report.append("")
        
        if failed:
            report.append("❌ FAILED BENCHMARKS:")
            for result in failed:
                reason = result['stderr'] if result['stderr'] else f"Return code: {result['return_code']}"
                report.append(f"  - {result['benchmark']}: {reason}")
            report.append("")

        report.append("DETAILED RESULTS:")
        for result in self.results:
            status = "✅" if result["success"] else "❌"
            report.append(f"{status} {result['benchmark']}")
            report.append(f"  Execution time: {result['execution_time']:.2f}s")
            report.append(f"  Return code: {result['return_code']}")
            if result['stderr']:
                report.append(f"  Error: {result['stderr'][:200]}...")
            report.append("")
        
        return "\n".join(report)
    
    def save_results(self, output_dir: str = "benchmark_results"):
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        json_file = os.path.join(output_dir, f"benchmark_results_{timestamp}.json")
        with open(json_file, "w") as f:
            json.dump({
                "config": self.config,
                "results": self.results,
                "summary": {
                    "total_benchmarks": len(self.results),
                    "successful": len([r for r in self.results if r["success"]]),
                    "failed": len([r for r in self.results if not r["success"]]),
                    "total_execution_time": time.time() - self.start_time if self.start_time else 0
                }
            }, f, indent=2)

        summary_file = os.path.join(output_dir, f"benchmark_summary_{timestamp}.txt")
        with open(summary_file, "w") as f:
            f.write(self.generate_summary_report())

        csv_file = os.path.join(output_dir, f"benchmark_summary_{timestamp}.csv")
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Benchmark", "Success", "Execution Time", "Return Code", "Error"])
            for result in self.results:
                writer.writerow([
                    result["benchmark"],
                    result["success"],
                    result["execution_time"],
                    result["return_code"],
                    result["stderr"][:100] if result["stderr"] else ""
                ])
        
        self.log_message(f"Results saved to:")
        self.log_message(f"  - JSON: {json_file}")
        self.log_message(f"  - Summary: {summary_file}")
        self.log_message(f"  - CSV: {csv_file}")


def main():
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Run all benchmarks systematically")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout per benchmark in seconds")
    parser.add_argument("--use_bayesian", action="store_true", default=True, help="Use Bayesian constraint acquisition")
    parser.add_argument("--use_passive_constraints", action="store_true", help="Use passive learning constraints")
    parser.add_argument("--passive_solutions", type=int, help="Number of solutions for passive learning")
    parser.add_argument("--passive_output_dir", type=str, default="output", help="Passive learning output directory")
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS, help="Specific benchmarks to run")
    parser.add_argument("--output_dir", type=str, default="benchmark_results", help="Output directory for results")
    
    args = parser.parse_args()

    benchmarks_to_run = args.benchmarks if args.benchmarks else BENCHMARKS

    config = {
        "timeout": args.timeout,
        "use_bayesian": args.use_bayesian,
        "use_passive_constraints": args.use_passive_constraints,
        "passive_solutions": args.passive_solutions,
        "passive_output_dir": args.passive_output_dir
    }

    runner = BenchmarkRunner(config)
    runner.benchmarks = benchmarks_to_run
    results = runner.run_all_benchmarks()

    summary = runner.generate_summary_report()
    print("\n" + summary)

    runner.save_results(args.output_dir)

    failed_count = len([r for r in results if not r["success"]])
    sys.exit(failed_count)


if __name__ == "__main__":
    main()