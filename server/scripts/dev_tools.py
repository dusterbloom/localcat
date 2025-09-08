#!/usr/bin/env python3
"""
Test runner for LocalCat development
"""

import asyncio
import sys
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any

# Add server directory to path
server_dir = Path(__file__).parent / "server"
sys.path.insert(0, str(server_dir))

from config import get_config, EnvironmentType
from dev_tools import DevToolsServer, DevToolConfig, create_dev_tools_server
from tests.conftest import benchmark_function, measure_memory_usage


class TestRunner:
    """Test runner for LocalCat development"""
    
    def __init__(self):
        self.config = get_config()
        self.test_results: List[Dict[str, Any]] = []
    
    def run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests"""
        print("🧪 Running unit tests...")
        
        try:
            # Run pytest for unit tests
            result = subprocess.run(
                ["python", "-m", "pytest", "server/tests/unit/", "-v", "--tb=short"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent
            )
            
            return {
                "type": "unit_tests",
                "success": result.returncode == 0,
                "output": result.stdout,
                "errors": result.stderr,
                "return_code": result.returncode
            }
        except Exception as e:
            return {
                "type": "unit_tests",
                "success": False,
                "error": str(e)
            }
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests"""
        print("🔗 Running integration tests...")
        
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "server/tests/integration/", "-v", "--tb=short"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent
            )
            
            return {
                "type": "integration_tests",
                "success": result.returncode == 0,
                "output": result.stdout,
                "errors": result.stderr,
                "return_code": result.returncode
            }
        except Exception as e:
            return {
                "type": "integration_tests",
                "success": False,
                "error": str(e)
            }
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests"""
        print("⚡ Running performance tests...")
        
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "server/tests/performance/", "-v", "--tb=short", "-m", "performance"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent
            )
            
            return {
                "type": "performance_tests",
                "success": result.returncode == 0,
                "output": result.stdout,
                "errors": result.stderr,
                "return_code": result.returncode
            }
        except Exception as e:
            return {
                "type": "performance_tests",
                "success": False,
                "error": str(e)
            }
    
    def run_linting(self) -> Dict[str, Any]:
        """Run code linting"""
        print("🔍 Running linting...")
        
        try:
            # Run flake8 if available
            result = subprocess.run(
                ["python", "-m", "flake8", "server/", "--max-line-length=100", "--ignore=E203,W503"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent
            )
            
            return {
                "type": "linting",
                "success": result.returncode == 0,
                "output": result.stdout,
                "errors": result.stderr,
                "return_code": result.returncode
            }
        except FileNotFoundError:
            return {
                "type": "linting",
                "success": True,
                "output": "Flake8 not installed, skipping linting",
                "errors": "",
                "return_code": 0
            }
        except Exception as e:
            return {
                "type": "linting",
                "success": False,
                "error": str(e)
            }
    
    def run_type_checking(self) -> Dict[str, Any]:
        """Run type checking"""
        print("🔬 Running type checking...")
        
        try:
            # Run mypy if available
            result = subprocess.run(
                ["python", "-m", "mypy", "server/", "--ignore-missing-imports"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent
            )
            
            return {
                "type": "type_checking",
                "success": result.returncode == 0,
                "output": result.stdout,
                "errors": result.stderr,
                "return_code": result.returncode
            }
        except FileNotFoundError:
            return {
                "type": "type_checking",
                "success": True,
                "output": "MyPy not installed, skipping type checking",
                "errors": "",
                "return_code": 0
            }
        except Exception as e:
            return {
                "type": "type_checking",
                "success": False,
                "error": str(e)
            }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and checks"""
        print("🚀 Running all tests and checks...")
        
        tests = [
            self.run_unit_tests,
            self.run_integration_tests,
            self.run_performance_tests,
            self.run_linting,
            self.run_type_checking
        ]
        
        results = []
        for test_func in tests:
            result = test_func()
            results.append(result)
        
        # Generate summary
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.get("success", False))
        
        summary = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
            "results": results,
            "timestamp": time.time()
        }
        
        return summary
    
    def print_results(self, results: Dict[str, Any]):
        """Print test results"""
        print("\n" + "="*60)
        print("📊 TEST RESULTS SUMMARY")
        print("="*60)
        
        print(f"Total Tests: {results['total_tests']}")
        print(f"Passed: {results['passed_tests']}")
        print(f"Failed: {results['failed_tests']}")
        print(f"Success Rate: {results['success_rate']:.1f}%")
        
        print("\n" + "-"*40)
        for result in results["results"]:
            status = "✅ PASS" if result.get("success", False) else "❌ FAIL"
            print(f"{status} {result['type'].replace('_', ' ').title()}")
            
            if result.get("error"):
                print(f"  Error: {result['error']}")
            
            if result.get("output"):
                # Show first few lines of output
                output_lines = result["output"].split('\n')[:5]
                for line in output_lines:
                    if line.strip():
                        print(f"  {line}")
                if len(result["output"].split('\n')) > 5:
                    print("  ...")
        
        print("="*60)
        
        if results["success_rate"] == 100:
            print("🎉 All tests passed!")
        elif results["success_rate"] >= 80:
            print("⚠️  Most tests passed, but some issues need attention")
        else:
            print("🚨 Many tests failed - please review the issues")


class DevServerManager:
    """Manager for development server"""
    
    def __init__(self):
        self.dev_server = None
        self.server_process = None
    
    async def start_dev_server(self, port: int = 8080):
        """Start development server"""
        print(f"🚀 Starting development server on port {port}...")
        
        config = DevToolConfig(
            enable_debug_server=True,
            debug_server_port=port,
            enable_hot_reload=True,
            enable_real_time_metrics=True
        )
        
        self.dev_server = create_dev_tools_server(config)
        
        try:
            await self.dev_server.start_server()
        except KeyboardInterrupt:
            print("\n🛑 Development server stopped")
    
    def start_bot_server(self):
        """Start bot server"""
        print("🤖 Starting bot server...")
        
        try:
            self.server_process = subprocess.Popen(
                ["python", "server/bot.py"],
                cwd=Path(__file__).parent
            )
            print(f"Bot server started with PID: {self.server_process.pid}")
        except Exception as e:
            print(f"Error starting bot server: {e}")
    
    def stop_bot_server(self):
        """Stop bot server"""
        if self.server_process:
            print("🛑 Stopping bot server...")
            self.server_process.terminate()
            self.server_process.wait()
            print("Bot server stopped")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LocalCat Development Tools")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--performance", action="store_true", help="Run performance tests only")
    parser.add_argument("--lint", action="store_true", help="Run linting only")
    parser.add_argument("--type-check", action="store_true", help="Run type checking only")
    parser.add_argument("--dev-server", action="store_true", help="Start development server")
    parser.add_argument("--bot-server", action="store_true", help="Start bot server")
    parser.add_argument("--port", type=int, default=8080, help="Development server port")
    
    args = parser.parse_args()
    
    if args.test:
        # Run all tests
        runner = TestRunner()
        results = runner.run_all_tests()
        runner.print_results(results)
        sys.exit(0 if results["success_rate"] == 100 else 1)
    
    elif args.unit:
        # Run unit tests only
        runner = TestRunner()
        results = runner.run_unit_tests()
        runner.print_results({"results": [results], "total_tests": 1, "passed_tests": 1 if results["success"] else 0, "failed_tests": 0 if results["success"] else 1, "success_rate": 100 if results["success"] else 0})
        sys.exit(0 if results["success"] else 1)
    
    elif args.integration:
        # Run integration tests only
        runner = TestRunner()
        results = runner.run_integration_tests()
        runner.print_results({"results": [results], "total_tests": 1, "passed_tests": 1 if results["success"] else 0, "failed_tests": 0 if results["success"] else 1, "success_rate": 100 if results["success"] else 0})
        sys.exit(0 if results["success"] else 1)
    
    elif args.performance:
        # Run performance tests only
        runner = TestRunner()
        results = runner.run_performance_tests()
        runner.print_results({"results": [results], "total_tests": 1, "passed_tests": 1 if results["success"] else 0, "failed_tests": 0 if results["success"] else 1, "success_rate": 100 if results["success"] else 0})
        sys.exit(0 if results["success"] else 1)
    
    elif args.lint:
        # Run linting only
        runner = TestRunner()
        results = runner.run_linting()
        runner.print_results({"results": [results], "total_tests": 1, "passed_tests": 1 if results["success"] else 0, "failed_tests": 0 if results["success"] else 1, "success_rate": 100 if results["success"] else 0})
        sys.exit(0 if results["success"] else 1)
    
    elif args.type_check:
        # Run type checking only
        runner = TestRunner()
        results = runner.run_type_checking()
        runner.print_results({"results": [results], "total_tests": 1, "passed_tests": 1 if results["success"] else 0, "failed_tests": 0 if results["success"] else 1, "success_rate": 100 if results["success"] else 0})
        sys.exit(0 if results["success"] else 1)
    
    elif args.dev_server:
        # Start development server
        manager = DevServerManager()
        asyncio.run(manager.start_dev_server(args.port))
    
    elif args.bot_server:
        # Start bot server
        manager = DevServerManager()
        try:
            manager.start_bot_server()
            print("Press Ctrl+C to stop the server...")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            manager.stop_bot_server()
    
    else:
        # Default: show help
        parser.print_help()
        print("\n🔧 LocalCat Development Tools")
        print("\nCommon usage:")
        print("  python scripts/dev_tools.py --test          # Run all tests")
        print("  python scripts/dev_tools.py --dev-server   # Start dev server")
        print("  python scripts/dev_tools.py --bot-server   # Start bot server")
        print("  python scripts/dev_tools.py --unit          # Run unit tests")
        print("  python scripts/dev_tools.py --integration  # Run integration tests")


if __name__ == "__main__":
    main()