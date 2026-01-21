#!/usr/bin/env python3
"""
Test runner for the Plant Diagnostic System.
Runs all tests and provides a comprehensive report.
"""

import unittest
import sys
import os
import time
from io import StringIO

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_utils import check_required_files, check_gpu_availability


def run_all_tests():
    """Run all tests and return results."""
    # Discover and run all tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(os.path.abspath(__file__))
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Create test runner
    stream = StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2)
    
    # Run tests
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # Get output
    output = stream.getvalue()
    
    return {
        'result': result,
        'output': output,
        'duration': end_time - start_time,
        'tests_run': result.testsRun,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'skipped': len(result.skipped)
    }


def print_test_report(test_results):
    """Print a comprehensive test report."""
    print("=" * 80)
    print("PLANT DIAGNOSTIC SYSTEM - TEST REPORT")
    print("=" * 80)
    
    # System information
    print("\nSYSTEM INFORMATION:")
    print(f"  Python version: {sys.version}")
    print(f"  GPU available: {check_gpu_availability()}")
    
    # Check required files
    missing_files = check_required_files()
    if missing_files:
        print(f"  Missing files: {', '.join(missing_files)}")
    else:
        print("  All required files present")
    
    # Test results
    print(f"\nTEST RESULTS:")
    print(f"  Tests run: {test_results['tests_run']}")
    print(f"  Failures: {test_results['failures']}")
    print(f"  Errors: {test_results['errors']}")
    print(f"  Skipped: {test_results['skipped']}")
    print(f"  Duration: {test_results['duration']:.2f} seconds")
    
    # Success rate
    if test_results['tests_run'] > 0:
        success_rate = ((test_results['tests_run'] - test_results['failures'] - test_results['errors']) / test_results['tests_run']) * 100
        print(f"  Success rate: {success_rate:.1f}%")
    
    # Detailed output
    if test_results['output']:
        print(f"\nDETAILED OUTPUT:")
        print("-" * 80)
        print(test_results['output'])
    
    # Summary
    print("\n" + "=" * 80)
    if test_results['failures'] == 0 and test_results['errors'] == 0:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")
        if test_results['failures'] > 0:
            print(f"   {test_results['failures']} test(s) failed")
        if test_results['errors'] > 0:
            print(f"   {test_results['errors']} test(s) had errors")
    print("=" * 80)


def main():
    """Main test runner function."""
    print("Running Plant Diagnostic System tests...")
    
    # Run tests
    test_results = run_all_tests()
    
    # Print report
    print_test_report(test_results)
    
    # Return appropriate exit code
    if test_results['failures'] > 0 or test_results['errors'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()


