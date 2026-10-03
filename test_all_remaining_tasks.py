"""
Master Test Runner: All Remaining Tasks (Tasks 3, 4, 5, and 6)
=============================================================
Owner: Vedant (Member A)
Feature: Overlapping-Speech Attribution Suite

Executes the verification suite across all remaining modules in one command:
    python voice_intelligence/test_all_remaining_tasks.py
"""

import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from voice_intelligence.test_task3_overlap import run_task3_verification
    from voice_intelligence.test_task4_fingerprint import run_task4_verification
    from voice_intelligence.test_task5_attribution import run_task5_verification
    from voice_intelligence.test_task6_benchmarks import run_task6_verification
    from voice_intelligence.test_voice_registration_and_recognition import run_test as run_registration_test
except ImportError:
    from test_task3_overlap import run_task3_verification
    from test_task4_fingerprint import run_task4_verification
    from test_task5_attribution import run_task5_verification
    from test_task6_benchmarks import run_task6_verification
    from test_voice_registration_and_recognition import run_test as run_registration_test


def main():
    t_start = time.time()
    print("\n" + "#" * 72)
    print("#  OVERLAPPING-SPEECH ATTRIBUTION ENGINE: FULL REMAINING SUITE RUNNER  #")
    print("#  Owner: Vedant (Member A) | Tasks 3, 4, 5, 6 & Voice Registration    #")
    print("#" * 72)

    results = {}

    # Task 3
    try:
        results["Task 3: Overlap Detection"] = run_task3_verification()
    except Exception as e:
        print(f"❌ Task 3 Failed: {e}")
        results["Task 3: Overlap Detection"] = False

    # Task 4
    try:
        results["Task 4: Voice Fingerprinting"] = run_task4_verification()
    except Exception as e:
        print(f"❌ Task 4 Failed: {e}")
        results["Task 4: Voice Fingerprinting"] = False

    # Task 5
    try:
        results["Task 5: Attribution Engine"] = run_task5_verification()
    except Exception as e:
        print(f"❌ Task 5 Failed: {e}")
        results["Task 5: Attribution Engine"] = False

    # Task 6
    try:
        results["Task 6: Patent Benchmarks"] = run_task6_verification()
    except Exception as e:
        print(f"❌ Task 6 Failed: {e}")
        results["Task 6: Patent Benchmarks"] = False

    # Voice Registration & Recognition from Saved Data
    try:
        results["Voice Registration & Recognition"] = run_registration_test()
    except Exception as e:
        print(f"❌ Voice Registration & Recognition Failed: {e}")
        results["Voice Registration & Recognition"] = False

    elapsed = time.time() - t_start

    print("\n" + "=" * 72)
    print("📊 REMAINING TASKS VERIFICATION SUMMARY")
    print("=" * 72)
    all_passed = True
    for task_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  • {task_name:<36}: {status}")
        if not passed:
            all_passed = False

    print("-" * 72)
    print(f"  Total Execution Time: {elapsed:.2f}s")
    if all_passed:
        print("  🎉 ALL REMAINING TASKS (3, 4, 5, 6) PASSED WITH ZERO ERRORS!")
    else:
        print("  ⚠️ Some tasks encountered issues.")
    print("=" * 72 + "\n")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
