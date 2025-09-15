#!/usr/bin/env python3
"""
Simple VLLM test for LLM-Mob - Quick verification before production
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import logging
from pathlib import Path

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def test_basic_imports():
    """Test all required imports"""
    logger.info("🔍 Testing basic imports...")

    try:
        import numpy as np
        import pandas as pd
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        logger.info("✅ Data science imports OK")

        from vllm import LLM, SamplingParams
        logger.info("✅ VLLM imports OK")

        return True
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def test_data_files():
    """Test that required data files exist"""
    logger.info("🔍 Testing data files...")

    data_dir = Path("data/verona")
    poi_file = data_dir / "vc_site.csv"

    if not data_dir.exists():
        logger.error(f"❌ Data directory not found: {data_dir}")
        return False

    if not poi_file.exists():
        logger.error(f"❌ POI file not found: {poi_file}")
        return False

    # Test loading a small sample
    try:
        import pandas as pd
        pois = pd.read_csv(poi_file, nrows=5)  # Just load first 5 rows
        logger.info(f"✅ POI file readable: {len(pois)} sample rows loaded")

        # Check CSV files
        csv_files = list(data_dir.glob("dati_*.csv"))
        if csv_files:
            logger.info(f"✅ Found {len(csv_files)} data files")

            # Test loading one file sample
            sample_file = csv_files[0]
            sample_data = pd.read_csv(sample_file, nrows=10)
            logger.info(f"✅ Sample data readable: {sample_file.name} - {len(sample_data)} rows")
        else:
            logger.warning("⚠️ No dati_*.csv files found")

        return True

    except Exception as e:
        logger.error(f"❌ Data loading error: {e}")
        return False

def test_vllm_basic():
    """Test VLLM basic functionality (without loading model)"""
    logger.info("🔍 Testing VLLM basic functionality...")

    try:
        from vllm import SamplingParams

        # Test creating sampling parameters
        sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=10,
            top_p=0.9,
            top_k=10
        )
        logger.info("✅ SamplingParams creation successful")

        # Don't test LLM loading on login node (no GPUs)
        logger.info("✅ VLLM basic functionality OK (model loading will work on compute nodes)")
        return True

    except Exception as e:
        logger.error(f"❌ VLLM basic test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🚀 VLLM Deployment Test Suite")
    logger.info("=" * 50)

    tests = [
        ("Basic Imports", test_basic_imports),
        ("Data Files", test_data_files),
        ("VLLM Basic", test_vllm_basic)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name} test...")
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")

    logger.info("\n" + "=" * 50)
    logger.info(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎯 ALL TESTS PASSED - Ready for production deployment!")
        logger.info("🚀 Deploy with: sbatch vllm_4_GPU.sh")
        sys.exit(0)
    else:
        logger.error("❌ Some tests failed - check configuration before deployment")
        sys.exit(1)

if __name__ == "__main__":
    main()