#!/usr/bin/env python3
"""
Script di test per verificare l'installazione dell'ambiente clustering.
Esegui: source llm/bin/activate && python test_clustering_env.py
"""

import sys
from pathlib import Path

def test_imports():
    """Test import di tutte le librerie necessarie"""
    print("🔍 Test import librerie...")

    try:
        import pandas as pd
        print(f"  ✅ pandas {pd.__version__}")
    except ImportError as e:
        print(f"  ❌ pandas: {e}")
        return False

    try:
        import numpy as np
        print(f"  ✅ numpy {np.__version__}")
    except ImportError as e:
        print(f"  ❌ numpy: {e}")
        return False

    try:
        import matplotlib
        import matplotlib.pyplot as plt
        print(f"  ✅ matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"  ❌ matplotlib: {e}")
        return False

    try:
        import seaborn as sns
        print(f"  ✅ seaborn {sns.__version__}")
    except ImportError as e:
        print(f"  ❌ seaborn: {e}")
        return False

    try:
        import sklearn
        print(f"  ✅ scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"  ❌ scikit-learn: {e}")
        return False

    try:
        import scipy
        print(f"  ✅ scipy {scipy.__version__}")
    except ImportError as e:
        print(f"  ❌ scipy: {e}")
        return False

    try:
        import jupyter
        print(f"  ✅ jupyter")
    except ImportError as e:
        print(f"  ❌ jupyter: {e}")
        return False

    try:
        import jupyterlab
        print(f"  ✅ jupyterlab")
    except ImportError as e:
        print(f"  ❌ jupyterlab: {e}")
        return False

    try:
        import tqdm
        print(f"  ✅ tqdm {tqdm.__version__}")
    except ImportError as e:
        print(f"  ❌ tqdm: {e}")
        return False

    return True

def test_sklearn_components():
    """Test componenti specifici di scikit-learn"""
    print("\n🔍 Test componenti scikit-learn...")

    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.metrics import silhouette_score, davies_bouldin_score
        print("  ✅ Tutti i componenti sklearn importati correttamente")
        return True
    except ImportError as e:
        print(f"  ❌ Errore import sklearn: {e}")
        return False

def test_data_files():
    """Test esistenza file di dati"""
    print("\n🔍 Test file di dati...")

    data_dir = Path("data/verona")

    if not data_dir.exists():
        print(f"  ❌ Directory {data_dir} non trovata")
        return False

    poi_file = data_dir / "vc_site.csv"
    if not poi_file.exists():
        print(f"  ❌ File POI {poi_file} non trovato")
        return False
    print(f"  ✅ File POI trovato: {poi_file}")

    # Test file visite
    visit_files = list(data_dir.glob("dati_*.csv"))
    if not visit_files:
        print(f"  ⚠️  Nessun file visite trovato in {data_dir}")
        return True  # Non bloccante

    print(f"  ✅ Trovati {len(visit_files)} file visite:")
    for vf in sorted(visit_files):
        print(f"      - {vf.name}")

    return True

def test_notebook_exists():
    """Test esistenza notebook"""
    print("\n🔍 Test notebook...")

    notebook_file = Path("notebook/clustering_analysis.ipynb")
    if not notebook_file.exists():
        print(f"  ❌ Notebook {notebook_file} non trovato")
        return False

    print(f"  ✅ Notebook trovato: {notebook_file}")
    return True

def test_kernel_installation():
    """Test installazione kernel Jupyter"""
    print("\n🔍 Test kernel Jupyter...")

    import subprocess

    try:
        result = subprocess.run(
            ["jupyter", "kernelspec", "list"],
            capture_output=True,
            text=True,
            check=True
        )

        if "llm" in result.stdout:
            print("  ✅ Kernel 'llm' installato correttamente")
            return True
        else:
            print("  ⚠️  Kernel 'llm' non trovato")
            print("  💡 Esegui: python -m ipykernel install --user --name=llm --display-name='Python (llm)'")
            return False
    except Exception as e:
        print(f"  ❌ Errore verifica kernel: {e}")
        return False

def main():
    """Esegue tutti i test"""
    print("="*70)
    print("🚀 TEST AMBIENTE CLUSTERING ANALYSIS")
    print("="*70)

    results = {
        "Import librerie": test_imports(),
        "Componenti sklearn": test_sklearn_components(),
        "File di dati": test_data_files(),
        "Notebook": test_notebook_exists(),
        "Kernel Jupyter": test_kernel_installation()
    }

    print("\n" + "="*70)
    print("📊 RISULTATI TEST")
    print("="*70)

    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:.<40} {status}")
        if not passed:
            all_passed = False

    print("="*70)

    if all_passed:
        print("\n🎉 Tutti i test sono passati!")
        print("\n📝 Prossimi passi:")
        print("   1. Attiva l'ambiente: source llm/bin/activate")
        print("   2. Avvia Jupyter: cd notebook && jupyter lab clustering_analysis.ipynb")
        print("   3. Seleziona kernel 'Python (llm)' nel notebook")
        return 0
    else:
        print("\n⚠️  Alcuni test sono falliti. Controlla gli errori sopra.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
