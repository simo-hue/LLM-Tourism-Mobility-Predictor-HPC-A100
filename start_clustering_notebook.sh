#!/bin/bash

# Script di avvio rapido per il notebook di analisi clustering
# Usage: ./start_clustering_notebook.sh

set -e  # Exit on error

echo "========================================="
echo "🚀 AVVIO CLUSTERING ANALYSIS NOTEBOOK"
echo "========================================="

# Check if virtual environment exists
if [ ! -d "llm" ]; then
    echo "❌ Errore: ambiente virtuale 'llm' non trovato"
    echo "💡 Crea l'ambiente con: python3 -m venv llm"
    exit 1
fi

# Activate virtual environment
echo "📦 Attivazione ambiente virtuale..."
source llm/bin/activate

# Check if required packages are installed
echo "🔍 Verifica dipendenze..."
python -c "import pandas, numpy, sklearn, matplotlib, seaborn" 2>/dev/null || {
    echo "❌ Dipendenze mancanti!"
    echo "💡 Installa con: pip install -r requirements_clustering.txt"
    exit 1
}

# Check if notebook exists
if [ ! -f "notebook/clustering_analysis.ipynb" ]; then
    echo "❌ Errore: notebook non trovato"
    exit 1
fi

echo "✅ Ambiente configurato correttamente"
echo ""
echo "🌐 Avvio JupyterLab..."
echo "   Il browser si aprirà automaticamente tra pochi secondi..."
echo "   URL: http://localhost:8888"
echo ""
echo "📝 Ricorda di selezionare il kernel 'Python (llm)' nel notebook"
echo ""
echo "⚠️  Premi CTRL+C per fermare il server Jupyter"
echo ""

# Start JupyterLab
cd notebook
jupyter lab clustering_analysis.ipynb --no-browser 2>/dev/null || jupyter lab clustering_analysis.ipynb
