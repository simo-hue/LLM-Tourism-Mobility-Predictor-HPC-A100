#!/bin/bash
#SBATCH --job-name=vllm_prod
#SBATCH --account=IscrC_LLM-Mob
#SBATCH --partition=boost_usr_prod
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --output=mobility-vllm_prod-%j.out

echo "🚀 VERONA CARD - VLLM PRODUCTION"
echo "================================================"
echo "🚀 VLLM-ULTRA MODE: 4x A100 tensor parallelism, Mistral-7B, 50-100x faster than Ollama"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodo: $(hostname)"
echo "Data: $(date)"
echo ""

# ============= SETUP AMBIENTE =============
echo "📦 Setup ambiente HPC..."
module purge
module load python/3.11.6--gcc--8.5.0
module load cuda/12.3
source $WORK/venv/bin/activate

echo "✅ Python: $(python3 --version)"
echo "✅ CUDA: $(nvcc --version | grep release)"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NVIDIA_VISIBLE_DEVICES=0,1,2,3

# Debug GPU iniziale
echo ""
echo "🔍 GPU DETECTION:"
nvidia-smi --query-gpu=index,name,memory.total,temperature.gpu --format=csv,noheader
echo ""

# ============= VLLM ENVIRONMENT SETUP =============
# Ottimizzazioni per VLLM con 4x A100 64GB
export VLLM_TENSOR_PARALLEL_SIZE=4
export VLLM_GPU_MEMORY_UTILIZATION=0.85
export VLLM_MAX_MODEL_LEN=1024
export VLLM_BATCH_SIZE=256
export VLLM_WORKER_USE_RAY=0
export VLLM_USE_MODELSCOPE=0
export VLLM_ATTENTION_BACKEND=FLASHINFER

# Ottimizzazioni CUDA per tensor parallelism
export CUDA_LAUNCH_BLOCKING=0
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo "🔧 VLLM Configuration:"
echo "  Tensor Parallel Size: $VLLM_TENSOR_PARALLEL_SIZE"
echo "  GPU Memory Utilization: $VLLM_GPU_MEMORY_UTILIZATION"
echo "  Batch Size: $VLLM_BATCH_SIZE"
echo ""

# ============= VERIFICA PREREQUISITI =============
echo "📋 Verifica prerequisiti VLLM..."

# Verifica che VLLM sia installato
if ! python3 -c "import vllm; print(f'VLLM version: {vllm.__version__}')" 2>/dev/null; then
    echo "❌ ERRORE: VLLM non installato!"
    echo "💡 Installa con: pip install vllm"
    exit 1
else
    echo "✅ VLLM disponibile"
fi

# Verifica GPU
if [ $(nvidia-smi --query-gpu=count --format=csv,noheader) -lt 4 ]; then
    echo "❌ ERRORE: Servono almeno 4 GPU per tensor parallelism"
    exit 1
else
    echo "✅ 4x GPU disponibili per tensor parallelism"
fi

# Verifica memoria GPU
MIN_GPU_MEMORY=60000  # 60GB minimum per A100
for gpu_id in 0 1 2 3; do
    gpu_memory=$(nvidia-smi --id=$gpu_id --query-gpu=memory.total --format=csv,noheader,nounits)
    if [ $gpu_memory -lt $MIN_GPU_MEMORY ]; then
        echo "❌ ERRORE: GPU $gpu_id ha solo ${gpu_memory}MB (minimo: ${MIN_GPU_MEMORY}MB)"
        exit 1
    fi
done
echo "✅ Tutte le GPU hanno memoria sufficiente"

# Verifica file Python VLLM
if [ ! -f "veronacard_mob_vllm.py" ]; then
    echo "❌ ERRORE: File veronacard_mob_vllm.py non trovato!"
    exit 1
else
    echo "✅ Script VLLM trovato"
fi

echo ""

# ============= SETUP DIRECTORY TEMPORANEA =============
CUSTOM_TMP="$WORK/tmp_vllm_$SLURM_JOB_ID"
mkdir -p "$CUSTOM_TMP"
chmod 700 "$CUSTOM_TMP"

# Export variabili temporanee
export TMPDIR="$CUSTOM_TMP"
export TMP="$CUSTOM_TMP"
export TEMP="$CUSTOM_TMP"

echo "📁 Directory temporanea: $CUSTOM_TMP"
WORK_AVAILABLE=$(df "$WORK" | tail -1 | awk '{print $4}')
WORK_AVAILABLE_GB=$((WORK_AVAILABLE / 1024 / 1024))
echo "💾 Spazio disponibile: ${WORK_AVAILABLE_GB}GB"

if [ $WORK_AVAILABLE_GB -lt 50 ]; then
    echo "❌ ERRORE: Spazio insufficiente (${WORK_AVAILABLE_GB}GB < 50GB)"
    exit 1
fi

# ============= FUNZIONE DI CLEANUP =============
cleanup() {
    echo ""
    echo "🧹 Cleanup finale..."

    # Kill eventuali processi VLLM
    pkill -f vllm 2>/dev/null || true
    pkill -f "veronacard_mob_vllm.py" 2>/dev/null || true

    # Rimuovi directory temporanea
    if [ -n "$CUSTOM_TMP" ] && [ -d "$CUSTOM_TMP" ]; then
        echo "Removing $CUSTOM_TMP..."
        rm -rf "$CUSTOM_TMP"
    fi

    echo "✅ Cleanup completato"
}
trap cleanup EXIT

# ============= MONITORING GPU AVANZATO =============
advanced_gpu_monitor() {
    echo "📊 Starting Advanced GPU Monitor per VLLM (ogni 60s)"

    while true; do
        sleep 180

        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "📊 VLLM GPU STATUS - $(date '+%Y-%m-%d %H:%M:%S')"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        # Mostra utilizzo GPU dettagliato per tensor parallelism
        nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits | \
        while IFS=',' read -r idx name util_gpu util_mem mem_used mem_total temp power; do
            # Calcola percentuale memoria
            mem_percent=$(echo "scale=1; $mem_used * 100 / $mem_total" | bc -l 2>/dev/null || echo "0")

            # Status per VLLM tensor parallelism
            if [ "$util_gpu" -gt 90 ]; then
                status="🔥 VLLM-MAX"
            elif [ "$util_gpu" -gt 70 ]; then
                status="🚀 VLLM-HIGH"
            elif [ "$util_gpu" -gt 30 ]; then
                status="⚡ VLLM-WORK"
            elif [ "$util_gpu" -gt 5 ]; then
                status="💤 VLLM-IDLE"
            else
                status="❄️ UNUSED"
            fi

            printf "GPU %s: %s\n" "$idx" "$status"
            printf "  Compute: %3d%% | Memory: %3d%% (%s/%s MB) | VRAM: %s%%\n" \
                   "$util_gpu" "$util_mem" "$mem_used" "$mem_total" "$mem_percent"
            printf "  Temp: %d°C | Power: %s W\n" "$temp" "$power"
            echo ""
        done

        # Mostra processi VLLM
        echo "🔄 VLLM Processes:"
        if pgrep -f "veronacard_mob_vllm.py" >/dev/null; then
            vllm_pids=$(pgrep -f "veronacard_mob_vllm.py")
            for pid in $vllm_pids; do
                cpu_usage=$(ps -p $pid -o %cpu= 2>/dev/null | tr -d ' ' || echo "0")
                mem_usage=$(ps -p $pid -o rss= 2>/dev/null | awk '{printf "%.1f", $1/1024/1024}' || echo "0")

                echo "  VLLM Main (PID $pid): ✅ CPU: ${cpu_usage}% | RAM: ${mem_usage}GB"
            done
        else
            echo "  VLLM: Not running or completed"
        fi

        # Statistiche Python processing
        if [ -f "vllm_production_execution.log" ]; then
            echo ""
            echo "🐍 VLLM Processing Stats:"

            processed=$(grep -c "BATCH_COMPLETE" vllm_production_execution.log 2>/dev/null || echo "0")
            throughput=$(grep "cards/second" vllm_production_execution.log 2>/dev/null | tail -1 | grep -o '[0-9.]\+' | head -1 || echo "0")
            errors=$(grep -c "ERROR\|Error" vllm_production_execution.log 2>/dev/null || echo "0")

            echo "  Batches completed: $processed"
            echo "  Throughput: ${throughput} cards/sec"
            echo "  Errors: $errors"
        fi

        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    done
}

# Avvia monitor avanzato in background
advanced_gpu_monitor &
MONITOR_PID=$!

# ============= PREPARAZIONE GPU PER VLLM =============
echo ""
echo "🚀 PREPARAZIONE 4x A100 PER VLLM TENSOR PARALLELISM"
echo "===================================================="

# Verifica che tutte le GPU siano libere
echo "🔍 Verifica GPU disponibilità..."
for gpu_id in 0 1 2 3; do
    gpu_processes=$(nvidia-smi --id=$gpu_id --query-compute-apps=pid --format=csv,noheader,nounits | wc -l)
    if [ $gpu_processes -gt 0 ]; then
        echo "⚠️ GPU $gpu_id ha processi attivi - li termino"
        nvidia-smi --id=$gpu_id --query-compute-apps=pid --format=csv,noheader,nounits | xargs -r kill -9 2>/dev/null || true
        sleep 2
    fi
done

echo "✅ Tutte le GPU sono pronte per VLLM"

echo "✅ GPU setup completato (warming saltato per stabilità)"

# ============= ESECUZIONE VLLM =============
cd /leonardo_work/IscrC_LLM-Mob/LLM-Mob-As-Mobility-Interpreter

echo ""
echo "🚀 AVVIO VLLM PROCESSING"
echo "========================"
echo "🔧 Configurazione: 4x A100 Tensor Parallelism, Batch Size 256"
echo "📊 Modello: Qwen2.5-7B-Instruct"
echo "🎯 Target: 50-100x performance vs Ollama"
echo ""

if [ -f "data/verona/vc_site.csv" ]; then
    echo "🚀 Avvio VLLM con configurazione ottimizzata..."

    # Pre-flight check veloce
    echo "🔍 Pre-flight check VLLM..."
    if ! python3 -c "
import vllm
from vllm import LLM, SamplingParams
print('✅ VLLM import successful')
print('🔧 Checking GPU availability...')
import torch
print(f'✅ CUDA GPUs: {torch.cuda.device_count()}')
" 2>&1; then
        echo "❌ VLLM pre-flight check failed!"
        exit 1
    fi

    echo "✅ Pre-flight check passed - starting VLLM processing"
    echo ""

    # Esegui VLLM con logging completo
    python3 -u veronacard_mob_vllm.py \
        --append 2>&1 | tee vllm_production_execution.log
    PYTHON_EXIT=$?

else
    echo "❌ File data/verona/vc_site.csv non trovato!"
    PYTHON_EXIT=1
fi

# Stop monitor
kill $MONITOR_PID 2>/dev/null || true

# ============= REPORT FINALE =============
echo ""
echo "============================================"
echo "🎯 VLLM JOB COMPLETATO"
echo "============================================"
echo "Exit code Python: $PYTHON_EXIT"
echo "GPU utilizzate: 4x A100 (Tensor Parallelism)"
echo "Tempo totale: $SECONDS secondi"

if [ -f "vllm_production_execution.log" ]; then
    echo ""
    echo "📊 STATISTICHE FINALI:"

    # Calcola throughput medio
    total_batches=$(grep -c "BATCH_COMPLETE" vllm_production_execution.log 2>/dev/null || echo "0")
    avg_throughput=$(grep "cards/second" vllm_production_execution.log 2>/dev/null | grep -o '[0-9.]\+' | awk '{sum+=$1; count++} END {if(count>0) printf "%.1f", sum/count; else print "0"}')
    total_errors=$(grep -c "ERROR\|Error" vllm_production_execution.log 2>/dev/null || echo "0")

    echo "  Total batches: $total_batches"
    echo "  Average throughput: ${avg_throughput} cards/sec"
    echo "  Total errors: $total_errors"

    # Performance comparison estimate
    if [ "$avg_throughput" != "0" ] && [ $(echo "$avg_throughput > 0" | bc -l 2>/dev/null || echo "0") -eq 1 ]; then
        # Stima performance vs Ollama (assumendo ~0.5 cards/sec per Ollama)
        performance_multiplier=$(echo "scale=1; $avg_throughput / 0.5" | bc -l 2>/dev/null || echo "N/A")
        echo "  🚀 Performance vs Ollama: ~${performance_multiplier}x faster"
    fi
fi

echo "============================================"

exit $PYTHON_EXIT