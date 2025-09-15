#!/bin/bash
#SBATCH --job-name=time_prod
#SBATCH --account=IscrC_LLM-Mob
#SBATCH --partition=boost_usr_prod
#SBATCH --time=00:20:00 
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --output=mobility-qwen_time_prod-%j.out

echo "🚀 VERONA CARD - TIME PRODUCTION"
echo "================================================"
echo "🚀 MISTRAL-MAXIMUM GPU MODE: 1 GPU sequenziale, mistral:7b, 95% VRAM, ALL layers GPU, ottimizzato per velocità"
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

export CUDA_VISIBLE_DEVICES=0
export NVIDIA_VISIBLE_DEVICES=0

# Debug GPU iniziale
echo ""
echo "🔍 GPU DETECTION:"
nvidia-smi --query-gpu=index,name,memory.total,temperature.gpu --format=csv,noheader
echo ""

# ============= SETUP DIRECTORY TEMPORANEA =============
CUSTOM_TMP="$WORK/tmp_ollama_$SLURM_JOB_ID"
mkdir -p "$CUSTOM_TMP"
chmod 700 "$CUSTOM_TMP"

# Export variabili temporanee
export TMPDIR="$CUSTOM_TMP"
export TMP="$CUSTOM_TMP"
export TEMP="$CUSTOM_TMP"
export OLLAMA_TMPDIR="$CUSTOM_TMP"

echo "📁 Directory temporanea: $CUSTOM_TMP"
WORK_AVAILABLE=$(df "$WORK" | tail -1 | awk '{print $4}')
WORK_AVAILABLE_GB=$((WORK_AVAILABLE / 1024 / 1024))
echo "💾 Spazio disponibile: ${WORK_AVAILABLE_GB}GB"

if [ $WORK_AVAILABLE_GB -lt 30 ]; then
    echo "❌ ERRORE: Spazio insufficiente (${WORK_AVAILABLE_GB}GB < 30GB)"
    exit 1
fi

# ============= CONFIGURAZIONE OLLAMA =============
OLLAMA_BIN="/leonardo_work/IscrC_LLM-Mob/opt/bin/ollama"

if [ ! -f "$OLLAMA_BIN" ]; then
    echo "❌ ERRORE: Ollama non trovato in $OLLAMA_BIN"
    exit 1
fi

# Ottimizzazioni per A100 64GB
export OLLAMA_DEBUG=1
export OLLAMA_MODELS="$WORK/.ollama/models"
export OLLAMA_CACHE_DIR="$WORK/.ollama/cache"
export OLLAMA_NUM_PARALLEL=1              # 🔧 CONSERVATIVE: Un solo processo parallelo
export OLLAMA_MAX_LOADED_MODELS=1          # 🔧 CONSERVATIVE: Un solo modello caricato
export OLLAMA_KEEP_ALIVE="8h"              # 🔧 EXTENDED: Mantieni modello in memoria
export OLLAMA_MAX_QUEUE=10                 # 🔧 OPTIMIZED: Coda più grande per maggiore throughput
export OLLAMA_CONCURRENT_REQUESTS=1  # 🔧 SEQUENTIAL: Only 1 request at a time
export OLLAMA_REQUEST_TIMEOUT=1200    # 🔧 CRITICAL: 20 min timeout per temporal complexity
export OLLAMA_LOAD_TIMEOUT=1800       # 🔧 EXTENDED: 30 min per model loading
export OLLAMA_GPU_OVERHEAD=0         # 🚀 MAXIMUM: No GPU overhead limit
export OLLAMA_MEMORY_LIMIT=64GB      # 🚀 MAXIMUM: Full A100 64GB memory
export OLLAMA_LLM_LIBRARY="cuda_v12"
export OLLAMA_FLASH_ATTENTION=1

# 🚀 MAXIMUM GPU UTILIZATION - Force all layers on GPU
export OLLAMA_GPU_MEMORY_FRACTION=0.95  # 🚀 MAXIMUM: 95% VRAM for maximum GPU usage
export OLLAMA_CUDA_VISIBLE_DEVICES=0
export OLLAMA_MAX_CONTEXT=2048  # 🚀 INCREASED: Aligned with Python config
export OLLAMA_BATCH_SIZE=512    # 🚀 DOUBLED: Maximum batch size for GPU throughput
export OLLAMA_NUM_GPU_LAYERS=-1          # 🚀 CRITICAL: Force ALL layers on GPU (-1 = all)
export OLLAMA_GPU_LAYERS=-1              # 🚀 ALTERNATIVE: Ensure all layers on GPU

# 🔧 TIMEOUT TEMPORAL-OPTIMIZED ALLINEATI CON PYTHON
# Timeout estesi per temporal processing complexity
export OLLAMA_SERVER_TIMEOUT=1200     # 🔧 CRITICAL: 20 min server timeout
export OLLAMA_CONNECT_TIMEOUT=120     # 🔧 TEMPORAL-EXTENDED: 2 min connect timeout

# ============= CLEANUP PREVENTIVO =============
echo ""
echo "🧹 Cleanup preventivo..."
pkill -f ollama 2>/dev/null || true
sleep 20

# Cleanup vecchie directory temporanee
find "$WORK" -maxdepth 1 -name "tmp_ollama_*" -type d -user $(whoami) -mmin +120 -exec rm -rf {} + 2>/dev/null || true

# ============= DEFINIZIONE VARIABILI GLOBALI =============
SERVER_PID1=""

# ============= FUNZIONE DI CLEANUP PER EXIT =============
cleanup() {
    echo ""
    echo "🧹 Cleanup finale..."
    
    # Kill processi Ollama
    if [ -n "$SERVER_PID1" ] && kill -0 $SERVER_PID1 2>/dev/null; then
        echo "Stopping PID $SERVER_PID1..."
        kill -TERM $SERVER_PID1 2>/dev/null
    fi
    
    sleep 10
    pkill -f ollama 2>/dev/null || true
    
    # Rimuovi directory temporanea
    if [ -n "$CUSTOM_TMP" ] && [ -d "$CUSTOM_TMP" ]; then
        echo "Removing $CUSTOM_TMP..."
        rm -rf "$CUSTOM_TMP"
    fi
    
    echo "✅ Cleanup completato"
}
trap cleanup EXIT

# ============= FUNZIONE DI AVVIO SENZA TIMEOUT =============
start_ollama_gpu() {
    local gpu_id=$1
    local port=$2
    local is_master=$3
    
    echo ""
    echo "🔧 Avvio GPU $gpu_id su porta $port..."
    
    # Crea cache directory dedicata
    local gpu_cache="$OLLAMA_CACHE_DIR/gpu${gpu_id}"
    mkdir -p "$gpu_cache"
    
    # 🔴 CRITICO: Nessun comando timeout, processo libero di vivere
    CUDA_VISIBLE_DEVICES=$gpu_id \
    OLLAMA_HOST=127.0.0.1:$port \
    OLLAMA_MAX_LOADED_MODELS=1 \
    OLLAMA_TMPDIR="$CUSTOM_TMP" \
    OLLAMA_CACHE_DIR="$gpu_cache" \
    $OLLAMA_BIN serve > llama3time_ollama_gpu${gpu_id}.log 2>&1 &
    
    local pid=$!
    echo "✅ GPU $gpu_id PID: $pid (NO TIMEOUT)"
    
    # Salva PID globalmente
    eval "SERVER_PID$((gpu_id+1))=$pid"
    
    # Verifica che il processo sia vivo
    sleep 5
    if ! kill -0 $pid 2>/dev/null; then
        echo "❌ Processo GPU $gpu_id morto immediatamente!"
        tail -20 llama3time_ollama_gpu${gpu_id}.log
        return 1
    fi
    
    # Se è la GPU master, aspetta il caricamento completo
    if [ "$is_master" = "true" ]; then
        echo "⏳ GPU $gpu_id è MASTER - attesa caricamento modello SENZA LIMITI..."
        
        local attempts=0
        while true; do
            ((attempts++))
            
            # Check processo ancora vivo
            if ! kill -0 $pid 2>/dev/null; then
                echo "❌ Processo GPU $gpu_id terminato inaspettatamente!"
                echo "📜 Ultimi log:"
                tail -30 llama3time_ollama_gpu${gpu_id}.log
                return 1
            fi
            
            # Test API - timeout coerente
            if curl -s --connect-timeout 30 "http://127.0.0.1:$port/api/tags" >/dev/null 2>&1; then
                echo "   🌐 API risponde, test modello..."
                
                # Test caricamento modello - usa il modello raccomandato - ULTRA TIMEOUT
                local test_response=$(curl -s -X POST \
                    --connect-timeout 30 \
                    --max-time 300 \
                    "http://127.0.0.1:$port/api/generate" \
                    -H "Content-Type: application/json" \
                    -d '{
                        "model":"mistral:7b",
                        "prompt":"Hi",
                        "stream":false,
                        "options":{
                            "num_predict":1,
                            "num_ctx":2048,
                            "num_batch":512,
                            "num_thread":112,
                            "num_gpu_layers":-1,
                            "temperature":0.4
                        }
                    }' 2>&1)
                
                if echo "$test_response" | grep -q '"done":true'; then
                    echo "   ✅ GPU $gpu_id PRONTA dopo $attempts tentativi!"
                    return 0
                elif echo "$test_response" | grep -q "model.*not found"; then
                    echo "   ⚠️ Modello non trovato, potrebbe essere in download..."
                fi
            fi
            
            # Feedback periodico
            if [ $((attempts % 10)) -eq 0 ]; then
                echo "   ⏳ Tentativo $attempts - GPU $gpu_id ancora in caricamento..."
                echo "   📊 Memoria GPU:"
                nvidia-smi --id=$gpu_id --query-gpu=memory.used,memory.total --format=csv,noheader
                
                # Check log per progresso
                local progress=$(grep "model load progress" llama3time_ollama_gpu${gpu_id}.log | tail -1)
                [ -n "$progress" ] && echo "   📈 $progress" || echo "   📈 Modello in caricamento su GPU $gpu_id..."
            fi
            
            sleep 30  # Check ogni 30 secondi
            
            # Safety check dopo 30 minuti
            if [ $attempts -gt 60 ]; then
                echo "   ⚠️ GPU $gpu_id impiega più di 30 minuti..."
                echo "   Continuo ad aspettare (Ctrl+C per interrompere)..."
            fi
        done
    fi
    
    return 0
}

# ============= AVVIO SEQUENZIALE CONTROLLATO =============
echo ""
echo "🚀 AVVIO SISTEMA OLLAMA"
echo "========================"

# 🔧 SEQUENTIAL-OPTIMIZED: Avvia solo 1 GPU per sequential processing stability
echo "🔧 STRATEGIA SEQUENTIAL-OPTIMIZED: Avvio singola GPU per processing sequenziale con Mistral 7B"
if ! start_ollama_gpu 0 39001 true; then
    echo "❌ ERRORE CRITICO: GPU 0 fallita"
    exit 1
fi

echo ""
echo "✅ GPU 0 completamente operativa con modello Mistral 7B caricato"
echo "⏳ Pausa 90s per stabilizzazione completa..."
sleep 90

# Disabilitazione di tutte le altre GPU per esecuzione sequenziale
echo "📝 GPU 1, 2 e 3 disabilitate per modalità sequenziale"

echo "⏳ Attesa finale stabilizzazione sistema (60s)..."
sleep 60

# ============= VERIFICA FINALE =============
echo ""
echo "🔍 VERIFICA FINALE SISTEMA"
echo "==========================="

# 🔧 SEQUENTIAL: Test solo la GPU attiva (GPU 0)
WORKING_GPUS=0
WORKING_PORTS=""

for i in 0; do
    port=$((39001 + i))
    
    echo -n "GPU $i (porta $port): "
    
    # Test completo - ULTRA TIMEOUT per coerenza Python
    if curl -s "http://127.0.0.1:$port/api/tags" >/dev/null 2>&1; then
        test_resp=$(curl -s -X POST \
            --max-time 300 \
            "http://127.0.0.1:$port/api/chat" \
            -H "Content-Type: application/json" \
            -d '{
                "model":"mistral:7b",
                "messages":[{"role":"user","content":"Say OK"}],
                "stream":false,
                "options":{
                    "num_predict":2,
                    "num_ctx":2048,
                    "num_batch":512,
                    "num_thread":112,
                    "num_gpu_layers":-1,
                    "temperature":0.4
                }
            }' 2>&1)
        
        if echo "$test_resp" | grep -q '"done":true'; then
            echo "✅ OPERATIVA"
            ((WORKING_GPUS++))
            [ -z "$WORKING_PORTS" ] && WORKING_PORTS="$port" || WORKING_PORTS="$WORKING_PORTS,$port"
        else
            echo "⚠️ API risponde ma modello non pronto"
        fi
    else
        echo "❌ NON RISPONDE"
    fi
done

echo ""
echo "📊 RISULTATO: $WORKING_GPUS/1 GPU operativa (modalità sequenziale)"

if [ $WORKING_GPUS -eq 0 ]; then
    echo "❌ ERRORE: Nessuna GPU operativa!"
    echo ""
    echo "=== Log GPU 0 (ultime 30 righe) ==="
    tail -30 llama3time_ollama_gpu0.log 2>/dev/null || echo "Log non disponibile"
    exit 1
fi

# Salva porte funzionanti
echo "$WORKING_PORTS" > ollama_ports.txt
echo "✅ Porte salvate: $WORKING_PORTS"

# ============= MONITORING AVANZATO GPU =============
advanced_gpu_monitor() {
    echo "📊 Starting Advanced GPU Monitor (ogni 60s)"
    
    while true; do
        sleep 180
        
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "📊 GPU STATUS - $(date '+%Y-%m-%d %H:%M:%S')"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        # Mostra utilizzo GPU dettagliato
        nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits | \
        while IFS=',' read -r idx name util_gpu util_mem mem_used mem_total temp power; do
            # Calcola percentuale memoria
            mem_percent=$(echo "scale=1; $mem_used * 100 / $mem_total" | bc -l 2>/dev/null || echo "0")
            
            # Colori per output (se supportato)
            if [ "$util_gpu" -gt 80 ]; then
                status="🔥 HIGH"
            elif [ "$util_gpu" -gt 50 ]; then
                status="✅ GOOD"
            elif [ "$util_gpu" -gt 10 ]; then
                status="⚡ LOW"
            else
                status="💤 IDLE"
            fi
            
            printf "GPU %s: %s\n" "$idx" "$status"
            printf "  Compute: %3d%% | Memory: %3d%% (%s/%s MB)\n" \
                   "$util_gpu" "$util_mem" "$mem_used" "$mem_total"
            printf "  Temp: %d°C | Power: %s W\n" "$temp" "$power"
            echo ""
        done
        
        # Mostra processi Ollama (solo GPU 0 attiva)
        echo "🔄 Processi Ollama (modalità sequenziale):"
        for i in 0; do
            eval "pid=\$SERVER_PID$((i+1))"
            port=$((39001 + i))
            
            if [ -n "$pid" ] && kill -0 $pid 2>/dev/null; then
                # CPU usage del processo
                cpu_usage=$(ps -p $pid -o %cpu= 2>/dev/null | tr -d ' ' || echo "0")
                # Memoria del processo
                mem_usage=$(ps -p $pid -o rss= 2>/dev/null | awk '{printf "%.1f", $1/1024/1024}' || echo "0")
                
                echo "  GPU $i (PID $pid): ✅ CPU: ${cpu_usage}% | RAM: ${mem_usage}GB | Port: $port"
                
                # Test veloce della porta - timeout coerente con Python
                if timeout 30s curl -s "http://127.0.0.1:$port/api/tags" >/dev/null 2>&1; then
                    echo "    └─ API: ✅ Responsive"
                else
                    echo "    └─ API: ⚠️ Slow/Unresponsive"
                fi
            else
                echo "  GPU $i: ❌ Process not running"
            fi
        done
        
        # Statistiche Python se in esecuzione
        if pgrep -f "veronacard_mob_with_geom_time" >/dev/null; then
            echo ""
            echo "🐍 Python Processing (Sequential Mistral 7B):"

            # Linee processate dal log
            if [ -f "mistral_sequential_execution.log" ]; then
                processed=$(grep -c "Processing card" mistral_sequential_execution.log 2>/dev/null || echo "0")
                errors=$(grep -c "ERROR\|Error" mistral_sequential_execution.log 2>/dev/null || echo "0")
                success_rate=$(grep -c "SUCCESS" mistral_sequential_execution.log 2>/dev/null || echo "0")
                echo "  Cards processed: $processed"
                echo "  Success rate: $success_rate"
                echo "  Errors: $errors"
            fi
        else
            echo ""
            echo "🐍 Python: Not running or completed"
        fi
        
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    done
}

# Avvia monitor avanzato in background
advanced_gpu_monitor &
ADV_MONITOR_PID=$!

# ============= ESECUZIONE PYTHON =============
cd /leonardo_work/IscrC_LLM-Mob/LLM-Mob-As-Mobility-Interpreter

echo ""
echo "🐍 AVVIO PYTHON"
echo "==============="
echo ""

if [ -f "data/verona/vc_site.csv" ]; then
    echo "🔧 SEQUENTIAL: Processing con --append e configurazione sequenziale con Mistral 7B"
    python3 -u veronacard_mob_with_geom_time_parrallel_ollama.py \
        --append 2>&1 | tee mistral_sequential_execution.log
    PYTHON_EXIT=$?
else
    echo "❌ File non trovato!"
    PYTHON_EXIT=1
fi

# Stop monitors
kill $ADV_MONITOR_PID 2>/dev/null || true

echo ""
echo "============================================"
echo "📊 JOB COMPLETATO"
echo "Exit code Python: $PYTHON_EXIT"
echo "GPU utilizzate: $WORKING_GPUS"
echo "Tempo totale: $SECONDS secondi"
echo "============================================"

exit $PYTHON_EXIT