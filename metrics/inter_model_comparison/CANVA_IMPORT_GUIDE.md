# 🎨 GUIDA IMPORT CSV IN CANVA - Confronto Modelli LLM

## 📊 File CSV Creati e Grafici Consigliati

### 1. **BAR_CHART_model_performance_comparison.csv**
**📊 Grafico: BAR CHART (Grafico a barre orizzontali)**
- **Utilizzo**: Confronto performance complessiva tra modelli
- **Import Canva**:
  - X-axis: `model_name`
  - Y-axis: `performance_score` o `hit_rate`
  - Colore: `organization`
- **Messaggio**: Quale modello performa meglio nel complesso

### 2. **COLUMN_CHART_strategy_hit_rates.csv**
**📊 Grafico: COLUMN CHART (Grafico a colonne verticali)**
- **Utilizzo**: Confronto efficacia delle strategie
- **Import Canva**:
  - X-axis: `strategy_name`
  - Y-axis: `hit_rate`
- **Messaggio**: Quale strategia funziona meglio per la predizione

### 3. **SCATTER_PLOT_hit_rate_vs_speed.csv**
**📊 Grafico: SCATTER PLOT (Grafico a dispersione)**
- **Utilizzo**: Analisi hit rate vs velocità di processing
- **Import Canva**:
  - X-axis: `Processing_Time`
  - Y-axis: `Hit_Rate`
  - Colore/Dimensione: `model_name`
- **Messaggio**: Trade-off tra accuratezza e velocità

### 4. **PIE_CHART_predictions_distribution.csv**
**📊 Grafico: PIE CHART (Grafico a torta)**
- **Utilizzo**: Distribuzione del carico di lavoro tra modelli
- **Import Canva**:
  - Etichette: `Model`
  - Valori: `Percentage`
- **Messaggio**: Quanto lavoro ha svolto ogni modello

### 5. **HEATMAP_model_strategy_hit_rates.csv**
**📊 Grafico: HEATMAP (Mappa di calore)**
- **Utilizzo**: Matrice performance modello x strategia
- **Import Canva**:
  - Righe: modelli (prima colonna)
  - Colonne: strategie (header)
  - Valori: hit rates (intensità colore)
- **Messaggio**: Quale combinazione modello-strategia è ottimale

### 6. **RADAR_CHART_top_model_profiles.csv**
**📊 Grafico: RADAR CHART (Grafico radar/ragnatela)**
- **Utilizzo**: Profilo completo dei top 5 modelli
- **Import Canva**:
  - Etichette: `model_name`
  - Assi: `Hit_Rate`, `Success_Rate`, `Speed_Score`, `Overall_Score`
- **Messaggio**: Punti di forza e debolezza di ogni modello

### 7. **TABLE_detailed_model_ranking.csv**
**📊 Formato: TABLE (Tabella)**
- **Utilizzo**: Classifica dettagliata con tutti i numeri
- **Import Canva**: Import diretto come tabella
- **Messaggio**: Ranking completo con metriche dettagliate

### 8. **DONUT_CHART_anchor_point_distribution.csv**
**📊 Grafico: DONUT CHART (Grafico a ciambella)**
- **Utilizzo**: Distribuzione strategia anchor point
- **Import Canva**:
  - Etichette: `Anchor_Point`
  - Valori: `Percentage`
- **Messaggio**: Confronto tra approcci Middle vs Penultimate

### 9. **LINE_CHART_organization_performance_trends.csv**
**📊 Grafico: LINE CHART (Grafico a linee)**
- **Utilizzo**: Trend performance per organizzazione
- **Import Canva**:
  - X-axis: `strategy_name`
  - Y-axis: `hit_rate`
  - Linee diverse: `organization`
- **Messaggio**: Come ogni organizzazione migliora con strategie avanzate

### 10. **AREA_CHART_complexity_vs_performance.csv**
**📊 Grafico: AREA CHART (Grafico ad area)**
- **Utilizzo**: Relazione complessità strategia vs performance
- **Import Canva**:
  - X-axis: `complexity_level`
  - Y-axis: `hit_rate`
  - Area: rappresenta il progresso
- **Messaggio**: Il valore aggiunto della complessità

## 🎯 Combinazioni Consigliate per Slide

### **Slide 1: Overview Performance**
- **BAR_CHART_model_performance_comparison.csv** (principale)
- **TABLE_detailed_model_ranking.csv** (dettaglio)

### **Slide 2: Analisi Strategie**
- **COLUMN_CHART_strategy_hit_rates.csv** (principale)
- **AREA_CHART_complexity_vs_performance.csv** (trend)

### **Slide 3: Trade-off e Ottimizzazione**
- **SCATTER_PLOT_hit_rate_vs_speed.csv** (principale)
- **RADAR_CHART_top_model_profiles.csv** (profili)

### **Slide 4: Distribuzione e Configurazioni**
- **PIE_CHART_predictions_distribution.csv** (carico lavoro)
- **DONUT_CHART_anchor_point_distribution.csv** (configurazioni)

### **Slide 5: Heatmap e Insights**
- **HEATMAP_model_strategy_hit_rates.csv** (matrice completa)
- **LINE_CHART_organization_performance_trends.csv** (trend organizzazioni)

## 📝 Consigli per Canva

1. **Colori Coerenti**: Usa sempre gli stessi colori per gli stessi modelli/organizzazioni
2. **Font Leggibili**: Arial o Helvetica per massima leggibilità
3. **Scala Appropriata**: Assicurati che i grafici siano proporzionati ai dati
4. **Etichette Chiare**: Aggiungi titoli descrittivi a ogni grafico
5. **Legenda**: Includi sempre una legenda quando usi colori/forme diverse

## ⚡ Quick Import Steps

1. **Apri Canva** → Crea presentazione
2. **Aggiungi elemento** → Grafici
3. **Seleziona tipo grafico** (seguendo la guida sopra)
4. **Carica CSV** → Seleziona il file appropriato
5. **Configura assi** secondo le indicazioni
6. **Personalizza stile** per coerenza visiva

🎯 **Risultato**: Presentazione professionale con dati reali sui confronti tra modelli LLM!