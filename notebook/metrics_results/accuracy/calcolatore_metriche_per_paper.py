import pandas as pd
from pathlib import Path
import sys

# --- Configurazione ---

# 1. Definiamo la cartella radice da cui iniziare la ricerca.
root_dir = Path('.') 

# 2. Elenco dei file specifici da cercare
target_filenames = ['top1_metrics_canva.csv', 'hit5_metrics_canva.csv']

# 3. Nome del file di output
output_filename = 'summary_metrics_results.csv'

# --- Fine Configurazione ---


def process_files():
    """
    Scansiona le sottocartelle, legge i file target, calcola le metriche
    e salva i risultati in un unico file CSV.
    """
    
    # Lista per raccogliere tutti i risultati
    all_results = []
    
    print(f"Avvio della scansione da: {root_dir.resolve()}")
    print(f"Ricerca file: {', '.join(target_filenames)}\n")
    
    for filename in target_filenames:
        
        found_files = list(root_dir.rglob(f'**/{filename}'))
        
        if not found_files:
            print(f"ATTENZIONE: Nessun file trovato per '{filename}'")
            continue
            
        print(f"Trovati {len(found_files)} file per '{filename}':")
        
        for filepath in found_files:
            try:
                # 6. Leggiamo il file CSV utilizzando pandas
                df = pd.read_csv(filepath)
                
                if df.empty or len(df.columns) < 2:
                    print(f"  - ATTENZIONE: File vuoto o colonne insufficienti: {filepath}")
                    continue
                    
                # 7. Selezioniamo la SECONDA colonna (indice 1) per i calcoli
                data_column = pd.to_numeric(df.iloc[:, 1], errors='coerce')
                data_column = data_column.dropna() 
                
                if data_column.empty:
                    print(f"  - ATTENZIONE: Nessun dato numerico valido nella seconda colonna: {filepath}")
                    continue

                # 8. Calcoliamo media e DEVIAZIONE STANDARD (MODIFICATO)
                mean_val = data_column.mean()
                max_val = data_column.max()
                std_dev_val = data_column.std() # <-- Calcoliamo la deviazione standard
                
                # 9. Estraiamo le informazioni dal percorso per il contesto
                relative_path = filepath.relative_to(root_dir)
                parts = relative_path.parts
                
                # Costruiamo un dizionario con i risultati (MODIFICATO)
                result_data = {
                    'Path': str(relative_path),
                    'Mean': mean_val,
                    'StdDev': std_dev_val,
                    'Max': max_val
                }
                
                # 10. Aggiungiamo il dizionario alla nostra lista
                all_results.append(result_data)
                
            except pd.errors.EmptyDataError:
                print(f"  - ERRORE: Il file è vuoto (EmptyDataError): {filepath}")
            except Exception as e:
                print(f"  - ERRORE: Impossibile processare il file {filepath}. Dettagli: {e}")

    # 11. Dopo aver processato tutti i file, creiamo un DataFrame riassuntivo
    if not all_results:
        print("\nOperazione completata. Nessun dato valido è stato trovato o processato.")
        return

    summary_df = pd.DataFrame(all_results)
    
    # 12. Salviamo il DataFrame in un nuovo file CSV
    try:
        summary_df.to_csv(output_filename, index=False, float_format='%.2f')
        print(f"\n--- Operazione completata! ---")
        # Messaggio di output aggiornato
        print(f"I risultati (Media e Deviazione Standard) sono stati salvati in: {output_filename}")
        print("\nAnteprima dei primi 5 risultati:")
        print(summary_df.head())
    except Exception as e:
        print(f"\nERRORE: Impossibile salvare il file di output {output_filename}. Dettagli: {e}")


# Avviamo la funzione principale
if __name__ == "__main__":
    
    # Controllo dipendenze
    try:
        import pandas
        from pathlib import Path
    except ImportError:
        print("ERRORE: Le librerie 'pandas' e 'pathlib' sono necessarie.")
        print("Installale con: pip install pandas")
        sys.exit(1)
        
    process_files()