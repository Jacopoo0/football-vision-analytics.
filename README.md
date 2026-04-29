# ⚽ Football Match Analysis AI

> **Analizza video di partite di calcio in tempo reale** - Identifica automaticamente le squadre e visualizza statistiche live in una dashboard laterale interattiva.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Status](https://img.shields.io/badge/Status-Active-green)
![License](https://img.shields.io/badge/License-MIT-orange)

---

## 📺 Come Funziona

L'applicazione **elabora video di partite di calcio** e fornisce un'analisi visuale in tempo reale:

```
VIDEO INPUT                    PROCESSING                    OUTPUT
    ↓                              ↓                            ↓
  [Match]  →  [Team Detection]  →  [Dashboard Display]
              [Player Tracking]     • Team Identification
              [Statistics]          • Live Match Stats
              [Data Analysis]       • Performance Metrics
```

### 🎯 Flusso di Utilizzo

1. **Carica un video** di una partita di calcio
2. L'AI **identifica automaticamente** le due squadre
3. **Analizza il gameplay** in tempo reale
4. Mostra il video con una **dashboard laterale** contenente:
   - Nome e logo delle squadre
   - Punteggio e tempo di gioco
   - Statistiche in tempo reale
   - Metriche di performance

---

## ✨ Funzionalità Principali

### 🤖 Riconoscimento Squadre
- **Identificazione automatica** delle maglie/colori delle squadre
- **Riconoscimento logo** e informazioni ufficiali
- Classificazione in tempo reale dei giocatori per team

### 📊 Dashboard Laterale
Visualizza contemporaneamente al video:
- 👥 **Nomi squadre** e stemmi
- 🎯 **Punteggio** aggiornato
- ⏱️ **Tempo di gioco** e fase del match
- 📈 **Statistiche live**: possesso palla, tiri, falli, etc.
- 🏃 **Performance giocatori**: velocità, distanza percorsa, passaggi
- 🔄 **Event tracking**: gol, ammonizioni, sostituzioni

### 🎬 Video Processing
- Supporta **video MP4, AVI, MOV**
- Elaborazione frame-by-frame
- Overlay di dati in tempo reale
- Output video annotato salvabile

---

## 🛠️ Tecnologie Utilizzate

| Categoria | Tecnologie |
|-----------|-----------|
| **Linguaggio** | Python 3.9+ |
| **Computer Vision** | OpenCV, YOLOv8, MediaPipe |
| **Machine Learning** | TensorFlow, PyTorch, Scikit-learn |
| **Data Processing** | Pandas, NumPy |
| **Visualizzazione** | Matplotlib, Plotly, OpenCV |
| **Interfaccia** | Streamlit / PyQt5 |

---

## 🚀 Quick Start

### 1️⃣ Clona il Repository
```bash
git clone https://github.com/Jacopoo0/football-match-analysis-ai.git
cd football-match-analysis-ai
```

### 2️⃣ Installa Dipendenze
```bash
pip install -r requirements.txt
```

### 3️⃣ Esegui l'Analisi
```bash
python main.py --video match.mp4
```

### 4️⃣ Visualizza Risultati
L'applicazione aprirà una finestra con:
- **Video della partita** a sinistra
- **Dashboard dati** a destra in tempo reale

---

## 💡 Esempi di Utilizzo

### Analizzare una Partita
```python
from analysis import FootballAnalyzer

analyzer = FootballAnalyzer()
result = analyzer.analyze_video('match.mp4')

# result contiene:
# - Team 1 e Team 2 identificate
# - Frame-by-frame stats
# - Events timeline
# - Performance metrics
```

### Esportare Dashboard
```bash
python main.py --video match.mp4 --export dashboard.mp4 --format mp4
```

---

## 📁 Struttura Progetto

```
football-match-analysis-ai/
├── main.py                    # Entry point principale
├── analysis/
│   ├── team_detector.py       # Riconoscimento squadre
│   ├── player_tracker.py      # Tracking giocatori
│   └── stats_calculator.py    # Calcolo statistiche
├── dashboard/
│   ├── dashboard_builder.py   # Generazione dashboard
│   └── data_formatter.py      # Formattazione dati
├── video/
│   ├── processor.py           # Elaborazione video
│   └── overlays.py            # Overlay dati su video
├── models/                     # Modelli ML pre-trained
└── requirements.txt
```

---

## 📊 Output Generato

L'applicazione produce:

✅ **Video annotato** con overlay di squadre e statistiche  
✅ **Dashboard laterale live** con metriche aggiornate  
✅ **Report JSON** con dati completi della partita  
✅ **Timeline di eventi** (gol, cartellini, sostituzioni)  
✅ **Grafici di performance** salvabili come immagini  

---

## 🎮 Interfaccia

La dashboard laterale mostra in tempo reale:

```
┌─────────────────────┐
│   SQUAD 1  │ 2 - 1  │ SQUAD 2
├─────────────────────┤
│ ⏱️  45:30           │
├─────────────────────┤
│ Possesso Palla      │
│ ████░░░░░ 56%       │
├─────────────────────┤
│ Tiri in porta       │
│ Squad 1:    8       │
│ Squad 2:    5       │
├─────────────────────┤
│ Passaggi Completati │
│ Squad 1:   243      │
│ Squad 2:   187      │
├─────────────────────┤
│ Falli Commessi      │
│ Squad 1:   12       │
│ Squad 2:   15       │
└─────────────────────┘
```

---

## 🔮 Roadmap

- **v1.0** ✅ Riconoscimento squadre base
- **v1.1** 🔄 Tracking giocatori avanzato
- **v1.2** 📅 Prossimamente: Riconoscimento tattica/formazione
- **v2.0** 📅 Prossimamente: Multi-camera support
- **v2.1** 📅 Prossimamente: Real-time API per streaming live

---

## 🤝 Contribuire

Le pull request sono benvenute! Per modifiche significative, apri prima un issue per discutere i cambiamenti proposti.

```bash
1. Fork il repository
2. Crea un branch feature (git checkout -b feature/AmazingFeature)
3. Commit i cambiamenti (git commit -m 'Add AmazingFeature')
4. Push al branch (git push origin feature/AmazingFeature)
5. Apri una Pull Request
```

---

## 📝 License

Questo progetto è distribuito sotto la licenza MIT - vedi il file [LICENSE](LICENSE) per i dettagli.

---

## 📧 Support

Per domande o feedback: [Email](mailto:support@example.com) | [Issues](https://github.com/Jacopoo0/football-match-analysis-ai/issues)

---

**Made with ⚽ and 🤖 by [Jacopoo0](https://github.com/Jacopoo0)**
