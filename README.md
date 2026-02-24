# Softwaredesign Abschlussprojekt
**Autoren:** Nico Dörr, Elias Spiegl  
**Semester:** 3 (2025 / 26) 

## Projektbeschreibung

Dieses Projekt ist eine interaktive Web-Anwendung zur 2D-Topologieoptimierung, die mit Python und Streamlit entwickelt wurde. Die Software ermöglicht es Anwendern, einen zweidimensionalen Bauraum zu definieren, Randbedingungen (wie Fest- und Loslager) sowie externe Kräfte auf diskrete Massepunkte aufzubringen. 

Der zugrundeliegende Löser basiert auf einer vereinfachten Finite-Elemente-Methode (FEM). Die Struktur wird als ein Gitter aus Knoten modelliert, die durch elastische Federelemente (horizontal, vertikal und diagonal) verbunden sind. Das Optimierungsziel ist es, Material iterativ zu entfernen, während die maximale Steifigkeit erhalten bleibt und physikalische Singularitäten vermieden werden.



## Installation und Ausführung

### Voraussetzungen
Stelle sicher, dass Python (Version 3.10 oder neuer empfohlen) auf deinem System installiert ist.

### 1. Abhängigkeiten installieren
Klone das Repository und installiere die benötigten Bibliotheken aus der `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 2. Anwendung starten
Führe das Hauptskript über Streamlit aus in dem du im Terminal folgenden Befehl eingibst:
```bash
streamlit run main.py
```

### Erklärung zur Ausführung: 
Dieser Befehl startet einen lokalen Streamlit-Webserver. Daraufhin öffnet sich automatisch dein Standard-Webbrowser (standardmäßig unter der lokalen Adresse http://localhost:8501). In dieser grafischen Benutzeroberfläche (Web-UI) kannst du das Programm interaktiv bedienen. Die UI ist grob in Sidebar-Workflows (Modellverwaltung, Kräfte/Lager-Definition) und eine zentrale Visualisierungs- sowie Steuerungsansicht für den Optimierer unterteilt.



## Erweiterungen im Projekt
Um die Qualität der generierten Topologien, die Recheneffizienz und die allgemeine User Experience zu verbessern, haben wir das Projekt um folgende Funktionen erweitert:

### 1. Dynamische Entfernungsraten (Remove-Ratio)
Der Algorithmus passt die Aggressivität, mit der Material entfernt wird, während der Laufzeit selbstständig an. In den ersten Iterationen werden vorsichtig nur 1% der Knoten entfernt, um Grundstrukturen aufzubauen. Später erhöht sich die Rate auf 1.5%, um Rechenzeit zu sparen und effizienter zur Ziel-Masse zu gelangen.

### 2. Sensitivitätsfilter & Struktur-Bestrafung
Anstatt Knoten rein nach ihrer individuellen Energie zu entfernen, nutzt unser Optimierer einen Nachbarschafts-Filter. Die Energie eines Knotens wird mit der seiner Nachbarn geglättet ("Sensitivity Filter"). Zusätzlich bestraft der Algorithmus dünne "Zick-Zack-Linien" (Knoten mit nur wenigen Nachbarn) massiv. Dadurch wird die Bildung von stabilen, massiven Netzwerken (z.B. Dreiecksstrukturen) gefördert und nutzlose, instabile Fäden werden frühzeitig gelöscht.

### 3. Intelligentes Post-Processing ("Beautifier")
Nach Erreichen der Zielmasse kann ein Glättungs-Algorithmus ("Nach Optimierung verschönern") ausgeführt werden. Dieser durchläuft die Topologie und schließt kleine Löcher, entfernt isolierte "Ausreißer" (Spitzen) und verdickt lokal stark beanspruchte Hotspots basierend auf der Dehnungsenergie. Dies führt zu realistischeren, glatteren und fertigungsgerechteren Strukturen.

### 4. FEM-Visualisierung (Farbskalen)
Wir haben eine "Plasma"-Farbskala integriert, die Elemente basierend auf ihrer Auslastung (axiale Dehnung, Kraft, elastische Energie oder Energie pro Länge) einfärbt. Über die UI kann die Linienstärke angepasst und der Fokus auf bestimmte Elementausrichtungen (z.B. nur Diagonalen oder nur Horizontal/Vertikal) gelegt werden. Dies hilft bei der Analyse des Kraftflusses.

