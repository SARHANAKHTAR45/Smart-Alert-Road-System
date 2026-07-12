# 🚧 Smart Road Alert System

An intelligent transportation solution that models road networks as weighted 
directed graphs and uses **Dijkstra's Algorithm with dynamic weight adjustment** 
to detect hazards, recompute safe routes, and alert drivers in real time.

## Overview
Traditional navigation systems rely on static routing and passive signage, which 
often fail to account for rapidly changing road conditions. This project takes an 
active, algorithm-driven approach — continuously ingesting sensor, weather, and 
crowdsourced hazard data, updating graph edge weights on the fly, and rerouting 
affected vehicles through an incremental Dijkstra recomputation.

## Key Features
- **Graph-based road modelling** — junctions as nodes, road segments as weighted 
  directed edges (risk score derived from surface quality, traffic density, 
  accident history, and weather).
- **Dijkstra's Algorithm (Min-Heap)** — `O((V + E) log V)` shortest-safe-path 
  computation using a `PriorityQueue<Node>`.
- **Dynamic hazard handling** — edge weights update on hazard detection, 
  triggering incremental route recomputation (~60% less overhead than full 
  re-execution).
- **Real-time alert engine** — identifies affected vehicles and dispatches 
  notifications within ~1.4s of hazard detection.
- **Java Swing GUI** — interactive graph visualization with color-coded risk 
  levels, live route highlighting, and a comparison table of algorithm 
  performance.
- **Benchmarking** — Dijkstra's vs. Bellman-Ford vs. A* across graphs of 50–500 
  nodes, measured on execution time and memory usage.

## Tech Stack
- **Language:** Java 17+
- **Core:** Java Collections Framework (`HashMap`, `PriorityQueue`, `LinkedList`)
- **GUI:** Java Swing & AWT
- **Build:** IntelliJ IDEA / Eclipse / `javac`

## Results
Dijkstra's Algorithm consistently outperformed Bellman-Ford (up to ~14x faster 
at 500 nodes) while remaining competitive with A*, making it the most suitable 
choice for real-time, dynamically updating road networks.

## Project Structure
├── Node.java # Graph node for priority queue
├── RoadNetwork.java # Weighted directed graph model
├── Dijkstra.java # Core shortest-path algorithm
├── AlertEngine.java # Hazard detection + rerouting logic
├── SmartRoadGUI.java # Swing-based visualization
└── Main.java # Entry point / simulation


## Author
Sarhan Akhtar — CS43: Design and Analysis of Algorithms, RIT Bangalore


