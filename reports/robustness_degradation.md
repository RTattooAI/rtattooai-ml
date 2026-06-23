# Robustez del embedding a la degradación del tatuaje

- Galería: 25 tatuajes; 4 versiones por nivel.
- Degradación: decoloración + borrado parcial + desenfoque + envejecimiento + oclusión (nivel 0 = limpio).

| Nivel degradación | Coseno ↔ diseño limpio | Top-1 |
| --- | --- | --- |
| 0.0 | 1.000 | 100.0% |
| 0.2 | 0.918 | 100.0% |
| 0.4 | 0.758 | 97.0% |
| 0.6 | 0.582 | 86.0% |
| 0.8 | 0.318 | 28.0% |

**Tolerancia:** el sistema mantiene top-1 ≥ 80% hasta un nivel de degradación de **0.6**.
