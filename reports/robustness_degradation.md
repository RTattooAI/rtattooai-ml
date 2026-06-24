# Robustez del embedding a la degradación del tatuaje

- Galería: 25 tatuajes; 4 versiones por nivel.
- Degradación: decoloración + borrado parcial + desenfoque + envejecimiento + oclusión (nivel 0 = limpio).

| Nivel degradación | Coseno ↔ diseño limpio | Top-1 |
| --- | --- | --- |
| 0.0 | 1.000 | 100.0% |
| 0.2 | 0.968 | 100.0% |
| 0.4 | 0.923 | 99.0% |
| 0.6 | 0.852 | 90.0% |
| 0.8 | 0.677 | 54.0% |

**Tolerancia:** el sistema mantiene top-1 ≥ 80% hasta un nivel de degradación de **0.6**.
