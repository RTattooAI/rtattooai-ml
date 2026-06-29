# Benchmark de latencia local (serving/) — proxy mientras no hay despliegue real

> Medido en hardware local (Intel64 Family 6 Model 154 Stepping 3, GenuineIntel), NO en el Container App de Azure final. Sirve como cota orientativa hasta poder medir en el entorno real de despliegue; re-medir con esta misma herramienta una vez desplegado.

| Escenario | N | cold start (ms) | media (ms) | mediana (ms) | p90 (ms) | p95 (ms) | máx (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full (multi-hilo) | 39 | 9319.4 | 1081.6 | 944 | 1434 | 2843 | 2978 |
| 1 vCPU (proxy) | 39 | 637.4 | 1292.3 | 1205 | 1629 | 2455 | 3666 |

**Nota:** `cold_start_ms` es el costo de cargar YOLOv8m ONNX + DINOv2 + galería una sola vez (al iniciar el contenedor); no se paga por request.
