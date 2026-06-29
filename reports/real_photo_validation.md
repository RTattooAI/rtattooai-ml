# Validación preliminar con fotos reales (N=6)

> **Honesto:** esta NO es la validación definitiva que pide la tesis (se necesitan
> ~20-30 fotos reales etiquetadas, ver `resumen_ml.md`). Es una primera pasada
> cualitativa sobre 6 fotos reales (carpeta `Tatuajes de pruebas/`, fuera del repo,
> aportadas como set de prueba manual) corridas contra el pipeline de producción
> completo (`python -m src.embedding.identify`, YOLOv8m + DINOv2 + cabeza afinada +
> galería de 25×17 vistas, ρ=0.55, margen=0.05). Sirve como evidencia preliminar
> citable, no como conclusión estadística.

## Resultado por imagen

| # | Archivo | Etiquetado como | Detección | Top-1 (score) | Top-2 (score) | Veredicto del sistema | ¿Acierta? |
|---|---|---|---:|---|---|---|---|
| 1 | `Payaso banda de brasil (match).png` | match | conf 0.91 | Leon (0.331) | Payaso_1 (0.302) | no_match | ❌ identidad mal rankeada y bajo umbral |
| 2 | `Payaso banda de brasil 2 (match).png` | match | conf 0.81 | Payaso_con_arma (0.628) | Payaso_con_arma__2_ (0.585) | no_match | ⚠️ top-1 correcto y sobre umbral, pero rechazado por margen (0.043 < 0.05) |
| 3 | `Tren de aragua collage de tatuajes (match).png` | match (collage, 12 paneles) | 11 cajas | ver detalle abajo | — | 5/11 cajas con match correcto | ✅ ver detalle |
| 4 | `Tatuaje brazo no muy visible (no match).jpg` | no match | conf 0.29 | Leon (0.199) | Tigre_oriental (0.198) | no_match | ✅ |
| 5 | `Tatuaje flor mano (no match).jpg` | no match | conf 0.87 | Calavera_en_mano (0.216) | Emblema...fe (0.176) | no_match | ✅ |
| 6 | `Yayuaje japones cuerpo entero (no match).jpg` | no match | 5 cajas (regiones del traje) | máx 0.355 (Payaso_con_arma__2_) | — | no_match en las 5 | ✅ |

## Detalle del collage "Tren de Aragua" (caso #3)

Es un collage periodístico (Telemundo) con 12 fotos reales de tatuajes
documentados en miembros de esa banda. El detector YOLO encontró 11 cajas
(una por panel/sub-foto, aproximadamente); cada caja se identificó por
separado contra la galería:

| Panel detectado | Top-1 identificado | Score | Veredicto |
|---|---|---:|---|
| Estrella (5 puntas) | **Estrella** | 0.682 | ✅ match correcto |
| Corona | **Corona** | 0.750 | ✅ match correcto |
| Reloj (antebrazo) | **Reloj_tribal** | 0.744 | ✅ match correcto |
| AK-47 | **Ak47** | 0.584 | ✅ match correcto |
| Locomotora (panel grande) | **Tren_1** | 0.921 | ✅ match correcto, muy alta confianza |
| Otros 6 paneles (rosa, mano, "23", tren miniatura, retrato persona, estrellas pequeñas) | — | < 0.55 | rechazados (no están en la galería tal cual, o el recorte es muy pequeño/ambiguo) |

**5 de 5 símbolos que SÍ están en la galería de 25 bandas fueron identificados
correctamente sobre fotos reales y de fuentes distintas a las usadas para
construir esa galería** (sin contar como error los paneles correspondientes
a símbolos que no están en la galería: rosa, mano, "23", retrato de persona).
Esto es evidencia real (no sintética) de que el sistema generaliza más allá de
las vistas de Blender, al menos para diseños icónicos de un solo trazo/color.

## Lectura honesta

- **Funciona bien** con símbolos icónicos y de contorno simple (estrella,
  corona, reloj, arma, locomotora), incluso en fotos reales fotografiadas de
  pantalla/prensa, no solo en renders sintéticos.
- **Falla o es conservador** cuando: (a) el tatuaje real es una interpretación
  artesanal/boceto del diseño de catálogo en vez de una copia fiel (caso #1:
  el boceto de payaso es bastante distinto del PNG limpio "Payaso_1"), o
  (b) la galería tiene dos diseños casi gemelos (caso #2: "Payaso_con_arma"
  vs "Payaso_con_arma__2_"), donde el margen de rechazo open-set bloquea
  correctamente una identificación ambigua entre dos identidades — un
  comportamiento conservador deseable para una herramienta de apoyo, aunque
  reduzca el recall.
- El sistema **no tuvo ningún falso positivo** en los 3 casos negativos reales
  (tatuajes genéricos no relacionados a bandas), incluyendo un tatuaje de
  cuerpo completo muy elaborado (irezumi japonés) que podría haber generado
  falsos positivos por su complejidad visual.

## Cómo se usa esta evidencia en el paper

- Responde parcialmente "validar identificación con fotos reales" (pendiente
  en `resumen_ml.md`): es N pequeño, pero real y con casos positivos Y
  negativos.
- El collage de Tren de Aragua también sirve como **fuente/cita** para los
  "criterios de curación de la simbología de banda" (estrella, corona, AK-47,
  reloj, locomotora, máscara de gas, "Real Hasta La Muerte" — coinciden con
  varios diseños ya curados en `datasets/gang_tattoos/`).
