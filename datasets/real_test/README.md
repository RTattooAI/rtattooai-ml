# datasets/real_test — Imágenes reales del contexto peruano

> **Esta carpeta está intencionalmente vacía de datos.** El contenido lo
> coloca el equipo; **no se commitea** (está en `.gitignore`).

## Propósito

Aquí van **imágenes reales de tatuajes de bandas peruanas** (p. ej. extraídas
de notas de prensa / fuentes públicas) para medir por separado la **brecha
sim-to-real**: qué tan bien generaliza el detector entrenado con datasets
públicos a las condiciones reales del caso de uso de la PNP.

Estas imágenes **no** se usan para entrenar; solo para reportar una métrica de
test independiente y honesta sobre el dominio objetivo.

## Cómo usarla

1. Coloca las imágenes aquí, en formato YOLO si tienes anotaciones:
   ```
   datasets/real_test/
   ├── images/   # .jpg / .png
   └── labels/   # .txt YOLO (clase 0 = tatuaje), opcional
   ```
2. Crea un `data.yaml` apuntando a esta carpeta (1 clase: `tatuaje`).
3. Evalúa el modelo definitivo solo sobre este conjunto y reporta la métrica
   por separado de la del test público.

## Importante (procedencia y ética)

- Documentar la **fuente y fecha** de cada imagen (Hito 7 del proyecto).
- Respetar derechos de autor y normativa de datos personales.
- Recordar que el sistema es de **apoyo preliminar**, no identificación
  definitiva.
