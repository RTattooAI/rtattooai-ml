# Blender desde cero — generar vistas de tatuajes (ruta procedural)

Esta ruta NO necesita modelar ni descargar humanos: un script
(`generate_synthetic_procedural.py`) **construye toda la escena solo** (piel
curva + luz + cámara) y renderiza los recortes. Tú casi no tocas Blender.

---

## 1) Instalar Blender

1. Descarga Blender (gratis) de <https://www.blender.org/download/> (versión 4.x).
2. Instálalo y ábrelo. Verás una escena con un cubo, una cámara y una luz
   (no importa, el script borra todo y crea lo suyo).

## 2) Preparar los tatuajes y las rutas

1. Copia tus PNG de tatuajes a una carpeta, p. ej. `C:/tattoos/`. Lo ideal es
   que tengan **fondo transparente** (los `gang_tattoos` ya lo tienen).
2. Abre `blender/generate_synthetic_procedural.py` en un editor de texto y edita
   arriba:
   ```python
   INPUT_TATTOOS = r"C:/tattoos"          # carpeta con tus PNG
   OUTPUT_PATH   = r"C:/tattoos_OUTPUT"   # dónde se guardan los recortes
   ```
   - Si tus tatuajes tienen **fondo blanco** (no transparente):
     `USE_LUMINANCE_MASK = True`.

## 3) Correr en MODO TEST (rápido)

El script ya viene en test (`MAX_TATTOOS = 2`, 3 vistas c/u):

1. En Blender, arriba, haz clic en la pestaña **Scripting**.
2. En el editor de texto del centro: **Text → Open** y elige
   `generate_synthetic_procedural.py` (o pega su contenido en un texto nuevo).
3. (Windows) **Window → Toggle System Console** para ver el progreso.
4. Haz clic en **▶ Run Script** (o `Alt+P`).
5. Espera. Al terminar, abre `OUTPUT/crops/<tatuaje>/` y mira los PNG.

## 4) Revisar el test y ajustar (config arriba del script)

| Lo que ves | Qué ajustar |
|---|---|
| Todo el plano sale "tatuado" / negro | Tus PNG no tienen alpha → `USE_LUMINANCE_MASK = True` |
| El tatuaje se ve muy chico/grande | `TAT_SCALE` (sube para más grande) |
| La piel se ve muy plana | sube el rango de `BEND_ANGLE` (curvatura) |
| Tonos de piel pocos | agrega RGB a `SKIN_COLORS` |
| Muy oscuro / muy claro | `LIGHT_ENERGY` (con luz) o pon HDRIs (ver abajo) |
| Quiero más/menos variación de cámara | `CAM_AZIMUTH`, `CAM_ELEV`, `CAM_DIST` |

Repite el test hasta que los recortes se vean como "un tatuaje sobre piel".

## 5) (Opcional) Iluminación realista con HDRIs

Para que la luz se parezca a fotos reales, descarga HDRIs gratis de
<https://polyhaven.com/hdris> (elige unos pocos, formato `.hdr`), ponlos en una
carpeta y edita: `HDRI_FOLDER = r"C:/hdris"`. El script los usará al azar.

## 6) PRODUCCIÓN (todas las vistas)

En el script, pon:
```python
MAX_TATTOOS = None     # todos los tatuajes
N_PER_TATTOO = 16      # vistas por tatuaje (sube para más datos)
```
Vuelve a **Run**. Con 25 tatuajes × 16 = 400 recortes (rápido en EEVEE).
Para no bloquear Blender, también puedes correr **headless** desde una terminal:
```bash
blender --background --python blender/generate_synthetic_procedural.py
```

## 7) Llevar los recortes al pipeline ML

1. Copia la carpeta `OUTPUT/crops` a la máquina del repo, p. ej.
   `datasets/blender_crops/` (gitignored).
2. En el repo (venv activo):
   ```bash
   python -m src.embedding.build_gallery --sources datasets/gang_tattoos datasets/blender_crops
   python -m src.embedding.compare_galleries --gallery-b datasets/gang_tattoos datasets/blender_crops --synthetic-queries 6
   streamlit run app/streamlit_full.py
   ```
3. El reporte `reports/gallery_ablation.md` te dice cuánto mejoró la galería
   (evidencia para la tesis).

---

## Notas

- Es una **aproximación** (piel curva, no un humano). Suficiente para los crops
  de galería; si luego quieres más realismo, pasamos a un modelo humano
  (MakeHuman/MB-Lab) — ver `blender/README.md`.
- La validación final siempre es contra **fotos reales** (`compare_galleries
  --queries ...` y `calibrate`).
- No pude probar el script fuera de Blender (usa `bpy`): si algo falla en el
  test, copia el mensaje de la System Console y se ajusta.
