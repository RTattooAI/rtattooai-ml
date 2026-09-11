"""Quita de la galería base las «bandas» que en realidad son nombres de diseño.

El problema, visto en el sistema desplegado. La galería base trae 25 identidades
y su campo `bands` contiene esto:

    Ak47, Calavera_en_llamas, Jordan_23, Corona, Leon, Estrella, HG_KING,
    Mascara_de_Gas, Parca, Emblema_ornamental_con_fútbol_y_fe…

No son organizaciones criminales: son descripciones del dibujo, heredadas de cómo
se nombraron los archivos del conjunto inicial. Pero el sistema las presenta en el
campo «banda asociada», de modo que ante una coincidencia responde que el tatuaje
**pertenece a la banda «Jordan_23»**. Quien lo lea entenderá el nombre de una
organización. Es una atribución falsa por construcción, y en una herramienta cuyo
resultado puede orientar una detención es el peor error posible: no es que se
equivoque de banda, es que afirma una pertenencia que nadie ha establecido.

La pieza correcta ya existe en el sistema: una coincidencia sin banda se presenta
como `unclassified` — «este tatuaje ya se había visto, pero nadie ha determinado a
qué corresponde» — y la interfaz la muestra como «Sin clasificar», con su aviso de
que no es una atribución. Es exactamente lo que estas 25 entradas son.

Qué hace este script: vacía `bands` y **conserva `names`**. El nombre del diseño es
cierto y sirve para reconocer la entrada; lo que era falso es presentarlo como
organización. Quien determine después que un diseño corresponde a una banda real
puede asignarla desde el catálogo, que es donde esa decisión debe tomarse y quedar
registrada con su responsable.

Uso:
    python -m src.embedding.clear_gallery_bands --dry-run
    python -m src.embedding.clear_gallery_bands
    python -m src.embedding.clear_gallery_bands --galeria /app/models/gallery_dinov2.npz
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Este script tiene que poder ejecutarse tambien DENTRO del contenedor, que es
# donde vive la galeria desplegada y donde no existe el arbol del repositorio.
try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.shared import paths

    GALERIA = paths.MODELS / "gallery_dinov2.npz"
except (IndexError, ImportError):
    GALERIA = Path("/app/models/gallery_dinov2.npz")


def main() -> int:
    ap = argparse.ArgumentParser(description="Vacía las bandas falsas de la galería base")
    ap.add_argument("--galeria", default=str(GALERIA))
    ap.add_argument("--dry-run", action="store_true", help="solo informar, sin escribir")
    ap.add_argument("--sin-copia", action="store_true",
                    help="no dejar copia de seguridad (no recomendado)")
    args = ap.parse_args()

    ruta = Path(args.galeria)
    if not ruta.exists():
        print(f"No existe {ruta}.")
        return 1

    datos = np.load(ruta, allow_pickle=True)
    contenido = {k: datos[k] for k in datos.keys()}

    if "bands" not in contenido:
        print("La galería no tiene campo 'bands'; no hay nada que hacer.")
        return 0

    bandas = np.asarray(contenido["bands"], dtype=object)
    con_banda = [str(b) for b in bandas if str(b).strip()]
    distintas = sorted(set(con_banda))

    print(f"Galería: {ruta}")
    print(f"  vectores:            {len(bandas)}")
    print(f"  con banda asignada:  {len(con_banda)}")
    print(f"  bandas distintas:    {len(distintas)}")
    if distintas:
        print(f"  ejemplos:            {', '.join(distintas[:6])}")

    if not con_banda:
        print("\nYa no hay ninguna banda asignada. Nada que hacer.")
        return 0

    if args.dry_run:
        print("\n(dry-run) No se ha escrito nada.")
        return 0

    if not args.sin_copia:
        marca = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        copia = ruta.with_suffix(f".bandas-antes-{marca}.npz")
        shutil.copy2(ruta, copia)
        print(f"\nCopia de seguridad: {copia.name}")

    contenido["bands"] = np.array([""] * len(bandas), dtype=object)
    contenido["bands_cleared_at"] = np.array(
        datetime.now(timezone.utc).isoformat(timespec="seconds"))

    np.savez(ruta, **contenido)
    print(f"Escrito: {ruta}")
    print("\nLas 25 identidades siguen reconociéndose; ahora se presentan como")
    print("«Sin clasificar» en vez de atribuirse a una organización inexistente.")
    print("Reinicie el servicio de inferencia para que recargue la galería.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
