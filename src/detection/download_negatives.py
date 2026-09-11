"""Descarga fotografías de personas SIN tatuajes desde Wikimedia Commons.

Para qué hacen falta:

  - **CP015 y el estado `no_detection`**: hasta ahora no había ninguna fotografía
    con la que comprobar que el sistema dice "no encontré tatuajes" en vez de
    inventarse algo.
  - **Negativos de calibración**: el umbral de coincidencia se calibra con
    positivos que deben aceptarse y negativos que deben rechazarse. Sin negativos
    reales el umbral se elige a ojo — que es exactamente lo que declaraba
    `match_config.json` sobre los valores heredados del paso sintético.
  - **Falsos positivos del detector**: una foto de una persona sin tatuajes en la
    que YOLO dibuja una caja es un fallo medible, y sin este conjunto no se puede
    reportar esa tasa.

Por qué Commons: las licencias son libres y verificables, y el manifiesto guarda
autor, licencia y enlace de cada archivo. En un trabajo que se va a sustentar, la
procedencia del material de prueba es parte del trabajo.

**Sobre la selección.** Un primer intento usó categorías anatómicas ("Human arms",
"Human backs") y trajo fotografía médica y material histórico de violencia: en
Commons esas categorías las alimentan sobre todo archivos clínicos e históricos.
Ahora se parte de categorías de la vida corriente, se descarta por título todo lo
que suene a contenido clínico, forense o a menores de edad, y **la selección final
es manual**: `screen_negatives.py` prepara hojas de contactos para revisarlas antes
de dar ninguna imagen por buena. Un filtro automático no sustituye a mirar.

Uso:
    python -m src.detection.download_negatives
    python -m src.detection.download_negatives --por-categoria 25
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.shared import paths

API = "https://commons.wikimedia.org/w/api.php"

# Escenas cotidianas donde se ve piel descubierta: manos, antebrazos, hombros,
# piernas. Se evitan las categorías anatómicas y las deportivas de alto nivel
# (donde el tatuaje es frecuente y arruinaría el conjunto de negativos).
CATEGORIAS = [
    "Manicure",
    "Nail art",
    "Handshakes",
    "Wristwatches worn on wrists",
    "Hands holding objects",
    "People writing",
    "Barefoot people",
    "Hand gestures",
    "Massage",
    "People cooking",
    "Yoga",
    "Swimmers",
]

# Licencias libres reconocibles. Lo que no declare una queda fuera: material sin
# licencia clara no debe acabar en el conjunto de prueba de una tesis.
LICENCIAS_OK = re.compile(r"(cc[- ]?by|cc[- ]?zero|cc0|public domain|pd-)", re.IGNORECASE)

# Todo lo que suene a clínico, forense o histórico-violento. Estas fotografías no
# aportan nada al experimento y no tienen por qué estar en el material del trabajo.
EXCLUIR_CLINICO = re.compile(
    r"(wound|injur|surger|surgic|disease|patient|dermat|lesion|scar|ulcer|rash|"
    r"amputat|mutilat|autops|corpse|cadaver|morgue|forensic|burn|fracture|"
    r"eczema|psoriasis|melanoma|tumor|tumour|blood|prosthe|anatomy|dissect|"
    r"herida|lesión|cadáver|quemadura)",
    re.IGNORECASE,
)

# Menores de edad: fuera del alcance de este trabajo y de este conjunto de datos.
EXCLUIR_MENORES = re.compile(
    r"(child|children|kid|kids|baby|babies|infant|toddler|boy|girl|school|"
    r"kindergarten|pupil|teen|juvenile|niño|niña|bebé|escolar)",
    re.IGNORECASE,
)

CABECERAS = {
    # Commons pide identificarse; una petición anónima en bucle se bloquea.
    "User-Agent": "RTattooAI-thesis/1.0 (investigacion academica; contacto via repositorio)"
}

MIN_BYTES = 40_000
MAX_BYTES = 25_000_000

# Ancho de la miniatura que se descarga. Suficiente para detectar tatuajes y
# respetuoso con el servicio, que pide expresamente no bajar originales en bloque.
ANCHO_MINIATURA = 1280

# Pausa entre descargas y espera tras un 429.
PAUSA = 1.0
ESPERA_429 = 20.0


def miembros(categoria: str, limite: int) -> list[str]:
    """Títulos de archivo de una categoría."""
    titulos: list[str] = []
    continuar = None

    while len(titulos) < limite:
        params = {
            "action": "query", "format": "json", "list": "categorymembers",
            "cmtitle": f"Category:{categoria}", "cmtype": "file",
            "cmlimit": min(500, limite - len(titulos)),
        }
        if continuar:
            params["cmcontinue"] = continuar

        r = requests.get(API, params=params, headers=CABECERAS, timeout=30)
        r.raise_for_status()
        data = r.json()

        titulos += [m["title"] for m in data.get("query", {}).get("categorymembers", [])]
        continuar = data.get("continue", {}).get("cmcontinue")
        if not continuar:
            break

    return titulos[:limite]


def informacion(titulos: list[str]) -> list[dict]:
    """URL, tamaño, autor y licencia de cada archivo. Se pide en lotes de 50."""
    salida = []

    for i in range(0, len(titulos), 50):
        lote = titulos[i : i + 50]
        # iiurlwidth pide la miniatura: Wikimedia limita la descarga en bloque de
        # originales (429) y en su propio mensaje pide usar miniaturas. 1280 px
        # sobra para lo que hace falta aqui y ademas descarga mucho mas rapido.
        r = requests.get(API, params={
            "action": "query", "format": "json", "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata", "iiurlwidth": ANCHO_MINIATURA,
            "titles": "|".join(lote),
        }, headers=CABECERAS, timeout=30)
        r.raise_for_status()

        for pagina in r.json().get("query", {}).get("pages", {}).values():
            info = (pagina.get("imageinfo") or [{}])[0]
            if not info.get("url"):
                continue
            meta = info.get("extmetadata", {})
            salida.append({
                "titulo": pagina.get("title", ""),
                "url": info.get("thumburl") or info["url"],
                "url_original": info["url"],
                "descripcion_url": info.get("descriptionurl", ""),
                "mime": info.get("mime", ""),
                "bytes": info.get("size", 0),
                "ancho": info.get("width", 0),
                "alto": info.get("height", 0),
                "autor": re.sub(r"<[^>]+>", "", meta.get("Artist", {}).get("value", "")).strip()[:200],
                "licencia": meta.get("LicenseShortName", {}).get("value", ""),
                "categorias_meta": re.sub(r"<[^>]+>", " ", meta.get("Categories", {}).get("value", ""))[:400],
            })
        time.sleep(0.3)

    return salida


def descartado_por(item: dict) -> str | None:
    """Motivo por el que no se descarga, o None si es aceptable."""
    texto = f"{item['titulo']} {item.get('categorias_meta', '')}"

    if EXCLUIR_CLINICO.search(texto):
        return "clinico"
    if EXCLUIR_MENORES.search(texto):
        return "menores"
    if item["mime"] not in ("image/jpeg", "image/png"):
        return "formato"
    if not (MIN_BYTES <= item["bytes"] <= MAX_BYTES):
        return "tamano"
    if min(item["ancho"], item["alto"]) < 400:
        return "resolucion"
    if not LICENCIAS_OK.search(item["licencia"] or ""):
        return "licencia"
    return None


def nombre_seguro(titulo: str) -> str:
    base = titulo.replace("File:", "")
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
    return base[:110]


def main() -> int:
    ap = argparse.ArgumentParser(description="Descarga negativos (personas sin tatuajes) de Wikimedia Commons")
    ap.add_argument("--por-categoria", type=int, default=25)
    ap.add_argument("--salida", default=str(paths.DATASETS / "negativos_personas" / "pendientes"))
    ap.add_argument("--candidatos", type=int, default=150,
                    help="cuántos títulos pedir por categoría antes de filtrar")
    ap.add_argument("--categorias", default=None,
                    help="categorías de Commons separadas por ';'. Por defecto, las de "
                         "CATEGORIAS. Sirve para reunir un corpus distinto sin tocar el "
                         "que ya produjo el banco de pruebas.")
    args = ap.parse_args()

    categorias = ([c.strip() for c in args.categorias.split(";") if c.strip()]
                  if args.categorias else CATEGORIAS)

    destino = Path(args.salida)
    destino.mkdir(parents=True, exist_ok=True)

    manifiesto: list[dict] = []
    vistos: set[str] = set()
    descartes: dict[str, int] = {}

    for categoria in categorias:
        print(f"\n{categoria}", flush=True)
        try:
            titulos = miembros(categoria, args.candidatos)
        except requests.RequestException as exc:
            print(f"  no se pudo consultar: {exc}", flush=True)
            continue

        items = informacion(titulos)
        aceptables = []
        for item in items:
            motivo = descartado_por(item)
            if motivo:
                descartes[motivo] = descartes.get(motivo, 0) + 1
            else:
                aceptables.append(item)

        print(f"  {len(titulos)} candidatos, {len(aceptables)} pasan los filtros", flush=True)

        descargados = 0
        for item in aceptables:
            if descargados >= args.por_categoria:
                break
            if item["url"] in vistos:
                continue
            vistos.add(item["url"])

            archivo = destino / nombre_seguro(item["titulo"])
            if archivo.exists():
                descargados += 1
                continue

            contenido = None
            for intento in range(3):
                try:
                    r = requests.get(item["url"], headers=CABECERAS, timeout=90)
                    if r.status_code == 429:
                        # El servicio pide bajar el ritmo: se espera en vez de insistir.
                        print(f"  429, esperando {ESPERA_429:.0f} s…", flush=True)
                        time.sleep(ESPERA_429 * (intento + 1))
                        continue
                    r.raise_for_status()
                    contenido = r.content
                    break
                except requests.RequestException as exc:
                    print(f"  fallo al descargar {item['titulo']}: {exc}", flush=True)
                    break

            if contenido is None:
                continue

            archivo.write_bytes(contenido)
            item["archivo"] = archivo.name
            item["categoria"] = categoria
            item.pop("categorias_meta", None)
            manifiesto.append(item)
            descargados += 1
            time.sleep(PAUSA)

        print(f"  descargadas: {descargados}", flush=True)

    ruta_manifiesto = destino / "manifiesto.json"
    previos = []
    if ruta_manifiesto.exists():
        previos = json.loads(ruta_manifiesto.read_text(encoding="utf-8"))

    conocidos = {x["archivo"] for x in previos}
    todos = previos + [x for x in manifiesto if x["archivo"] not in conocidos]
    ruta_manifiesto.write_text(json.dumps(todos, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nDescartes por filtro: {descartes}", flush=True)
    print(f"Total en {destino}: {len(todos)} imágenes", flush=True)
    print(f"Manifiesto con autor y licencia: {ruta_manifiesto}", flush=True)
    print("\nLas imágenes quedan en 'pendientes': hay que REVISARLAS a ojo antes de darlas", flush=True)
    print("por buenas. Siguiente paso: python -m src.detection.screen_negatives", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
