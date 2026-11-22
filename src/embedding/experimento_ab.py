"""Prueba de las ideas A (ensamble adaptativo) y B (predicción conformal) con tatuajes REALES.

Pregunta: si se enrola un tatuaje cualquiera desde una foto real, ¿el sistema lo
reconoce en otra versión de esa foto y rechaza los tatuajes que no conoce? ¿Mejora
eso al combinar dos codificadores, y el semáforo conformal cumple su garantía?

Datos
-----
* **Identidades:** tatuajes de by_tatoo, tattoodet y playground_tattootest. Estas
  fuentes traen varias versiones de la misma foto, hechas por los autores del
  dataset (nombres ``<persona>_<tatuaje>_<variante><n>``: a, ar, r, b, c, i). Que
  las haya hecho un tercero importa: no son nuestro simulador de degradación, así
  que la prueba no premia a un modelo por haber aprendido ese simulador.
  Se enrola UNA versión con el procedimiento del servicio (original + 4 vistas
  degradadas) y se consulta con las demás.
* **Distractores:** recortes reales de otras fuentes que también se enrolan, para
  que la galería tenga el tamaño de un catálogo real y no sea trivial.
* **Impostores:** recortes de DeMSI, tatuajes que no están en la galería y que el
  sistema debe rechazar.

Se usan las cajas anotadas, no el detector: se mide solo la etapa de comparación.

Uso:
    python -m src.embedding.experimento_ab
    python -m src.embedding.experimento_ab --distractores 300 --impostores 600
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.shared import paths
from src.embedding.degrade import degrade
from src.detection.prepare_dataset import read_label

FUENTES_ID = ["by_tatoo", "tattoodet", "playground_tattootest"]
FUENTES_DISTRACTOR = ["code_with_heady", "image_analysis_region", "lance_nelson"]
ENROLL_LEVELS = [0.2, 0.4, 0.6, 0.8]          # igual que serving/pipeline.py
NOMBRE = re.compile(r"^(\d+)_(\d+)(?:_([a-z]+)(\d+))?_(?:jpe?g|png)", re.I)
MARGEN = 0.06
LADO_MIN = 24
SALIDA = paths.REPORTS / "experimentos" / "ab_real"
REAL_CROPS_TRAIN = paths.DATASETS / "real_crops" / "train"

MODELOS = {
    "dinov2_sin_cabeza": "dinov2:sin-head",
    "dinov2_cabeza": "dinov2",
    "cnn_scratch": f"cnn:{paths.MODELS / 'cnn_scratch.pt'}",
    "cnn_imagenet": f"cnn:{paths.MODELS / 'cnn_imagenet.pt'}",
}


# ----------------------------------------------------------------- datos
def recortar(img_path: Path, lbl_path: Path) -> list[Image.Image]:
    if not lbl_path.exists():
        return []
    cajas = read_label(lbl_path)
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    out = []
    for cx, cy, w, h in cajas:
        mw, mh = w * MARGEN, h * MARGEN
        x1, x2 = int(max(0, (cx - w / 2 - mw) * W)), int(min(W, (cx + w / 2 + mw) * W))
        y1, y2 = int(max(0, (cy - h / 2 - mh) * H)), int(min(H, (cy + h / 2 + mh) * H))
        if x2 - x1 >= LADO_MIN and y2 - y1 >= LADO_MIN:
            out.append(im.crop((x1, y1, x2, y2)))
    return out


def phash(img: Image.Image) -> int:
    import imagehash
    return int(str(imagehash.phash(img, hash_size=8)), 16)


def hashes_entrenamiento() -> np.ndarray:
    """pHash de las imágenes del split train (para marcar identidades ya vistas)."""
    cache = SALIDA / "phash_train.npy"
    if cache.exists():
        return np.load(cache)
    hs = []
    for p in sorted((paths.UNIFIED_IMAGES / "train").glob("*")):
        try:
            hs.append(phash(Image.open(p).convert("RGB")))
        except Exception:  # noqa: BLE001
            pass
    arr = np.array(hs, dtype=np.uint64)
    np.save(cache, arr)
    return arr


def vista_en(h: int, train: np.ndarray, umbral: int = 4) -> bool:
    x = np.bitwise_xor(train, np.uint64(h))
    dist = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)
    return bool((dist <= umbral).any())


def identidades(min_versiones: int, train_h: np.ndarray):
    """Agrupa las versiones por tatuaje; solo fotos con un único tatuaje anotado."""
    grupos: dict[tuple, dict] = defaultdict(dict)
    for fuente in FUENTES_ID:
        for img in sorted((paths.RAW / fuente).glob("*/images/*")):
            m = NOMBRE.match(img.name)
            if not m:
                continue
            clave = (m[1], m[2])
            variante = (m[3] or "orig") + (m[4] or "")
            if variante in grupos[clave]:
                continue                                   # misma versión en otra fuente
            lbl = img.parent.parent / "labels" / (img.stem + ".txt")
            rec = recortar(img, lbl)
            if len(rec) != 1:
                continue
            grupos[clave][variante] = (rec[0], img)
    salida = []
    for clave, vers in sorted(grupos.items()):
        if len(vers) < min_versiones:
            continue
        orden = sorted(vers, key=lambda v: (v != "orig", v))  # se enrola la original si existe
        enrol = orden[0]
        vista = any(vista_en(phash(Image.open(p).convert("RGB")), train_h) for _, p in vers.values())
        salida.append({"id": f"{clave[0]}_{clave[1]}", "enrol": vers[enrol][0], "enrol_var": enrol,
                       "ruta": vers[enrol][1],
                       "consultas": [(v, vers[v][0]) for v in orden[1:]], "en_train": vista})
    return salida


def recortes_de(fuentes: list[str], n: int, rng: np.random.RandomState) -> list[Image.Image]:
    imgs = [p for f in fuentes for p in sorted((paths.RAW / f).glob("*/images/*"))]
    # una versión por foto de origen (Roboflow guarda varias aumentaciones de cada una)
    vistos, unicas = set(), []
    for p in imgs:
        base = p.name.split(".rf.")[0]
        if base not in vistos:
            vistos.add(base)
            unicas.append(p)
    rng.shuffle(unicas)
    out = []
    for p in unicas:
        rec = recortar(p, p.parent.parent / "labels" / (p.stem + ".txt"))
        out.extend(rec[:1])
        if len(out) >= n:
            break
    return out


def phash_diedrico(img: Image.Image) -> list[int]:
    """pHash de las 8 orientaciones: detecta copias de la misma foto aunque estén giradas."""
    out = []
    for k in range(4):
        r = img.rotate(90 * k, expand=True)
        out += [phash(r), phash(r.transpose(Image.FLIP_LEFT_RIGHT))]
    return out


def desconocidos_ampliados(excluir: np.ndarray, rng: np.random.RandomState, maximo: int) -> list[Image.Image]:
    """Más tatuajes desconocidos para que el semáforo pueda calibrarse por grupos.

    * DeMSI: todos los tatuajes anotados de cada foto (antes, solo el primero).
    * tattoodet y playground: solo fotos que NO siguen el esquema de versiones
      (esas son las identidades enroladas), una por foto de origen.
    Se descartan las copias de la misma foto (pHash en las 8 orientaciones) de
    cualquier imagen de entrenamiento o de la galería. Es un filtro por foto, no por
    parecido: no elimina a propósito los desconocidos difíciles.
    """
    out = []
    base = paths.DATASETS / "demsi_eval"
    for img in sorted((base / "images").rglob("*.jpg")):
        lbl = base / "labels" / img.relative_to(base / "images").with_suffix(".txt")
        out.extend(recortar(img, lbl))
    vistos = set()
    for fuente in ("tattoodet", "playground_tattootest"):
        for img in sorted((paths.RAW / fuente).glob("*/images/*")):
            b = img.name.split(".rf.")[0]
            if NOMBRE.match(img.name) or b in vistos:
                continue
            vistos.add(b)
            rec = recortar(img, img.parent.parent / "labels" / (img.stem + ".txt"))
            out.extend(rec[:2])
    rng.shuffle(out)
    limpio = []
    for im in out:
        if any(vista_en(h, excluir, umbral=6) for h in phash_diedrico(im)):
            continue
        limpio.append(im)
        if len(limpio) >= maximo:
            break
    return limpio


def recortes_demsi(n: int, rng: np.random.RandomState) -> list[Image.Image]:
    base = paths.DATASETS / "demsi_eval"
    imgs = sorted((base / "images").rglob("*.jpg")) + sorted((base / "images").rglob("*.png"))
    rng.shuffle(imgs)
    out = []
    for p in imgs:
        lbl = base / "labels" / p.relative_to(base / "images").with_suffix(".txt")
        out.extend(recortar(p, lbl)[:1])
        if len(out) >= n:
            break
    return out


# ----------------------------------------------------------------- calidad
def calidad(img: Image.Image) -> np.ndarray:
    """Nitidez (varianza del laplaciano), contraste (desv. de gris) y lado menor."""
    g = img.convert("L")
    lado = min(g.size)
    g224 = g.resize((224, 224), Image.BICUBIC)
    lap = np.asarray(g224.filter(ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], 1, 128)),
                     dtype="float32") - 128
    return np.array([lap.var(), np.asarray(g224, dtype="float32").std(), lado], dtype="float32")


# ----------------------------------------------------------------- puntajes
def puntajes(S: np.ndarray, vista_id: np.ndarray, n_id: int) -> np.ndarray:
    """Similitud de cada consulta contra cada identidad = máximo sobre sus vistas."""
    out = np.full((S.shape[0], n_id), -1.0, dtype="float32")
    for k in range(n_id):
        out[:, k] = S[:, vista_id == k].max(1)
    return out


def tpir_a_fpir(gen_top: np.ndarray, gen_ok: np.ndarray, imp_top: np.ndarray, fpir: float) -> float:
    t = np.quantile(imp_top, 1 - fpir)
    return float(((gen_top > t) & gen_ok).mean())


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    r = np.argsort(np.argsort(np.concatenate([pos, neg])))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))


def resumen(G: np.ndarray, verdad: np.ndarray, I: np.ndarray) -> dict:
    top = G.argmax(1)
    gen_top = G.max(1)
    ok = top == verdad
    rank = (G > G[np.arange(len(verdad)), verdad][:, None]).sum(1)
    imp_top = I.max(1)
    return {"top1": float(ok.mean()), "top5": float((rank < 5).mean()),
            "auc_abierto": auc(gen_top[ok], imp_top) if ok.any() else 0.0,
            "tpir_fpir1": tpir_a_fpir(gen_top, ok, imp_top, 0.01),
            "tpir_fpir5": tpir_a_fpir(gen_top, ok, imp_top, 0.05),
            "genuino_medio": float(gen_top[ok].mean()) if ok.any() else 0.0,
            "impostor_medio": float(imp_top.mean())}


# ----------------------------------------------------------------- fusión aprendida
TOPK = 5


def rasgos(Ga: np.ndarray, Gb: np.ndarray, Qz: np.ndarray):
    """Rasgos por (consulta, candidato) con las opiniones de los dos modelos.

    Candidatos = unión de los 5 primeros de cada modelo. Por candidato: los dos
    puntajes, su posición en cada ranking, el margen de cada modelo sobre su propio
    segundo, si ambos modelos lo ponen primero, y la calidad de la foto.
    """
    filas, q_idx, k_idx = [], [], []
    ra = np.argsort(-Ga, 1)
    rb = np.argsort(-Gb, 1)
    pos_a = np.argsort(ra, 1)
    pos_b = np.argsort(rb, 1)
    for q in range(len(Ga)):
        sa, sb = Ga[q], Gb[q]
        ma = sa[ra[q, 0]] - sa[ra[q, 1]]
        mb = sb[rb[q, 0]] - sb[rb[q, 1]]
        cands = list(dict.fromkeys(list(ra[q, :TOPK]) + list(rb[q, :TOPK])))
        for k in cands:
            filas.append([sa[k], sb[k], 1 / (1 + pos_a[q, k]), 1 / (1 + pos_b[q, k]),
                          sa[k] - sa[ra[q, 0]], sb[k] - sb[rb[q, 0]], ma, mb,
                          float(ra[q, 0] == k and rb[q, 0] == k), *Qz[q]])
            q_idx.append(q)
            k_idx.append(k)
    return np.array(filas, dtype="float32"), np.array(q_idx), np.array(k_idx)


def logistica(X, y, epocas=400, l2=1e-3):
    import torch
    Xt, yt = torch.tensor(X), torch.tensor(y, dtype=torch.float32)
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
    Xt = (Xt - mu) / sd
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    pos = max(1.0, float((yt == 0).sum() / max(1, (yt == 1).sum())))
    opt = torch.optim.LBFGS([w, b], max_iter=epocas)

    def cierre():
        opt.zero_grad()
        z = Xt @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            z, yt, pos_weight=torch.tensor(pos)) + l2 * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(cierre)
    return lambda Z: torch.sigmoid(((torch.tensor(Z) - mu) / sd) @ w.detach() + b.detach()).numpy()


def fusion_aprendida(Ga, Gb, Ia, Ib, verdad, grupo, Qg, Qi, semilla=0):
    """Fusión por apilamiento (stacking), validada fuera de muestra en 2 mitades por
    tatuaje: el modelo que puntúa una consulta nunca la vio al entrenarse."""
    def zq(Q):
        return np.log1p(np.maximum(Q, 0))
    Xg, qg, kg = rasgos(Ga, Gb, zq(Qg))
    Xi, qi, ki = rasgos(Ia, Ib, zq(Qi))
    yg = (kg == verdad[qg]).astype("float32")
    Fg = np.zeros_like(Ga) - 1
    Fi = np.zeros_like(Ia) - 1
    rng = np.random.RandomState(semilla)
    ids = np.unique(grupo)
    mitad = set(rng.choice(ids, len(ids) // 2, replace=False))
    en_a_g = np.array([grupo[q] in mitad for q in qg])
    imp_a = rng.rand(len(Ia)) < 0.5
    en_a_i = imp_a[qi]
    for entrena_g, entrena_i in ((en_a_g, en_a_i), (~en_a_g, ~en_a_i)):
        X = np.concatenate([Xg[entrena_g], Xi[entrena_i]])
        y = np.concatenate([yg[entrena_g], np.zeros(entrena_i.sum(), dtype="float32")])
        f = logistica(X, y)
        pg, pi = f(Xg[~entrena_g]), f(Xi[~entrena_i])
        Fg[qg[~entrena_g], kg[~entrena_g]] = pg
        Fi[qi[~entrena_i], ki[~entrena_i]] = pi
    return Fg, Fi


# ----------------------------------------------------------------- conformal
def cuantil_conformal(x: np.ndarray, nivel: float) -> float:
    n = len(x)
    k = min(n, math.ceil((n + 1) * nivel))
    return float(np.sort(x)[k - 1])


def semaforo(G: np.ndarray, tau: float, c: float):
    """Verde = un candidato, amarillo = varios, rojo = ninguno."""
    lim = max(tau, c)
    n = (G >= lim).sum(1)
    return np.where(n == 0, 0, np.where(n == 1, 1, 2)), G >= lim


def semaforo_fijo(G: np.ndarray, alto=0.70, bajo=0.55):
    t = G.max(1)
    return np.where(t >= alto, 1, np.where(t >= bajo, 2, 0))


def evaluar_semaforos(G, verdad, grupo, I, alfa, beta, reps, rng):
    ids = np.unique(grupo)
    filas = []
    for _ in range(reps):
        cal_ids = set(rng.choice(ids, len(ids) // 2, replace=False))
        gc = np.array([g in cal_ids for g in grupo])
        perm = rng.permutation(len(I))
        ic, it = perm[: len(I) // 2], perm[len(I) // 2:]
        tau = cuantil_conformal(I[ic].max(1), 1 - alfa)
        nc = 1 - G[gc][np.arange(gc.sum()), verdad[gc]]
        c = 1 - cuantil_conformal(nc, 1 - beta)
        col_g, cand = semaforo(G[~gc], tau, c)
        col_i, _ = semaforo(I[it], tau, c)
        v = verdad[~gc]
        top_ok = G[~gc].argmax(1) == v
        contiene = cand[np.arange(len(v)), v]
        filas.append({"tau": tau, "far": float((col_i > 0).mean()),
                      "verde_ok": float(((col_g == 1) & top_ok).mean()),
                      "verde_mal": float(((col_g == 1) & ~top_ok).mean()),
                      "amarillo": float((col_g == 2).mean()),
                      "amarillo_contiene": float(((col_g == 2) & contiene).mean()),
                      "rojo": float((col_g == 0).mean()),
                      "candidatos_amarillo": float(cand[col_g == 2].sum(1).mean()) if (col_g == 2).any() else 0.0})
    return {k: float(np.mean([f[k] for f in filas])) for k in filas[0]} | {
        "far_p95": float(np.quantile([f["far"] for f in filas], 0.95))}


def fijo(G, verdad, I):
    cg, ci = semaforo_fijo(G), semaforo_fijo(I)
    ok = G.argmax(1) == verdad
    return {"far": float((ci > 0).mean()), "far_verde": float((ci == 1).mean()),
            "verde_ok": float(((cg == 1) & ok).mean()), "verde_mal": float(((cg == 1) & ~ok).mean()),
            "amarillo": float((cg == 2).mean()), "rojo": float((cg == 0).mean())}


# ----------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="Ideas A y B con tatuajes reales")
    ap.add_argument("--min-versiones", type=int, default=3)
    ap.add_argument("--distractores", type=int, default=300)
    ap.add_argument("--impostores", type=int, default=600)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--modelos", nargs="*", default=None,
                    help="nombre=spec adicionales, p. ej. cabeza_v3=dinov2:head=models/embed_head_v3.pt")
    ap.add_argument("--etiqueta", default=None, help="sufijo de la carpeta de resultados")
    ap.add_argument("--desconocidos-ampliados", type=int, default=0,
                    help="usa N desconocidos de DeMSI completo + tattoodet/playground en vez de 600 de DeMSI")
    ap.add_argument("--rotaciones", action="store_true",
                    help="enrola además la original girada 90/180/270 grados")
    args = ap.parse_args()
    rng = np.random.RandomState(args.semilla)
    global SALIDA
    if args.rotaciones:
        SALIDA = SALIDA.parent / "ab_real_rot"
    if args.etiqueta:
        SALIDA = SALIDA.parent / f"{SALIDA.name}_{args.etiqueta}"
    modelos = dict(MODELOS)
    for m in args.modelos or []:
        k, v = m.split("=", 1)
        modelos[k] = v
    SALIDA.mkdir(parents=True, exist_ok=True)

    print("Cargando datos…", flush=True)
    train_h = hashes_entrenamiento()
    idents = identidades(args.min_versiones, train_h)
    distr = recortes_de(FUENTES_DISTRACTOR, args.distractores, rng)
    if args.desconocidos_ampliados:
        # recortes de entrenamiento y de la galería (las orientaciones se prueban del lado del desconocido)
        cache = SALIDA.parent / "phash_recortes_train.npy"
        if cache.exists():
            tr = np.load(cache)
        else:
            tr = np.array([phash(Image.open(f).convert("RGB")) for f in sorted(REAL_CROPS_TRAIN.glob("*.png"))],
                          dtype=np.uint64)
            np.save(cache, tr)
        gal = [i["enrol"] for i in idents] + [im for i in idents for _, im in i["consultas"]] + distr
        gal_h = np.array([phash(im) for im in gal], dtype=np.uint64)
        impost = desconocidos_ampliados(np.concatenate([tr, gal_h]), rng, args.desconocidos_ampliados)
    else:
        impost = recortes_demsi(args.impostores, rng)
    print(f"  identidades: {len(idents)} ({sum(i['en_train'] for i in idents)} con alguna versión en train) · "
          f"consultas: {sum(len(i['consultas']) for i in idents)} · distractores: {len(distr)} · "
          f"impostores: {len(impost)}", flush=True)

    # galería: cada identidad (y cada distractor) se enrola como en el servicio
    erng = np.random.RandomState(7)
    gal_imgs, gal_id = [], []
    for k, im in enumerate([i["enrol"] for i in idents] + distr):
        vistas = [im] + [degrade(im, lv, erng) for lv in ENROLL_LEVELS]
        if args.rotaciones:
            vistas += [im.rotate(a, expand=True) for a in (90, 180, 270)]
        for v in vistas:
            gal_imgs.append(v)
            gal_id.append(k)
    gal_id = np.array(gal_id)
    n_id = len(idents) + len(distr)
    q_imgs, verdad, grupo, var = [], [], [], []
    for k, i in enumerate(idents):
        for v, im in i["consultas"]:
            q_imgs.append(im)
            verdad.append(k)
            grupo.append(k)
            var.append(re.sub(r"\d+$", "", v))
    verdad, grupo, var = np.array(verdad), np.array(grupo), np.array(var)
    en_train = np.array([idents[k]["en_train"] for k in verdad])
    Q = np.stack([calidad(im) for im in q_imgs])
    QI = np.stack([calidad(im) for im in impost])

    from src.embedding.backbones import resolver
    sims = {}
    for nombre, spec in modelos.items():
        print(f"Embeddings: {nombre}", flush=True)
        emb = resolver(spec)
        eg, eq, ei = emb.embed(gal_imgs), emb.embed(q_imgs), emb.embed(impost)
        sims[nombre] = (puntajes(eq @ eg.T, gal_id, n_id), puntajes(ei @ eg.T, gal_id, n_id))
        del emb
        import torch
        torch.cuda.empty_cache()

    res = {"datos": {"identidades": len(idents), "consultas": int(len(verdad)), "distractores": len(distr),
                     "impostores": len(impost), "galeria_identidades": n_id,
                     "identidades_vistas_en_train": int(sum(i["en_train"] for i in idents)),
                     "consultas_por_variante": {v: int((var == v).sum()) for v in np.unique(var)}},
           "modelos": {}, "fusion": {}, "conformal": {}}

    for nombre, (G, I) in sims.items():
        r = resumen(G, verdad, I)
        r["por_variante"] = {v: float((G[var == v].argmax(1) == verdad[var == v]).mean()) for v in np.unique(var)}
        r["sin_fuga"] = resumen(G[~en_train], verdad[~en_train], I) if (~en_train).any() else None
        res["modelos"][nombre] = r

    # ---------------- idea A: ensamble fijo y adaptativo
    ws = np.round(np.linspace(0, 1, 11), 2)
    qscore = np.argsort(np.argsort(Q, 0), 0).mean(1) / len(Q)          # calidad combinada (rango)
    dinos = [k for k in sims if k.startswith("dinov2") or k.startswith("cabeza")]
    cnns = [k for k in sims if k.startswith("cnn")]
    pares = [(a, b) for a in dinos for b in cnns]
    res["fusion_aprendida"] = {}
    for a, b in pares:
        Fg, Fi = fusion_aprendida(*sims[a], *sims[b], verdad, grupo, Q, QI)
        res["fusion_aprendida"][f"{a}+{b}"] = resumen(Fg, verdad, Fi)
        sims[f"apilado:{a}+{b}"] = (Fg, Fi)
    for a, b in pares:
        (Ga, Ia), (Gb, Ib) = sims[a], sims[b]
        barrido = {}
        for w in ws:
            barrido[str(w)] = resumen(w * Ga + (1 - w) * Gb, verdad, w * Ia + (1 - w) * Ib)
        # validación cruzada por identidades: elegir w (fijo) y w por tercil de calidad en la mitad A, medir en B
        ids = np.unique(grupo)
        cv_fijo, cv_adapt, w_adapt_hist = [], [], []
        for rep in range(20):
            r2 = np.random.RandomState(100 + rep)
            cal = set(r2.choice(ids, len(ids) // 2, replace=False))
            mc = np.array([g in cal for g in grupo])
            cortes = np.quantile(qscore[mc], [1 / 3, 2 / 3])
            bins = np.digitize(qscore, cortes)
            acc = lambda w, m: ((w * Ga[m] + (1 - w) * Gb[m]).argmax(1) == verdad[m]).mean()
            wf = max(ws, key=lambda w: (acc(w, mc), -abs(w - 0.5)))
            wb = [max(ws, key=lambda w: (acc(w, mc & (bins == k)), -abs(w - wf))) for k in range(3)]
            w_adapt_hist.append(wb)
            mt = ~mc
            wq = np.array([wb[k] for k in bins])[:, None]
            cv_fijo.append(acc(wf, mt))
            cv_adapt.append((((wq * Ga + (1 - wq) * Gb)[mt]).argmax(1) == verdad[mt]).mean())
        res["fusion"][f"{a}+{b}"] = {
            "barrido_w_peso_de_" + a: barrido,
            "cv_top1_w_fijo": float(np.mean(cv_fijo)), "cv_top1_adaptativo": float(np.mean(cv_adapt)),
            "cv_top1_solo_a": float((Ga.argmax(1) == verdad).mean()),
            "cv_top1_solo_b": float((Gb.argmax(1) == verdad).mean()),
            "w_por_tercil_calidad_media": np.mean(w_adapt_hist, 0).round(2).tolist()}

    # ---------------- idea B: conformal frente a umbrales fijos
    mejores = {"dinov2_cabeza (producción)": sims["dinov2_cabeza"]}
    fa, fb = "dinov2_cabeza", "cnn_imagenet"
    wbest = float(max(ws, key=lambda w: res["fusion"][f"{fa}+{fb}"]["barrido_w_peso_de_" + fa][str(w)]["top1"]))
    mejores[f"ensamble w={wbest}"] = (wbest * sims[fa][0] + (1 - wbest) * sims[fb][0],
                                      wbest * sims[fa][1] + (1 - wbest) * sims[fb][1])
    fa2 = "dinov2_sin_cabeza"
    b2 = res["fusion"][f"{fa2}+{fb}"]["barrido_w_peso_de_" + fa2]
    w2 = float(max(ws, key=lambda w: b2[str(w)]["tpir_fpir1"]))
    mejores["dinov2_sin_cabeza"] = sims[fa2]
    mejores[f"ensamble sin cabeza w={w2}"] = (w2 * sims[fa2][0] + (1 - w2) * sims[fb][0],
                                             w2 * sims[fa2][1] + (1 - w2) * sims[fb][1])
    # el mejor de cada familia nueva (fijo y apilado), elegido por reconocimiento al 1 %
    for clave, v in res["fusion"].items():
        a, b = clave.split("+")
        if a in MODELOS and b in MODELOS:
            continue
        bw = v["barrido_w_peso_de_" + a]
        w3 = float(max(ws, key=lambda w: bw[str(w)]["tpir_fpir1"]))
        mejores[f"fijo {clave} w={w3}"] = (w3 * sims[a][0] + (1 - w3) * sims[b][0],
                                          w3 * sims[a][1] + (1 - w3) * sims[b][1])
    mejor_ap = max(res["fusion_aprendida"], key=lambda k: res["fusion_aprendida"][k]["tpir_fpir1"])
    mejores[f"apilado {mejor_ap}"] = sims[f"apilado:{mejor_ap}"]
    for k in modelos:
        if k not in MODELOS:
            mejores[k] = sims[k]
    res["tamano_galeria"] = {}
    for nombre, (G, I) in mejores.items():
        filas = {}
        for nd in (0, 50, 100, 200, len(distr)):
            cols = len(idents) + nd
            Gs, Is = G[:, :cols], I[:, :cols]
            f = fijo(Gs, verdad, Is)
            cf = evaluar_semaforos(Gs, verdad, grupo, Is, 0.01, 0.05, 50, np.random.RandomState(2))
            filas[str(cols)] = {"fijo_far": f["far"], "fijo_far_verde": f["far_verde"],
                                "conformal_far": cf["far"], "conformal_tau": cf["tau"],
                                "conformal_verde_ok": cf["verde_ok"], "conformal_rojo": cf["rojo"]}
        res["tamano_galeria"][nombre] = filas
    for nombre, (G, I) in mejores.items():
        res["conformal"][nombre] = {"fijo_0.55_0.70": fijo(G, verdad, I)}
        for alfa in (0.01, 0.05):
            res["conformal"][nombre][f"alfa={alfa}"] = evaluar_semaforos(
                G, verdad, grupo, I, alfa, 0.05, args.reps, np.random.RandomState(1))

    # distribuciones para gráficas
    np.savez_compressed(SALIDA / "puntajes.npz", verdad=verdad, var=var, en_train=en_train, Q=Q, QI=QI,
                        **{f"G_{k}": v[0] for k, v in sims.items() if ":" not in k},
                        **{f"I_{k}": v[1] for k, v in sims.items() if ":" not in k})
    (SALIDA / "resultados.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("datos",)}, indent=2, ensure_ascii=False))
    for k, v in res["fusion_aprendida"].items():
        print(f"apilado {k:35s} top1={v['top1']:.3f} auc={v['auc_abierto']:.3f} "
              f"tpir@1%={v['tpir_fpir1']:.3f} tpir@5%={v['tpir_fpir5']:.3f}")
    for k, v in res["modelos"].items():
        print(f"{k:20s} top1={v['top1']:.3f} top5={v['top5']:.3f} auc={v['auc_abierto']:.3f} "
              f"tpir@1%={v['tpir_fpir1']:.3f} tpir@5%={v['tpir_fpir5']:.3f}")
    for k, v in res["fusion"].items():
        print(f"{k:35s} solo_a={v['cv_top1_solo_a']:.3f} solo_b={v['cv_top1_solo_b']:.3f} "
              f"fijo={v['cv_top1_w_fijo']:.3f} adapt={v['cv_top1_adaptativo']:.3f} w={v['w_por_tercil_calidad_media']}")
    print(json.dumps(res["conformal"], indent=2, ensure_ascii=False))
    print(json.dumps(res["tamano_galeria"], indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
