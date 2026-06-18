"""Generación de dataset sintético de tatuajes — v7 (para la etapa de EMBEDDINGS).

Cambios clave respecto a v6 (enfocado a identificación, no detección):
  1. RECORTE AUTOMÁTICO del tatuaje: se proyecta la caja del decal a la cámara y
     se usa el RENDER POR REGIÓN de Blender (use_border + crop_to_border) → la
     salida ya es el recorte del tatuaje (sin PIL ni post-proceso).
  2. SALIDA por identidad: OUTPUT/crops/<tattoo_id>/<id>__<zona>__<piel>__<angulo>.png
     (estructura lista para `python -m src.embedding.build_gallery --sources OUTPUT/crops`).
  3. MANIFEST: OUTPUT/manifest.csv con (crop, identidad, categoría, zona, piel, ángulo, cámara).
  4. VISIBILIDAD: salta combinaciones donde el tatuaje no se ve (mira al lado
     opuesto de la cámara o queda fuera de cuadro) → evita recortes de piel vacía.
  5. DOMAIN RANDOMIZATION (opcional): jitter de cámara y de luces para acercar al
     "look" de foto real (clave para que el embedding sintético matchee con reales).

Correr DENTRO de Blender (no en el venv del repo):
    blender escena.blend --background --python blender/generate_dataset_v7.py
o pegar en el editor de scripts de Blender y ejecutar.

Nota: este script usa la API de Blender (bpy). No se puede probar fuera de
Blender; verifícalo con MODO TEST (1 tatuaje/categoría) antes de producción.
"""

import bpy
import csv
import os
import random
from math import radians, sin, cos
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view


# ========================================
# CONFIGURACIÓN GENERAL
# ========================================

BASE_PATH = r"C:/Users/Diego/Documents/Blender projects/Assets/Texturas_tatuajes/IMAGENES TEXTURA DATASET TATUAJES"
OUTPUT_PATH = r"C:/Users/Diego/Documents/Blender projects/Assets/Texturas_tatuajes/OUTPUT"
CROPS_PATH = os.path.join(OUTPUT_PATH, "crops")
MANIFEST_PATH = os.path.join(OUTPUT_PATH, "manifest.csv")

CATEGORIES = {
    "horizontal": "Tatuajes_Horizontal",
    "vertical":   "Tatuajes_Vertical",
    "compacto":   "Tatuajes_Compacto",
}

# Margen alrededor del tatuaje al recortar (fracción del tamaño de la caja).
CROP_MARGIN = 0.12
# Si el tatuaje ocupa menos de esto en pantalla, se descarta (queda muy chico).
MIN_BOX_FRAC = 0.04
# Umbral de "mira a la cámara": visible si normal·(cam→decal) < FACING_THR.
FACING_THR = -0.10
VISIBILITY_CHECK = True

# Domain randomization
DOMAIN_RANDOMIZATION = True
CAM_AZIMUTH_JITTER = 8.0     # grados
CAM_ELEV_JITTER = 6.0        # grados
CAM_DIST_JITTER = 0.06       # fracción de la distancia
LIGHT_ENERGY_JITTER = 0.35   # ±35% de energía
RANDOM_SEED = 42


# ---- (presets tomados de v6: ya estaban afinados) ----
SCALE_PRESETS = {
    "horizontal": {
        "pecho": (1.0, 1.0, 1.0), "espalda": (1.15, 1.0, 1.15),
        "bicep_izq": (0.7, 1.0, 1.4), "bicep_der": (0.7, 1.0, 1.4),
        "cintura_izq": (0.9, 1.0, 0.8), "cintura_der": (0.9, 1.0, 0.8),
    },
    "compacto": {
        "pecho": (0.7, 1.0, 1.4), "espalda": (0.75, 1.0, 1.50),
        "bicep_izq": (0.5, 1.0, 2.2), "bicep_der": (0.5, 1.0, 2.2),
        "cintura_izq": (0.6, 1.0, 1.2), "cintura_der": (0.6, 1.0, 1.2),
    },
    "vertical": {
        "pecho": (0.5, 1.0, 2.4), "espalda": (0.55, 1.0, 2.65),
        "bicep_izq": (1.15, 1.0, 2.0), "bicep_der": (1.15, 1.0, 2.0),
        "cintura_izq": (0.4, 1.0, 1.8), "cintura_der": (0.4, 1.0, 1.8),
    },
}

MAPPING_CONFIG = {
    "horizontal": {
        "default": {"location": (0.25, 0.25), "scale": (0.5, 0.5), "rotation": 0},
        "espalda": {"location": (0.75, 0.25), "scale": (-0.5, 0.5), "rotation": 0},
    },
    "compacto": {
        "default": {"location": (0, 0), "scale": (1, 1), "rotation": 0},
        "espalda": {"location": (1, 0), "scale": (-1, 1), "rotation": 0},
    },
    "vertical": {
        "default": {"location": (0, 0), "scale": (1, 1), "rotation": 0},
        "bicep_izq": {"location": (1, 0), "scale": (1, 1), "rotation": 90},
        "bicep_der": {"location": (1, 0), "scale": (1, 1), "rotation": 90},
        "espalda": {"location": (1, 0), "scale": (-1, 1), "rotation": 0},
    },
}

ZONES = {
    "pecho": ("TattooDecal_Pecho", False),
    "espalda": ("TattooDecal_Espalda", False),
    "bicep_izq": ("TattooDecal_Bicep1", False),
    "bicep_der": ("TattooDecal_Bicep1", True),
    "cintura_izq": ("TattooDecal_Cintura1", False),
    "cintura_der": ("TattooDecal_Cintura1", True),
}

ZONE_CAMERA_FROM = {
    "pecho": (0, -1, 0), "espalda": (0, 1, 0),
    "bicep_izq": (0, -1, 0), "bicep_der": (0, -1, 0),
    "cintura_izq": (-1, 0.15, 0), "cintura_der": (1, 0.15, 0),
}

CAMERA_DISTANCE = 0.4
ZONE_CAMERA_DISTANCE = {
    "pecho": 0.4, "espalda": 0.4, "bicep_izq": 0.4, "bicep_der": 0.4,
    "cintura_izq": 0.55, "cintura_der": 0.55,
}
ZONE_TARGET_OFFSET = {
    "pecho": (0, 0, 0), "espalda": (0, 0, 0), "bicep_izq": (0, 0, 0),
    "bicep_der": (0, 0, 0), "cintura_izq": (0, 0, 0.03), "cintura_der": (0, 0, 0.03),
}

SKIN_MATERIALS = {"light": "Skin_Light", "medium": "Skin_Medium", "dark": "Skin_Dark"}
CAMERA_ANGLES = {
    "frontal": (0, 0), "lateral_izq": (-45, 0), "lateral_der": (45, 0),
    "picado": (0, 30), "contrapicado": (0, -25),
}

# MODO TEST / PRODUCCIÓN
MAX_TATTOOS_PER_CATEGORY = 1
ONLY_SKINS = ["medium"]
ONLY_ANGLES = ["frontal"]
# Producción: MAX_TATTOOS_PER_CATEGORY = None; ONLY_SKINS = None; ONLY_ANGLES = None

RENDER_RES = 1024
MANNEQUIN = "Ch36"
CAMERA = "Cámara"


# ========================================
# FUNCIONES BASE (de v6)
# ========================================

def cargar_tatuaje_en_todos_decals(image_path):
    if not os.path.exists(image_path):
        print(f"  ⚠️  Imagen no existe: {image_path}")
        return False
    img = bpy.data.images.load(image_path, check_existing=True)
    decal_names = list({d for d, _ in ZONES.values()})
    success = 0
    for decal_name in decal_names:
        decal = bpy.data.objects.get(decal_name)
        if not decal or not decal.material_slots:
            continue
        mat = decal.material_slots[0].material
        if not mat or not mat.node_tree:
            continue
        for node in mat.node_tree.nodes:
            if node.type == "TEX_IMAGE":
                node.image = img
                success += 1
                break
    return success > 0


def aplicar_mapping(decal, location, scale, rotation):
    if not decal.material_slots:
        return
    mat = decal.material_slots[0].material
    if not mat or not mat.node_tree:
        return
    for node in mat.node_tree.nodes:
        if node.type == "MAPPING":
            node.inputs["Location"].default_value[0] = location[0]
            node.inputs["Location"].default_value[1] = location[1]
            node.inputs["Location"].default_value[2] = 0
            node.inputs["Scale"].default_value[0] = scale[0]
            node.inputs["Scale"].default_value[1] = scale[1]
            node.inputs["Scale"].default_value[2] = 1
            node.inputs["Rotation"].default_value[0] = 0
            node.inputs["Rotation"].default_value[1] = 0
            node.inputs["Rotation"].default_value[2] = radians(rotation)
            return


def get_mapping_config(category, zone_name):
    cat_cfg = MAPPING_CONFIG[category]
    return cat_cfg.get(zone_name, cat_cfg["default"])


def cambiar_piel(skin_key):
    mat = bpy.data.materials.get(SKIN_MATERIALS[skin_key])
    body = bpy.data.objects.get(MANNEQUIN)
    if body and mat and body.material_slots:
        body.material_slots[0].material = mat


def esconder_todos_decals():
    seen = set()
    for decal_name, _ in ZONES.values():
        if decal_name in seen:
            continue
        seen.add(decal_name)
        decal = bpy.data.objects.get(decal_name)
        if decal:
            decal.hide_viewport = True
            decal.hide_render = True


def aplicar_zona(zone_name, category, originales):
    decal_name, mirror = ZONES[zone_name]
    decal = bpy.data.objects.get(decal_name)
    if not decal or decal_name not in originales:
        return None
    esconder_todos_decals()
    decal.hide_viewport = False
    decal.hide_render = False
    sx, sy, sz = SCALE_PRESETS[category][zone_name]
    decal.scale = (sx, sy, sz)
    orig = originales[decal_name]
    decal.location = orig["location"].copy()
    if mirror:
        decal.location.x = -orig["location"].x
    cfg = get_mapping_config(category, zone_name)
    aplicar_mapping(decal, cfg["location"], cfg["scale"], cfg["rotation"])
    bpy.context.view_layer.update()
    return decal


def posicionar_camara(target_pos, zone_name, azimuth, elevation, distance):
    base_dir = Vector(ZONE_CAMERA_FROM.get(zone_name, (0, -1, 0)))
    if base_dir.length == 0:
        base_dir = Vector((0, -1, 0))
    base_dir.normalize()
    az, el = radians(azimuth), radians(elevation)
    rotated = Vector((base_dir.x * cos(az) - base_dir.y * sin(az),
                      base_dir.x * sin(az) + base_dir.y * cos(az), 0))
    rotated = Vector((rotated.x * cos(el), rotated.y * cos(el), sin(el)))
    if rotated.length == 0:
        rotated = Vector((0, -1, 0))
    rotated.normalize()
    cam = bpy.data.objects.get(CAMERA)
    if not cam:
        return None
    cam.location = (target_pos.x + rotated.x * distance,
                    target_pos.y + rotated.y * distance,
                    target_pos.z + rotated.z * distance)
    direction = Vector(target_pos) - Vector(cam.location)
    if direction.length > 0:
        cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    return cam


def restaurar_decals(originales):
    for decal_name, orig in originales.items():
        decal = bpy.data.objects.get(decal_name)
        if not decal:
            continue
        decal.location = orig["location"].copy()
        decal.scale = orig["scale"].copy()
        decal.hide_viewport = False
        decal.hide_render = False
        aplicar_mapping(decal, (0, 0), (1, 1), 0)


def validar_escena():
    ok = True
    if not bpy.data.objects.get(MANNEQUIN):
        print(f"⛔ Falta maniquí: {MANNEQUIN}"); ok = False
    if not bpy.data.objects.get(CAMERA):
        print(f"⛔ Falta cámara: {CAMERA}"); ok = False
    for zone_name, (decal_name, _) in ZONES.items():
        if not bpy.data.objects.get(decal_name):
            print(f"⛔ Falta decal de zona {zone_name}: {decal_name}"); ok = False
    return ok


# ========================================
# NUEVO EN v7: caja del tatuaje, visibilidad, recorte, DR
# ========================================

def decal_screen_bbox(scene, cam, decal):
    """Proyecta las 8 esquinas del decal a la cámara. Devuelve (minx,miny,maxx,maxy)
    en coords normalizadas [0,1] (origen abajo-izq, convención de Blender) o None
    si queda completamente detrás/fuera de cuadro."""
    xs, ys, any_front = [], [], False
    for corner in decal.bound_box:
        world = decal.matrix_world @ Vector(corner)
        co = world_to_camera_view(scene, cam, world)
        if co.z > 0:
            any_front = True
        xs.append(co.x)
        ys.append(co.y)
    if not any_front:
        return None
    minx, maxx = max(min(xs), 0.0), min(max(xs), 1.0)
    miny, maxy = max(min(ys), 0.0), min(max(ys), 1.0)
    if maxx <= minx or maxy <= miny:
        return None
    return minx, miny, maxx, maxy


def tattoo_visible(cam, decal, bbox):
    """Heurística de visibilidad: el tatuaje debe mirar hacia la cámara y ocupar
    un tamaño mínimo en pantalla."""
    if bbox is None:
        return False
    minx, miny, maxx, maxy = bbox
    if (maxx - minx) < MIN_BOX_FRAC or (maxy - miny) < MIN_BOX_FRAC:
        return False
    if not VISIBILITY_CHECK:
        return True
    # normal del decal (su +Z local) vs dirección cámara→decal
    normal = (decal.matrix_world.to_3x3() @ Vector((0, 0, 1))).normalized()
    cam_to_decal = (decal.matrix_world.translation - cam.matrix_world.translation).normalized()
    # Si la normal apunta hacia la cámara, normal·(cam→decal) es negativo.
    return normal.dot(cam_to_decal) < FACING_THR


def aplicar_domain_randomization(rng):
    """Jitter de luces (energía) para variar la iluminación. Devuelve estado previo."""
    prev = {}
    if not DOMAIN_RANDOMIZATION:
        return prev
    for obj in bpy.data.objects:
        if obj.type == "LIGHT" and obj.data:
            prev[obj.name] = obj.data.energy
            factor = 1.0 + rng.uniform(-LIGHT_ENERGY_JITTER, LIGHT_ENERGY_JITTER)
            obj.data.energy = max(0.0, obj.data.energy * factor)
    return prev


def restaurar_luces(prev):
    for name, energy in prev.items():
        obj = bpy.data.objects.get(name)
        if obj and obj.data:
            obj.data.energy = energy


def render_crop(scene, filepath, bbox):
    """Renderiza SOLO la región del tatuaje (con margen) -> la salida ya es el recorte."""
    minx, miny, maxx, maxy = bbox
    mw = (maxx - minx) * CROP_MARGIN
    mh = (maxy - miny) * CROP_MARGIN
    r = scene.render
    r.resolution_x = RENDER_RES
    r.resolution_y = RENDER_RES
    r.image_settings.file_format = "PNG"
    r.use_border = True
    r.use_crop_to_border = True
    r.border_min_x = max(0.0, minx - mw)
    r.border_max_x = min(1.0, maxx + mw)
    r.border_min_y = max(0.0, miny - mh)
    r.border_max_y = min(1.0, maxy + mh)
    r.filepath = filepath
    bpy.ops.render.render(write_still=True)


# ========================================
# MAIN
# ========================================

def main():
    if not validar_escena():
        print("\n⛔ Revisa la escena antes de generar el dataset.")
        return

    os.makedirs(CROPS_PATH, exist_ok=True)
    rng = random.Random(RANDOM_SEED)
    scene = bpy.context.scene

    originales = {}
    for decal_name, _ in ZONES.values():
        if decal_name in originales:
            continue
        decal = bpy.data.objects.get(decal_name)
        if decal:
            originales[decal_name] = {"location": decal.location.copy(),
                                      "scale": decal.scale.copy()}

    manifest = open(MANIFEST_PATH, "w", newline="", encoding="utf-8")
    writer = csv.writer(manifest)
    writer.writerow(["crop_path", "tattoo_id", "categoria", "zona", "piel",
                     "angulo", "cam_az", "cam_el", "dist"])

    skins = ONLY_SKINS if ONLY_SKINS else list(SKIN_MATERIALS.keys())
    angles = ONLY_ANGLES if ONLY_ANGLES else list(CAMERA_ANGLES.keys())
    cam = bpy.data.objects.get(CAMERA)

    total = saltados = 0
    for category, folder in CATEGORIES.items():
        folder_path = os.path.join(BASE_PATH, folder)
        if not os.path.exists(folder_path):
            print(f"⚠️  Carpeta no existe: {folder_path}")
            continue
        pngs = sorted(f for f in os.listdir(folder_path) if f.lower().endswith(".png"))
        if MAX_TATTOOS_PER_CATEGORY:
            pngs = pngs[:MAX_TATTOOS_PER_CATEGORY]
        print(f"\n═══ {category.upper()}: {len(pngs)} tatuajes ═══")

        for png in pngs:
            tattoo_id = os.path.splitext(png)[0]
            if not cargar_tatuaje_en_todos_decals(os.path.join(folder_path, png)):
                continue
            crop_dir = os.path.join(CROPS_PATH, tattoo_id)
            os.makedirs(crop_dir, exist_ok=True)
            print(f"\n  ▶ {tattoo_id}")

            for zone_name in ZONES:
                decal = aplicar_zona(zone_name, category, originales)
                if not decal:
                    continue
                bpy.context.view_layer.update()
                target = decal.matrix_world.translation.copy() + Vector(
                    ZONE_TARGET_OFFSET.get(zone_name, (0, 0, 0)))

                for skin in skins:
                    cambiar_piel(skin)
                    for angle in angles:
                        az, el = CAMERA_ANGLES[angle]
                        dist = ZONE_CAMERA_DISTANCE.get(zone_name, CAMERA_DISTANCE)
                        if DOMAIN_RANDOMIZATION:
                            az += rng.uniform(-CAM_AZIMUTH_JITTER, CAM_AZIMUTH_JITTER)
                            el += rng.uniform(-CAM_ELEV_JITTER, CAM_ELEV_JITTER)
                            dist *= 1.0 + rng.uniform(-CAM_DIST_JITTER, CAM_DIST_JITTER)
                        posicionar_camara(target, zone_name, az, el, dist)
                        bpy.context.view_layer.update()

                        bbox = decal_screen_bbox(scene, cam, decal)
                        if not tattoo_visible(cam, decal, bbox):
                            saltados += 1
                            continue

                        prev_lights = aplicar_domain_randomization(rng)
                        out_file = os.path.join(
                            crop_dir, f"{tattoo_id}__{zone_name}__{skin}__{angle}.png")
                        render_crop(scene, out_file, bbox)
                        restaurar_luces(prev_lights)

                        writer.writerow([out_file, tattoo_id, category, zone_name,
                                         skin, angle, round(az, 2), round(el, 2),
                                         round(dist, 3)])
                        total += 1
                        print(f"    ✓ {zone_name} | {skin} | {angle}")

            restaurar_decals(originales)

    manifest.close()
    # restaurar render sin borde
    scene.render.use_border = False
    scene.render.use_crop_to_border = False
    esconder_todos_decals()
    print(f"\n═══ TERMINADO: {total} recortes, {saltados} saltados (no visibles) ═══")
    print(f"Crops:    {CROPS_PATH}")
    print(f"Manifest: {MANIFEST_PATH}")
    print("Galería:  python -m src.embedding.build_gallery --sources datasets/gang_tattoos OUTPUT/crops")


main()
