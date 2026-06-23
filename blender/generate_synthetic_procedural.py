"""Generación sintética PROCEDURAL de vistas de tatuajes — autónomo.

Construye TODA la escena desde cero (no necesitas modelar nada):
  - una superficie de "piel" curva (plano subdividido + curvatura),
  - material de piel procedural con el tatuaje mezclado encima,
  - iluminación y cámara variadas por render.

Para cada PNG de tatuaje genera N vistas con variación (tono de piel, luz,
ángulo de cámara, escala/rotación/posición del tatuaje, opacidad), recortadas a
la zona de piel → ideales para ENRIQUECER LA GALERÍA de embeddings.

Salida (igual que v7, lista para build_gallery):
  OUTPUT/crops/<tattoo_id>/<id>__vNNN.png  +  OUTPUT/manifest.csv

Cómo correr (Windows, dentro de Blender):
  1. Edita INPUT_TATTOOS y OUTPUT_PATH abajo.
  2. Blender → pestaña "Scripting" → abre este archivo → Run Script (Alt+P).
  3. Window → Toggle System Console para ver el progreso.
O headless:  blender --background --python blender/generate_synthetic_procedural.py

No se puede probar fuera de Blender (usa bpy). Corre primero en MODO TEST.
"""

import bpy
import csv
import os
import glob
import random
from math import radians, sin, cos
from mathutils import Vector


# ========================= CONFIG =========================
# Carpeta con los PNG de los tatuajes (idealmente con fondo transparente).
INPUT_TATTOOS = r"C:/Users/Ricky/Documents/GitHub/rtattooai-ml/datasets/gang_tattoos"          # <-- EDITA (p.ej. tu datasets/gang_tattoos)
OUTPUT_PATH   = r"C:/Users/Ricky/Documents/GitHub/rtattooai-ml/datasets/output/gang_tattoos_out"                # <-- EDITA
CROPS_PATH    = os.path.join(OUTPUT_PATH, "crops")
MANIFEST_PATH = os.path.join(OUTPUT_PATH, "manifest.csv")

# Si tus tatuajes NO tienen transparencia (fondo blanco), pon True:
USE_LUMINANCE_MASK = False

N_PER_TATTOO = 16          # nº de vistas por tatuaje
RENDER_RES   = 512
SEED         = 42

# Modo test (rápido): limita nº de tatuajes y vistas. Pon None para producción.
MAX_TATTOOS   = 2
TEST_N        = 3          # vistas por tatuaje en modo test (si MAX_TATTOOS no es None)

# HDRIs opcionales para iluminación realista (carpeta con .hdr/.exr). Vacío = luces.
HDRI_FOLDER  = r""

# Paleta de tonos de piel (RGB lineal 0-1).
SKIN_COLORS = [
    (0.86, 0.68, 0.56), (0.78, 0.57, 0.44), (0.66, 0.46, 0.34),
    (0.52, 0.35, 0.25), (0.40, 0.26, 0.18), (0.90, 0.74, 0.62),
]

# Tamaño del plano de piel. GRANDE a propósito: así siempre llena el cuadro de
# piel y el tatuaje (centrado) nunca queda fuera de encuadre.
PLANE_SIZE = 4.0

# Rangos de variación
CAM_DIST = (3.0, 4.2)          # lo bastante lejos para que el plano llene el cuadro
CAM_AZIMUTH = (-55, 55)        # grados (más variación de ángulo)
CAM_ELEV = (-45, 45)
CAM_ROLL = (-15, 15)
TAT_SCALE = (0.30, 0.50)       # fracción del plano que ocupa el tatuaje (centrado, con margen)
TAT_ROT = (-25, 25)            # grados
TAT_OFFSET = 0.04              # casi centrado (evita que el tatuaje "se salga")
TAT_OPACITY = (0.55, 1.0)      # incluye algo de decoloración leve
LIGHT_ENERGY = (2.0, 7.0)
LIGHT_TEMP_TINT = 0.12         # variación cálido/frío
BEND_ANGLE = (-22, 22)         # curvatura de la piel (moderada, no pliega el tatuaje)


# ========================= ESCENA =========================
def clean_scene():
    """Borra todo para empezar de cero."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras):
        for block in list(coll):
            if block.users == 0:
                coll.remove(block)


def set_engine_eevee(scene):
    for eng in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            scene.render.engine = eng
            return
        except (TypeError, Exception):
            continue


def make_skin_plane():
    bpy.ops.mesh.primitive_plane_add(size=PLANE_SIZE, location=(0, 0, 0))
    plane = bpy.context.active_object
    plane.name = "SkinPatch"
    # subdividir para que la curvatura se vea suave
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.subdivide(number_cuts=40)
    bpy.ops.object.mode_set(mode="OBJECT")
    # modificador de curvatura (se actualiza por render)
    bend = plane.modifiers.new(name="Bend", type="SIMPLE_DEFORM")
    bend.deform_method = "BEND"
    bend.deform_axis = "X"
    bend.angle = 0.0
    bpy.ops.object.shade_smooth()
    return plane, bend


def make_skin_material(plane, tattoo_img):
    """Material: color de piel + tatuaje mezclado encima por su máscara."""
    mat = bpy.data.materials.new("SkinTattooMat")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = 0.55

    skin_rgb = nt.nodes.new("ShaderNodeRGB"); skin_rgb.name = "SkinRGB"
    skin_rgb.outputs[0].default_value = (0.8, 0.6, 0.5, 1)

    texcoord = nt.nodes.new("ShaderNodeTexCoord")
    # Sándwich restar/sumar 0.5 -> escala y rotación ocurren ALREDEDOR DEL CENTRO
    # del plano, así el tatuaje queda centrado sin importar la escala.
    sub = nt.nodes.new("ShaderNodeVectorMath"); sub.operation = "SUBTRACT"
    sub.inputs[1].default_value = (0.5, 0.5, 0.0)
    nt.links.new(texcoord.outputs["UV"], sub.inputs[0])
    mapping = nt.nodes.new("ShaderNodeMapping"); mapping.name = "TatMapping"
    nt.links.new(sub.outputs["Vector"], mapping.inputs["Vector"])
    add = nt.nodes.new("ShaderNodeVectorMath"); add.operation = "ADD"
    add.inputs[1].default_value = (0.5, 0.5, 0.0)
    nt.links.new(mapping.outputs["Vector"], add.inputs[0])

    tex = nt.nodes.new("ShaderNodeTexImage"); tex.name = "TatTex"
    tex.image = tattoo_img
    tex.extension = "CLIP"     # fuera del tatuaje = transparente, no se repite
    nt.links.new(add.outputs["Vector"], tex.inputs["Vector"])

    # máscara del tatuaje: alpha (o 1-luminancia si es fondo blanco)
    opacity = nt.nodes.new("ShaderNodeValue"); opacity.name = "TatOpacity"
    opacity.outputs[0].default_value = 1.0
    mult = nt.nodes.new("ShaderNodeMath"); mult.operation = "MULTIPLY"
    if USE_LUMINANCE_MASK:
        inv = nt.nodes.new("ShaderNodeInvert")
        lum = nt.nodes.new("ShaderNodeRGBToBW")
        nt.links.new(tex.outputs["Color"], lum.inputs["Color"])
        nt.links.new(lum.outputs["Val"], inv.inputs["Color"])
        nt.links.new(inv.outputs["Color"], mult.inputs[0])
    else:
        nt.links.new(tex.outputs["Alpha"], mult.inputs[0])
    nt.links.new(opacity.outputs[0], mult.inputs[1])

    mix = nt.nodes.new("ShaderNodeMix"); mix.data_type = "RGBA"
    nt.links.new(mult.outputs[0], mix.inputs["Factor"])
    nt.links.new(skin_rgb.outputs[0], mix.inputs["A"])
    nt.links.new(tex.outputs["Color"], mix.inputs["B"])
    nt.links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])

    plane.data.materials.clear()
    plane.data.materials.append(mat)
    return mat


def ensure_world():
    if bpy.context.scene.world is None:
        bpy.context.scene.world = bpy.data.worlds.new("World")
    w = bpy.context.scene.world
    w.use_nodes = True
    return w


def setup_lighting(world, hdris, rng):
    """HDRI si hay; si no, una luz Sun variada."""
    nt = world.node_tree
    nt.nodes.clear()
    bg = nt.nodes.new("ShaderNodeBackground")
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    if hdris:
        env = nt.nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(rng.choice(hdris), check_existing=True)
        nt.links.new(env.outputs["Color"], bg.inputs["Color"])
        bg.inputs["Strength"].default_value = rng.uniform(0.5, 1.4)
    else:
        bg.inputs["Strength"].default_value = rng.uniform(0.15, 0.4)


def make_sun():
    light_data = bpy.data.lights.new("Sun", type="SUN")
    sun = bpy.data.objects.new("Sun", light_data)
    bpy.context.collection.objects.link(sun)
    return sun


def make_camera(scene):
    cam_data = bpy.data.cameras.new("Cam")
    cam = bpy.data.objects.new("Cam", cam_data)
    bpy.context.collection.objects.link(cam)
    scene.camera = cam
    return cam


def place_camera(cam, rng):
    az = radians(rng.uniform(*CAM_AZIMUTH))
    el = radians(rng.uniform(*CAM_ELEV))
    dist = rng.uniform(*CAM_DIST)
    # base: mirando el plano (normal +Z) desde +Z
    d = Vector((sin(az) * cos(el), -sin(el), cos(az) * cos(el)))
    cam.location = d.normalized() * dist
    direction = -cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    cam.rotation_euler.rotate_axis("Z", radians(rng.uniform(*CAM_ROLL)))


def vary(mat, bend, sun, rng):
    """Aplica una variación aleatoria a piel/tatuaje/luz/curvatura."""
    nt = mat.node_tree
    skin = rng.choice(SKIN_COLORS)
    tint = rng.uniform(-LIGHT_TEMP_TINT, LIGHT_TEMP_TINT)
    nt.nodes["SkinRGB"].outputs[0].default_value = (
        min(max(skin[0] + tint, 0), 1), skin[1], min(max(skin[2] - tint, 0), 1), 1)
    nt.nodes["TatOpacity"].outputs[0].default_value = rng.uniform(*TAT_OPACITY)
    mp = nt.nodes["TatMapping"]
    # Escala del MAPEO = 1/fracción: el tatuaje ocupa esa fracción del plano,
    # centrado (por el sándwich). Location solo aplica un jitter pequeño.
    s = 1.0 / rng.uniform(*TAT_SCALE)
    mp.inputs["Scale"].default_value = (s, s, 1)
    mp.inputs["Rotation"].default_value[2] = radians(rng.uniform(*TAT_ROT))
    mp.inputs["Location"].default_value[0] = rng.uniform(-TAT_OFFSET, TAT_OFFSET)
    mp.inputs["Location"].default_value[1] = rng.uniform(-TAT_OFFSET, TAT_OFFSET)
    bend.angle = radians(rng.uniform(*BEND_ANGLE))
    if sun:
        sun.data.energy = rng.uniform(*LIGHT_ENERGY)
        sun.rotation_euler = (radians(rng.uniform(-50, 50)),
                              radians(rng.uniform(-50, 50)),
                              radians(rng.uniform(0, 360)))


def render(scene, filepath):
    scene.render.resolution_x = RENDER_RES
    scene.render.resolution_y = RENDER_RES
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)


# ========================= MAIN =========================
def main():
    rng = random.Random(SEED)
    scene = bpy.context.scene
    set_engine_eevee(scene)

    tattoos = sorted(glob.glob(os.path.join(INPUT_TATTOOS, "*.png")))
    if not tattoos:
        print(f"⛔ No hay PNG en INPUT_TATTOOS: {INPUT_TATTOOS}")
        return
    if MAX_TATTOOS:
        tattoos = tattoos[:MAX_TATTOOS]
    n_views = TEST_N if MAX_TATTOOS else N_PER_TATTOO

    hdris = []
    if HDRI_FOLDER:
        hdris = glob.glob(os.path.join(HDRI_FOLDER, "*.hdr")) + \
                glob.glob(os.path.join(HDRI_FOLDER, "*.exr"))

    os.makedirs(CROPS_PATH, exist_ok=True)
    clean_scene()
    plane, bend = make_skin_plane()
    world = ensure_world()
    sun = None if hdris else make_sun()
    cam = make_camera(scene)

    manifest = open(MANIFEST_PATH, "w", newline="", encoding="utf-8")
    writer = csv.writer(manifest)
    writer.writerow(["crop_path", "tattoo_id", "vista", "render"])

    total = 0
    print(f"═══ {len(tattoos)} tatuajes × {n_views} vistas ═══")
    for tpath in tattoos:
        tattoo_id = os.path.splitext(os.path.basename(tpath))[0].replace("-removebg-preview", "")
        img = bpy.data.images.load(tpath, check_existing=True)
        mat = make_skin_material(plane, img)
        crop_dir = os.path.join(CROPS_PATH, tattoo_id)
        os.makedirs(crop_dir, exist_ok=True)
        print(f"\n  ▶ {tattoo_id}")
        for v in range(n_views):
            setup_lighting(world, hdris, rng)
            vary(mat, bend, sun, rng)
            place_camera(cam, rng)
            bpy.context.view_layer.update()
            out_file = os.path.join(crop_dir, f"{tattoo_id}__v{v:03d}.png")
            render(scene, out_file)
            writer.writerow([out_file, tattoo_id, v, total])
            total += 1
            print(f"    ✓ vista {v+1}/{n_views}")
        # limpiar material previo para no acumular
        for m in list(bpy.data.materials):
            if m.users == 0:
                bpy.data.materials.remove(m)

    manifest.close()
    print(f"\n═══ TERMINADO: {total} recortes ═══")
    print(f"Crops:    {CROPS_PATH}")
    print(f"Manifest: {MANIFEST_PATH}")
    print("Galería:  python -m src.embedding.build_gallery --sources datasets/gang_tattoos OUTPUT/crops")


main()
