"""Build web/public/anatomy/heart.glb from BodyParts3D STL parts (CC BY-SA 2.1 JP).

Decimates the heart wall and the coronary vessels, recentres everything on the
heart, converts millimetres to decimetres and writes one GLB with named nodes.
"""
from pathlib import Path
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "anatomy" / "stl"
OUT = ROOT / "web" / "public" / "anatomy" / "heart.glb"

# name in the GLB -> (FMA ids, target face count)
PARTS = {
    "heart_wall": (["FMA7274"], 70000),
    "aorta": (["FMA3736", "FMA3768"], 6000),
    "svc": (["FMA4720"], 2000),
    "coronary_sinus": (["FMA4706", "FMA76751"], 5000),
    "left_main": (["FMA4685"], 600),
    "LAD": (["FMA3862nsn", "FMA71670"], 9000),
    "LCX": (["FMA3895"], 4000),
    "RCA": (["FMA3802", "FMA3818", "FMA3840nsn", "FMA76994"], 12000),
}


def load(ids):
    ms = [trimesh.load(SRC / f"{i}.stl", force="mesh") for i in ids]
    m = trimesh.util.concatenate(ms)
    m.merge_vertices()
    return m


def main():
    meshes = {k: load(v[0]) for k, v in PARTS.items()}
    centre = meshes["heart_wall"].bounds.mean(axis=0)
    scene = trimesh.Scene()
    for name, (ids, faces) in PARTS.items():
        m = meshes[name]
        if len(m.faces) > faces:
            m = m.simplify_quadric_decimation(face_count=faces)
        m.apply_translation(-centre)
        m.apply_scale(0.01)  # mm -> 1 unit = 100 mm
        # BodyParts3D: x = patient left, y = posterior?, z = superior. Convert to y-up.
        rot = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        m.apply_transform(rot)
        m.fix_normals()
        print(f"{name:15s} faces={len(m.faces):6d} bounds={np.round(m.bounds, 2).tolist()}")
        scene.add_geometry(m, node_name=name, geom_name=name)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    scene.export(OUT)
    print("wrote", OUT, OUT.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
