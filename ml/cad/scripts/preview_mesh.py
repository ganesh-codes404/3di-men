#!/usr/bin/env python3
"""python scripts/preview_mesh.py --mesh out.stl --out preview.png

Renders a static 3-quarter-view PNG of a mesh via matplotlib (no OpenGL/display
needed -- works over SSH, headless, etc). This is a debugging/demo convenience,
not part of the generation pipeline itself.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--elev", type=float, default=25)
    ap.add_argument("--azim", type=float, default=-55)
    args = ap.parse_args()

    mesh = trimesh.load(args.mesh, force="mesh")
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="3d")
    poly = Poly3DCollection(mesh.vertices[mesh.faces], facecolor="#c9d6e3", edgecolor="#3a3a3a", linewidths=0.15)
    ax.add_collection3d(poly)

    bounds = mesh.bounds
    center = bounds.mean(axis=0)
    radius = (bounds[1] - bounds[0]).max() / 2
    for axis, c in zip("xyz", center):
        getattr(ax, f"set_{axis}lim")(c - radius, c + radius)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_axis_off()
    ax.set_title(args.mesh.name, fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
