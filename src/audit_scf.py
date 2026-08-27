#!/usr/bin/env python3
"""Audit the extracted collinear NbP-H SCF series and build portfolio artifacts.

This module deliberately uses only the Python standard library. It reads VASP
text outputs in place, never modifies raw data, and writes small provenance,
CSV, JSON, and SVG artifacts suitable for version control.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def mat_vec(cell, frac):
    return [sum(frac[j] * cell[j][i] for j in range(3)) for i in range(3)]


def parse_poscar_text(text: str) -> dict:
    lines = [line.strip() for line in text.splitlines()]
    scale = float(lines[1].split()[0])
    raw_cell = [[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)]
    cell = [[scale * x for x in row] for row in raw_cell]
    symbols = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    idx = 7
    selective = lines[idx].lower().startswith("s")
    if selective:
        idx += 1
    direct = lines[idx].lower().startswith(("d", "f"))
    idx += 1
    species, positions, flags = [], [], []
    for symbol, count in zip(symbols, counts):
        for _ in range(count):
            fields = lines[idx].split()
            p = [float(x) for x in fields[:3]]
            positions.append(mat_vec(cell, p) if direct else [scale * x for x in p])
            species.append(symbol)
            flags.append(fields[3:6] if selective and len(fields) >= 6 else [])
            idx += 1
    return {
        "comment": lines[0],
        "cell_A": cell,
        "species": species,
        "positions_A": positions,
        "selective_flags": flags,
    }


def parse_poscar(path: Path) -> dict:
    return parse_poscar_text(path.read_text(errors="replace"))


def last_match(pattern: str, text: str, cast=float):
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    return cast(matches[-1]) if matches else None


def parse_outcar_text(text: str) -> dict:
    force_headers = list(re.finditer(r"TOTAL-FORCE \(eV/Angst\)", text))
    forces = []
    if force_headers:
        tail = text[force_headers[-1].end():]
        for line in tail.splitlines()[2:]:
            fields = line.split()
            if len(fields) < 6:
                if forces:
                    break
                continue
            try:
                values = [float(x) for x in fields[:6]]
            except ValueError:
                if forces:
                    break
                continue
            forces.append(values[3:6])

    norms = [math.sqrt(sum(x * x for x in f)) for f in forces]
    return {
        "completed": "General timing and accounting informations for this job" in text,
        "electronic_error": bool(re.search(r"Error EDDDAV|ZHEGV failed|BRMIX: very serious", text)),
        "nions": last_match(r"NIONS\s*=\s*(\d+)", text, int),
        "ispin": last_match(r"^\s*ISPIN\s*=\s*(\d+)", text, int),
        "lsorbit": last_match(r"^\s*LSORBIT\s*=\s*([TF])", text, str),
        "lnoncollinear": last_match(r"^\s*LNONCOLLINEAR\s*=\s*([TF])", text, str),
        "encut_eV": last_match(r"^\s*ENCUT\s*=\s*(%s)" % FLOAT, text),
        "ediff_eV": last_match(r"^\s*EDIFF\s*=\s*(%s)" % FLOAT, text),
        "ismear": last_match(r"^\s*ISMEAR\s*=\s*(-?\d+)", text, int),
        "sigma_eV": last_match(r"^\s*ISMEAR\s*=.*?SIGMA\s*=\s*(%s)" % FLOAT, text),
        "toten_eV": last_match(r"free\s+energy\s+TOTEN\s*=\s*(%s)\s+eV" % FLOAT, text),
        "energy_without_entropy_eV": last_match(
            r"energy\s+without entropy\s*=\s*(%s)" % FLOAT, text
        ),
        "energy_sigma0_eV": last_match(r"energy\(sigma->0\)\s*=\s*(%s)" % FLOAT, text),
        "forces_eV_per_A": forces,
        "force_count": len(forces),
        "max_force_eV_per_A": max(norms) if norms else None,
    }


def parse_outcar(path: Path) -> dict:
    return parse_outcar_text(path.read_text(errors="replace"))


def r_label(name: str) -> str | None:
    if "r0_nH" in name:
        return "clean"
    m = re.search(r"(?:^|_)r([0-8])(?:_|$)", name)
    return f"r{m.group(1)}" if m else None


def audit_case(directory: Path, source_prefix: str = "raw://03_SCF") -> dict:
    label = r_label(directory.name)
    poscar_path = directory / "POSCAR"
    outcar_path = directory / "OUTCAR"
    pos = parse_poscar(poscar_path)
    out = parse_outcar(outcar_path)
    h_indices = [i for i, s in enumerate(pos["species"]) if s == "H"]
    substrate = [i for i, s in enumerate(pos["species"]) if s != "H"]
    h_index = h_indices[0] if h_indices else None
    h_z = pos["positions_A"][h_index][2] if h_index is not None else None
    surface_z = max(pos["positions_A"][i][2] for i in substrate) if substrate else None
    height = h_z - surface_z if h_z is not None and surface_z is not None else None
    h_force = out["forces_eV_per_A"][h_index] if h_index is not None and out["force_count"] > h_index else None
    substrate_norms = []
    if out["force_count"] == len(pos["species"]):
        substrate_norms = [
            math.sqrt(sum(x * x for x in out["forces_eV_per_A"][i])) for i in substrate
        ]
    return {
        "label": label,
        "source_id": f"{source_prefix}/{directory.name}",
        "source_name": directory.name,
        "poscar_sha256": sha256(poscar_path),
        "outcar_sha256": sha256(outcar_path),
        "composition": {s: pos["species"].count(s) for s in sorted(set(pos["species"]))},
        "atom_count": len(pos["species"]),
        "cell_A": pos["cell_A"],
        "h_index_zero_based": h_index,
        "h_z_A": h_z,
        "surface_z_A": surface_z,
        "h_height_A": height,
        "h_force_eV_per_A": h_force,
        "max_substrate_force_eV_per_A": max(substrate_norms) if substrate_norms else None,
        **{k: v for k, v in out.items() if k != "forces_eV_per_A"},
    }


def method_signature(row: dict) -> tuple:
    return tuple(row.get(k) for k in (
        "ispin", "lsorbit", "lnoncollinear", "encut_eV", "ediff_eV", "ismear", "sigma_eV"
    ))


def svg_plot(rows: list[dict], path: Path) -> None:
    scan = [r for r in rows if r["label"] and r["label"].startswith("r")]
    scan.sort(key=lambda r: r["h_height_A"])
    if not scan:
        return
    x = [r["h_height_A"] for r in scan]
    e0 = min(r["energy_sigma0_eV"] for r in scan)
    y = [r["energy_sigma0_eV"] - e0 for r in scan]
    width, height = 800, 500
    left, right, top, bottom = 85, 30, 45, 70
    xmin, xmax = min(x), max(x)
    ymin, ymax = 0.0, max(y)
    if xmax == xmin:
        xmax += 1
    if ymax == ymin:
        ymax += 1
    sx = lambda v: left + (v - xmin) / (xmax - xmin) * (width - left - right)
    sy = lambda v: height - bottom - (v - ymin) / (ymax - ymin) * (height - top - bottom)
    points = " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in zip(x, y))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#18212b}.axis{stroke:#18212b;stroke-width:1.5}.grid{stroke:#dce3ea;stroke-width:1}.curve{fill:none;stroke:#1769aa;stroke-width:3}.point{fill:#ff8f00;stroke:white;stroke-width:1.5}</style>',
        f'<line class="axis" x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"/>',
    ]
    for i in range(6):
        value = ymin + i * (ymax - ymin) / 5
        yy = sy(value)
        parts += [f'<line class="grid" x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}"/>',
                  f'<text x="{left-10}" y="{yy+5:.1f}" text-anchor="end" font-size="13">{value:.2f}</text>']
    for i in range(5):
        value = xmin + i * (xmax - xmin) / 4
        xx = sx(value)
        parts.append(f'<text x="{xx:.1f}" y="{height-bottom+25}" text-anchor="middle" font-size="13">{value:.2f}</text>')
    parts.append(f'<polyline class="curve" points="{points}"/>')
    for row, a, b in zip(scan, x, y):
        parts += [f'<circle class="point" cx="{sx(a):.1f}" cy="{sy(b):.1f}" r="5"/>',
                  f'<text x="{sx(a):.1f}" y="{sy(b)-10:.1f}" text-anchor="middle" font-size="12">{row["label"]}</text>']
    parts += [
        f'<text x="{width/2}" y="25" text-anchor="middle" font-size="20" font-weight="bold">DFT constrained H-height scan on NbP</text>',
        f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-size="15">H height above topmost substrate atom (Å)</text>',
        f'<text x="20" y="{height/2}" text-anchor="middle" font-size="15" transform="rotate(-90 20 {height/2})">E(z) − E_min (eV)</text>',
        '</svg>'
    ]
    path.write_text("\n".join(parts), encoding="utf-8")


def write_extxyz(directories: list[Path], path: Path) -> None:
    """Write the private local benchmark used by ASE/MACE.

    This file contains atomic coordinates and is intentionally written under
    the git-ignored data directory, not the public artifacts directory.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for directory in directories:
            label = r_label(directory.name)
            if not label or not label.startswith("r"):
                continue
            pos = parse_poscar(directory / "POSCAR")
            out = parse_outcar(directory / "OUTCAR")
            if out["force_count"] != len(pos["species"]):
                raise SystemExit(f"Incomplete force array for {directory.name}")
            lattice = " ".join(f"{x:.12g}" for row in pos["cell_A"] for x in row)
            source = f"raw://03_SCF/{directory.name}"
            header = (
                f'Lattice="{lattice}" '
                'Properties=species:S:1:pos:R:3:REF_forces:R:3 '
                f'REF_energy={out["energy_sigma0_eV"]:.12g} '
                f'r_label={label} source={source} geometry_origin=SOC-relaxed '
                'label_method=collinear-SCF pbc="T T T"'
            )
            f.write(f"{len(pos['species'])}\n{header}\n")
            for symbol, xyz, force in zip(pos["species"], pos["positions_A"], out["forces_eV_per_A"]):
                values = [*xyz, *force]
                f.write(symbol + " " + " ".join(f"{x:.12g}" for x in values) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scf-root", type=Path, required=True)
    ap.add_argument("--h2-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("artifacts"))
    ap.add_argument("--dataset-out", type=Path, default=None,
                    help="optional git-ignored extxyz output for local MACE evaluation")
    args = ap.parse_args()

    directories = sorted(
        d for d in args.scf_root.iterdir()
        if d.is_dir() and r_label(d.name) and (d / "POSCAR").exists() and (d / "OUTCAR").exists()
    )
    rows = [audit_case(d) for d in directories]
    h2 = audit_case(args.h2_dir, source_prefix="raw://H2")
    h2["label"] = "H2"

    expected = {"clean", *(f"r{i}" for i in range(9))}
    observed = {r["label"] for r in rows}
    duplicate_labels = sorted(label for label in observed if sum(r["label"] == label for r in rows) > 1)
    signatures = {method_signature(r) for r in rows}
    scan = [r for r in rows if r["label"].startswith("r")]
    force_complete = all(r["force_count"] == r["atom_count"] for r in scan)
    substrate_hashes = set()
    for d in directories:
        if r_label(d.name) == "clean":
            continue
        p = parse_poscar(d / "POSCAR")
        substrate = [(s, [round(v, 8) for v in xyz]) for s, xyz in zip(p["species"], p["positions_A"]) if s != "H"]
        substrate_hashes.add(hashlib.sha256(json.dumps(substrate).encode()).hexdigest())

    summary = {
        "schema_version": 1,
        "energy_convention": "VASP energy(sigma->0)",
        "scf_root": "raw://03_SCF",
        "h2_directory": "raw://H2",
        "observed_labels": sorted(observed),
        "missing_labels": sorted(expected - observed),
        "duplicate_labels": duplicate_labels,
        "all_scf_completed": all(r["completed"] and not r["electronic_error"] for r in rows),
        "all_zscan_forces_complete": force_complete,
        "zscan_method_signatures": len(signatures),
        "zscan_substrate_geometry_hashes": len(substrate_hashes),
        "h2_completed": h2["completed"] and not h2["electronic_error"],
        "h2_force_count": h2["force_count"],
        "h2_max_force_eV_per_A": h2["max_force_eV_per_A"],
        "notes": [
            "Geometry origin is SOC relaxation; reference labels are collinear SCF.",
            "r0-r8 are frozen static points, not a relaxed pathway.",
            "Failed r7/r8 archives are excluded in favor of successful reruns.",
        ],
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "scf_audit.json").write_text(json.dumps({"summary": summary, "cases": rows, "h2": h2}, indent=2), encoding="utf-8")
    fields = [
        "label", "source_name", "atom_count", "h_z_A", "surface_z_A", "h_height_A",
        "energy_sigma0_eV", "toten_eV", "force_count", "h_fx_eV_per_A", "h_fy_eV_per_A",
        "h_fz_eV_per_A", "max_substrate_force_eV_per_A", "completed", "electronic_error",
        "ispin", "lsorbit", "lnoncollinear", "encut_eV", "ediff_eV", "ismear", "sigma_eV",
        "source_id", "poscar_sha256", "outcar_sha256",
    ]
    scan_rows = []
    scan_energies = [r["energy_sigma0_eV"] for r in scan]
    emin = min(scan_energies) if scan_energies else None
    for r in rows:
        hf = r.get("h_force_eV_per_A") or [None, None, None]
        flat = {k: r.get(k) for k in fields}
        flat.update(h_fx_eV_per_A=hf[0], h_fy_eV_per_A=hf[1], h_fz_eV_per_A=hf[2])
        flat["relative_energy_eV"] = r["energy_sigma0_eV"] - emin if r["label"].startswith("r") else None
        scan_rows.append(flat)
    csv_fields = fields[:6] + ["relative_energy_eV"] + fields[6:]
    with (args.out / "zscan_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(scan_rows)
    svg_plot(rows, args.out / "dft_zscan.svg")
    if args.dataset_out is not None:
        write_extxyz(directories, args.dataset_out)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
