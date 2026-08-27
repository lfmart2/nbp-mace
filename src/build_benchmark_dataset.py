#!/usr/bin/env python3
"""Build a provenance-rich, method-separated M0 benchmark dataset."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write


GROUPS = (
    {"id": "qe_clean_relax", "path": "qe/relax_clean_collinear/NbP.supercell_vasp.relax.out", "input": "qe/relax_clean_collinear/NbP.supercell_vasp.relax.in", "reader": "ase", "code": "QE", "physics": "collinear", "role": "primary"},
    {"id": "qe_h_relax_a", "path": "qe/relax_h_collinear_run_a/NbP_H.supercell.nspin.relax.out", "input": "qe/relax_h_collinear_run_a/NbP_noSpin_Relaxation.in", "reader": "ase", "code": "QE", "physics": "collinear", "role": "primary"},
    {"id": "qe_h_relax_b", "path": "qe/relax_h_collinear_run_b/NbP_H.supercell.spin.relax.out", "input": "qe/relax_h_collinear_run_b/NbP_H.supercell.spin.relax.in.gz", "reader": "ase", "code": "QE", "physics": "collinear", "role": "overlap_diagnostic"},
    {"id": "vasp_clean_relax_30", "path": "qe_tree_vasp/relax_clean_30/vasprun.xml", "reader": "ase", "code": "VASP", "physics": "collinear", "role": "primary"},
    {"id": "vasp_soc_clean_relax", "path": "vasp_soc/relax_clean_soc_48/OUTCAR", "reader": "outcar", "code": "VASP", "physics": "soc_noncollinear", "role": "primary"},
    {"id": "vasp_soc_h_relax", "path": "vasp_soc/relax_h_soc_13_incomplete/OUTCAR", "reader": "outcar", "code": "VASP", "physics": "soc_noncollinear", "role": "primary_incomplete"},
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def geometry_hash(atoms: Atoms, decimals: int = 6) -> str:
    digest = hashlib.sha256()
    digest.update(" ".join(atoms.get_chemical_symbols()).encode())
    digest.update(np.round(atoms.cell.array, decimals).tobytes())
    digest.update(np.round(atoms.positions, decimals).tobytes())
    return digest.hexdigest()


def read_poscar_template(path: Path) -> Atoms:
    """Read the conservative POSCAR subset used by the archived slab jobs."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    scale = float(lines[1].split()[0])
    cell = np.asarray([[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)]) * scale
    symbols = lines[5].split()
    counts = [int(value) for value in lines[6].split()]
    selective = lines[7].strip().lower().startswith("s")
    mode_index = 8 if selective else 7
    cartesian = lines[mode_index].strip().lower().startswith(("c", "k"))
    start = mode_index + 1
    natoms = sum(counts)
    positions, fixed = [], []
    for index, line in enumerate(lines[start : start + natoms]):
        fields = line.split()
        positions.append([float(value) for value in fields[:3]])
        if selective and len(fields) >= 6 and all(value.upper().startswith("F") for value in fields[3:6]):
            fixed.append(index)
    atoms = Atoms(sum(([symbol] * count for symbol, count in zip(symbols, counts)), []), cell=cell, pbc=True)
    if cartesian:
        atoms.positions = np.asarray(positions) * scale
    else:
        atoms.set_scaled_positions(positions)
    if fixed:
        atoms.set_constraint(FixAtoms(indices=fixed))
    return atoms


def parse_vasp_outcar(path: Path) -> list[Atoms]:
    """Parse ionic positions, forces, and sigma->0 energies from a VASP OUTCAR.

    This intentionally narrow parser handles the archived fixed-cell relaxation
    outputs that ASE cannot read because their headers lack an initial position
    block. Chemical symbols, cell, and constraints come from the sibling POSCAR.
    """
    template = read_poscar_template(path.with_name("POSCAR"))
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    starts = [i for i, line in enumerate(lines) if "TOTAL-FORCE (eV/Angst)" in line]
    frames: list[Atoms] = []
    energy_pattern = re.compile(r"energy\(sigma->0\)\s*=\s*([-+0-9.Ee]+)")
    for frame_index, start in enumerate(starts):
        positions, forces = [], []
        for line in lines[start + 2 : start + 2 + len(template)]:
            fields = line.split()
            if len(fields) < 6:
                raise ValueError(f"Incomplete force block {frame_index} in {path}")
            values = [float(value) for value in fields[:6]]
            positions.append(values[:3])
            forces.append(values[3:])
        stop = starts[frame_index + 1] if frame_index + 1 < len(starts) else len(lines)
        energy = None
        for line in lines[start:stop]:
            match = energy_pattern.search(line)
            if match:
                energy = float(match.group(1))
        if energy is None:
            raise ValueError(f"No sigma->0 energy for force block {frame_index} in {path}")
        atoms = Atoms(template.symbols, positions=positions, cell=template.cell, pbc=True)
        atoms.set_constraint(template.constraints)
        atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=np.asarray(forces))
        frames.append(atoms)
    return frames


def free_mask(atoms: Atoms) -> np.ndarray:
    mask = np.ones(len(atoms), dtype=bool)
    for constraint in atoms.constraints:
        if hasattr(constraint, "get_indices"):
            mask[np.asarray(constraint.get_indices(), dtype=int)] = False
    return mask


def qe_fixed_indices(path: Path, natoms: int) -> list[int]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        lines = handle.read().splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().upper().startswith("ATOMIC_POSITIONS")) + 1
    fixed = []
    for index, line in enumerate(lines[start : start + natoms]):
        fields = line.split()
        if len(fields) >= 7 and fields[-3:] == ["0", "0", "0"]:
            fixed.append(index)
    return fixed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("data/m0_benchmark.xyz"))
    parser.add_argument("--summary", type=Path, default=Path("artifacts/dataset_summary.json"))
    args = parser.parse_args()

    output_frames: list[Atoms] = []
    groups = []
    for spec in GROUPS:
        source = args.raw_root / spec["path"]
        frames = read(source, index=":") if spec["reader"] == "ase" else parse_vasp_outcar(source)
        if spec["code"] == "QE" and spec.get("input"):
            fixed = qe_fixed_indices(args.raw_root / spec["input"], len(frames[0]))
            for atoms in frames:
                atoms.set_constraint(FixAtoms(indices=fixed))
        seen: dict[str, int] = {}
        unique_frames = []
        duplicate_indices = []
        for original_index, atoms in enumerate(frames):
            key = geometry_hash(atoms)
            if key in seen:
                duplicate_indices.append({"index": original_index, "duplicate_of": seen[key]})
                continue
            seen[key] = original_index
            atoms.info.update({
                "source_group": spec["id"],
                "source_frame": original_index,
                "code": spec["code"],
                "physics": spec["physics"],
                "benchmark_role": spec["role"],
                "geometry_sha256": key,
                "REF_energy": float(atoms.get_potential_energy()),
            })
            atoms.arrays["REF_forces"] = np.asarray(atoms.get_forces(), dtype=float)
            atoms.arrays["movable_mask"] = free_mask(atoms)
            atoms.calc = None
            unique_frames.append(atoms)
            output_frames.append(atoms)
        energies = np.asarray([a.info["REF_energy"] for a in unique_frames])
        force_norms = np.concatenate([
            np.linalg.norm(a.arrays["REF_forces"][a.arrays["movable_mask"]], axis=1)
            for a in unique_frames
        ])
        groups.append({
            **{key: spec[key] for key in ("id", "code", "physics", "role")},
            "source_id": f"raw://{spec['path']}",
            "source_sha256": sha256(source),
            "frames_parsed": len(frames),
            "frames_unique": len(unique_frames),
            "duplicate_frames": duplicate_indices,
            "formula": unique_frames[0].get_chemical_formula(),
            "atoms_per_frame": len(unique_frames[0]),
            "energy_span_eV": float(np.ptp(energies)),
            "movable_force_norm_max_eV_per_A": float(force_norms.max()),
        })

    args.dataset.parent.mkdir(parents=True, exist_ok=True)
    write(args.dataset, output_frames, format="extxyz")
    summary = {
        "schema_version": 1,
        "description": "Method-separated relaxation trajectories for the M0 zero-shot benchmark",
        "deduplication": "SHA-256 of symbols and cell/Cartesian coordinates rounded to 1e-6 A; within source group only",
        "split_policy": "No training in M0. Metrics reported by source calculation; no random frame split.",
        "total_frames_parsed": sum(group["frames_parsed"] for group in groups),
        "total_frames_unique_within_group": len(output_frames),
        "groups": groups,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
