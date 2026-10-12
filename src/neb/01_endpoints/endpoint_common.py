"""Shared helpers for T1-01 (endpoint selection for an MLIP-driven CI-NEB of H on NbP).

Everything produced with these helpers is an MLIP estimate (MACE M0 / M1), not DFT.
Raw DFT on D: is only read, never written. Paths come from 90_env/env.ps1.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read

HERE = Path(__file__).resolve().parent
R0_SCF_NAME = "out_r0_scf_20260225_133316"          # from artifacts/zscan_manifest.csv
SOC_H_RELAX_REL = Path("vasp_soc") / "relax_h_soc_13_incomplete"


# --------------------------------------------------------------------------- paths
def env_path(name: str, required: bool = True) -> Path | None:
    value = os.environ.get(name, "").strip()
    if not value:
        if required:
            raise SystemExit(f"{name} is not set. Run:  . your private env.ps1 (RUNBOOK section 8)")
        return None
    return Path(value)


def out_dir(tag: str) -> Path:
    path = HERE / f"{_dt.date.today().isoformat()}_{tag}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "date": _dt.datetime.now().isoformat(timespec="seconds")}
    for mod in ("numpy", "ase", "torch", "mace"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            info[mod] = None
    return info


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, default=_json_default), encoding="utf-8")


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj))


# --------------------------------------------------------------------------- data
def zscan_path() -> Path:
    return env_path("NBP_DATA_DIR") / "zscan.xyz"


def load_zscan() -> list[Atoms]:
    return read(zscan_path(), index=":", format="extxyz")


def get_label(frames: list[Atoms], label: str) -> Atoms:
    for atoms in frames:
        if str(atoms.info.get("r_label")) == label:
            return atoms
    raise SystemExit(f"No frame with r_label={label} in {zscan_path()}")


def parse_poscar_flags(path: Path) -> tuple[Atoms, list[tuple[bool, bool, bool]] | None]:
    """Return (atoms, per-atom 'free' flags) from a VASP POSCAR/CONTCAR.

    flags is None when the file has no Selective dynamics block. True = free to move.
    """
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    scale = float(lines[1].split()[0])
    cell = np.array([[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)]) * scale
    symbols = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    idx = 7
    selective = lines[idx].strip().lower().startswith("s")
    if selective:
        idx += 1
    cartesian = lines[idx].strip().lower().startswith(("c", "k"))
    idx += 1
    n = sum(counts)
    pos, flags = [], []
    for line in lines[idx: idx + n]:
        f = line.split()
        pos.append([float(x) for x in f[:3]])
        if selective:
            flags.append(tuple(v.upper().startswith("T") for v in f[3:6]))
    species = sum(([s] * c for s, c in zip(symbols, counts)), [])
    atoms = Atoms(species, cell=cell, pbc=True)
    if cartesian:
        atoms.positions = np.array(pos) * scale
    else:
        atoms.set_scaled_positions(pos)
    return atoms, (flags if selective else None)


def min_image_xy(vec: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Minimum-image of in-plane vectors (cell a,b in-plane, c along z)."""
    vec = np.atleast_2d(np.asarray(vec, float)).copy()
    inv = np.linalg.inv(cell[:2, :2])
    frac = vec[:, :2] @ inv
    frac -= np.round(frac)
    vec[:, :2] = frac @ cell[:2, :2]
    return vec


def match_atoms(ref: Atoms, target: Atoms) -> tuple[np.ndarray, float]:
    """For each atom of ref, the index of the nearest same-species atom in target (min image)."""
    mapping = np.full(len(ref), -1)
    worst = 0.0
    used = set()
    for i, (s, p) in enumerate(zip(ref.get_chemical_symbols(), ref.positions)):
        cand = [j for j, t in enumerate(target.get_chemical_symbols()) if t == s and j not in used]
        if not cand:
            continue
        d = target.positions[cand] - p
        d = min_image_xy(d, target.cell.array)
        dist = np.linalg.norm(d, axis=1)
        k = int(np.argmin(dist))
        mapping[i] = cand[k]
        used.add(cand[k])
        worst = max(worst, float(dist[k]))
    return mapping, worst


def fixed_source_candidates() -> list[tuple[str, Path]]:
    out = []
    raw = env_path("NBP_MLIP_RAW_ROOT", required=False)
    if raw:
        out.append(("SOC H relaxation (origin of r0 geometry)", raw / SOC_H_RELAX_REL / "POSCAR"))
    scf = env_path("NBP_SCF_ROOT", required=False)
    if scf:
        out.append(("r0 collinear SCF", scf / R0_SCF_NAME / "POSCAR"))
    soc = env_path("NBP_SOC_RELAX", required=False)
    if soc:
        out.append(("SOC clean-slab relaxation", soc / "POSCAR"))
    return out


def dft_fixed_mask(atoms: Atoms, source: str = "auto") -> tuple[np.ndarray, dict]:
    """Boolean mask (True = fixed) for atoms, copied from a DFT POSCAR's selective-dynamics flags.

    Atoms are matched by species + nearest position, so ordering may differ. H is never fixed.
    """
    cands = fixed_source_candidates()
    if source != "auto":
        cands = [("user-given", Path(source))]
    for label, path in cands:
        if not path.is_file():
            continue
        ref, flags = parse_poscar_flags(path)
        if flags is None or all(all(f) for f in flags):
            continue
        mapping, worst = match_atoms(ref, atoms)
        mask = np.zeros(len(atoms), bool)
        for i, f in enumerate(flags):
            if mapping[i] >= 0 and not any(f):
                mask[mapping[i]] = True
        mask[np.array(atoms.get_chemical_symbols()) == "H"] = False
        partial = sum(1 for f in flags if any(f) and not all(f))
        return mask, {"source": label, "path": str(path), "max_match_distance_A": worst,
                      "n_fixed": int(mask.sum()), "n_partially_fixed_in_poscar": partial}
    raise SystemExit("No POSCAR with selective-dynamics flags found; pass --fixed-from <POSCAR>.")


# --------------------------------------------------------------------------- surface geometry
def h_index(atoms: Atoms) -> int:
    idx = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "H"]
    if len(idx) != 1:
        raise SystemExit(f"Expected exactly one H, found {len(idx)}")
    return idx[0]


def substrate_layers(atoms: Atoms, tol: float = 0.3) -> list[dict]:
    """Group non-H atoms into z-layers, top first."""
    sym = np.array(atoms.get_chemical_symbols())
    sub = np.where(sym != "H")[0]
    order = sub[np.argsort(-atoms.positions[sub, 2])]
    layers: list[dict] = []
    for i in order:
        z = atoms.positions[i, 2]
        if layers and abs(layers[-1]["z"] - z) < tol:
            layers[-1]["idx"].append(int(i))
        else:
            layers.append({"z": float(z), "idx": [int(i)]})
    for k, L in enumerate(layers):
        L["layer"] = k + 1
        L["species"] = "".join(sorted(set(sym[L["idx"]])))
        L["z"] = float(np.mean(atoms.positions[L["idx"], 2]))
    return layers


def surface_height(atoms: Atoms, z: float) -> float:
    """Repo convention: height above the topmost substrate atom."""
    sym = np.array(atoms.get_chemical_symbols())
    return float(z - atoms.positions[sym != "H", 2].max())


def classify_site(atoms: Atoms, xy, layers: list[dict] | None = None, nlayers: int = 4,
                  tol: float = 0.35) -> dict:
    """Describe a lateral position by the nearest projected atoms of the top layers."""
    layers = layers or substrate_layers(atoms)
    cell = atoms.cell.array
    point = np.array([xy[0], xy[1], 0.0])
    near = []
    for L in layers[:nlayers]:
        d = min_image_xy(atoms.positions[L["idx"]] - point, cell)
        dist = np.linalg.norm(d[:, :2], axis=1)
        k = int(np.argmin(dist))
        near.append({"layer": L["layer"], "species": atoms[L["idx"][k]].symbol,
                     "lateral_distance_A": float(dist[k])})
    # bridge of the top layer: midpoint of nearest top-layer pair (incl. periodic images)
    top = layers[0]
    a, b = cell[0], cell[1]
    images = [atoms.positions[i] + m * a + n * b for i in top["idx"]
              for m in (-1, 0, 1) for n in (-1, 0, 1)]
    images = np.array(images)
    nn = np.min([np.linalg.norm(images[i, :2] - images[j, :2])
                 for i in range(len(images)) for j in range(i + 1, len(images))])
    bridges = [(images[i, :2] + images[j, :2]) / 2 for i in range(len(images))
               for j in range(i + 1, len(images))
               if abs(np.linalg.norm(images[i, :2] - images[j, :2]) - nn) < 0.05]
    dbr = min(np.linalg.norm(min_image_xy(np.r_[br - point[:2], 0.0], cell)[0, :2]) for br in bridges)
    label = "other"
    below = next((f" over L{n['layer']} {n['species']}" for n in near[1:] if n["lateral_distance_A"] < tol), "")
    if near[0]["lateral_distance_A"] < tol:
        label = f"top (above L1 {near[0]['species']})"
    elif dbr < tol:
        label = f"bridge (L1 {top['species']}-{top['species']}){below}"
    else:
        for n in near[1:]:
            if n["lateral_distance_A"] < tol:
                label = f"hollow-type (above L{n['layer']} {n['species']})"
                break
        else:
            label = "hollow / low-symmetry"
    return {"label": label, "nearest_projected": near, "distance_to_L1_bridge_A": float(dbr)}


# --------------------------------------------------------------------------- models
def model_paths() -> dict[str, Path]:
    return {"M0": env_path("MACE_MP0_SMALL_CHECKPOINT"), "M1": env_path("MACE_M1_CHECKPOINT")}


def make_calc(model: str, device: str = "cpu", dtype: str = "float64"):
    from mace.calculators import MACECalculator
    path = model_paths()[model]
    if not path.is_file():
        raise SystemExit(f"Checkpoint for {model} not found: {path}")
    return MACECalculator(model_paths=str(path), device=device, default_dtype=dtype)
