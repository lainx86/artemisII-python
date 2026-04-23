from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))


def module_available(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def select_matplotlib_backend() -> str:
    """
    Prioritas backend:
    1. override manual via ARTEMIS_MPL_BACKEND
    2. TkAgg untuk jendela interaktif desktop yang paling stabil di Linux
    3. QtAgg sebagai fallback, dipaksa lewat xcb agar tidak masuk plugin wayland
    4. WebAgg jika display ada tapi toolkit desktop tidak sehat
    5. Agg untuk mode non-interaktif/headless
    """

    explicit_backend = os.environ.get("ARTEMIS_MPL_BACKEND")
    if explicit_backend:
        return explicit_backend

    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if not has_display:
        return "Agg"

    if module_available("tkinter"):
        return "TkAgg"

    if any(module_available(name) for name in ("PyQt5", "PySide6", "PySide2", "PyQt6")):
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
        return "QtAgg"

    return "WebAgg"

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "Dependensi inti belum lengkap. Install minimal: `python -m pip install numpy pandas`."
    ) from exc

try:
    import matplotlib as mpl
    MPL_BACKEND = select_matplotlib_backend()
    mpl.use(MPL_BACKEND)
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
except ImportError as exc:
    raise SystemExit(
        "Matplotlib belum terpasang. Install dengan: `python -m pip install matplotlib`."
    ) from exc

def is_conda_managed_path(path: Path) -> bool:
    path_text = str(path).lower()
    return any(marker in path_text for marker in ("micromamba", "conda", "mamba", "anaconda", "miniconda"))


def ffmpeg_candidate_paths() -> list[Path]:
    candidates: list[Path] = []
    explicit_path = os.environ.get("ARTEMIS_FFMPEG_PATH")
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())

    for preferred in ("/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        candidates.append(Path(preferred))

    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if entry:
            candidates.append(Path(entry) / "ffmpeg")

    existing_candidates: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            existing_candidates.append(candidate.resolve())

    explicit_resolved = str(Path(explicit_path).expanduser().resolve()) if explicit_path else None
    explicit: list[Path] = []
    system: list[Path] = []
    non_conda: list[Path] = []
    conda_managed: list[Path] = []

    for candidate in existing_candidates:
        candidate_str = str(candidate)
        if explicit_resolved and candidate_str == explicit_resolved:
            explicit.append(candidate)
        elif candidate_str in {"/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"}:
            system.append(candidate)
        elif is_conda_managed_path(candidate):
            conda_managed.append(candidate)
        else:
            non_conda.append(candidate)

    return explicit + system + non_conda + conda_managed


def probe_ffmpeg(ffmpeg_path: Path) -> tuple[bool, str | None]:
    try:
        subprocess.run(
            [str(ffmpeg_path), "-version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True, None
    except (OSError, subprocess.CalledProcessError) as exc:
        return False, str(exc)


def resolve_ffmpeg() -> tuple[str | None, str | None]:
    candidates = ffmpeg_candidate_paths()
    if not candidates:
        return None, "ffmpeg tidak ditemukan di PATH. Output MP4 akan fallback ke GIF."

    errors: list[str] = []
    for candidate in candidates:
        is_ok, error = probe_ffmpeg(candidate)
        if is_ok:
            return str(candidate), None
        errors.append(f"{candidate}: {error}")

    return (
        None,
        "ffmpeg ditemukan tetapi tidak ada binary yang usable. "
        + "Candidate yang dicoba: "
        + " | ".join(errors)
        + ". Output MP4 akan fallback ke GIF.",
    )


FFMPEG_PATH, FFMPEG_WARNING = resolve_ffmpeg()
if FFMPEG_PATH:
    mpl.rcParams["animation.ffmpeg_path"] = FFMPEG_PATH


def prepare_runtime_backend(show_plots: bool) -> str:
    current_backend = plt.get_backend()

    if not show_plots:
        if current_backend.lower() != "agg":
            plt.switch_backend("Agg")
        return plt.get_backend()

    try:
        probe_fig = plt.figure()
        plt.close(probe_fig)
        return plt.get_backend()
    except Exception as exc:
        warnings.warn(
            f"Backend interaktif {current_backend!r} gagal dipakai ({exc}). "
            "Mencoba fallback ke WebAgg."
        )
        try:
            plt.switch_backend("WebAgg")
            probe_fig = plt.figure()
            plt.close(probe_fig)
            return plt.get_backend()
        except Exception as webagg_exc:
            raise RuntimeError(
                "Tidak ada backend interaktif yang berhasil dipakai. "
                "Coba jalankan dari sesi desktop lokal atau set "
                "`ARTEMIS_MPL_BACKEND=TkAgg` / `QtAgg` secara manual."
            ) from webagg_exc


class EphemerisParseError(RuntimeError):
    """Raised when an ephemeris file cannot be parsed into a usable table."""


@dataclass
class ManualParserConfig:
    """
    Fallback parser for non-Horizons text files.

    UBAH BAGIAN INI JIKA FORMAT FILE ANDA BERBEDA:
    Aktifkan parser manual hanya jika auto-parser Horizons gagal karena file
    Anda memakai delimiter atau urutan kolom yang berbeda.
    """

    enabled: bool = False
    delimiter: str | None = None
    skiprows: int = 0
    comment: str | None = None
    column_names: list[str] | None = None
    time_column: str = "time"
    x_column: str = "x"
    y_column: str = "y"
    z_column: str = "z"
    julian_day_column: str | None = None


@dataclass
class AppConfig:
    # UBAH BAGIAN INI JIKA NAMA FILE ANDA BERBEDA
    moon_file: Path = Path("moon_ephemeris.txt")
    artemis_file: Path = Path("artemis_ephemeris.txt")

    # UBAH BAGIAN INI JIKA INGIN NAMA OUTPUT BERBEDA
    output_dir: Path = Path("outputs")
    static_plot_name: str = "artemis_ii_static_xy.png"
    animation_name: str = "artemis_ii_animation.mp4"

    # UBAH BAGIAN INI JIKA INGIN BIDANG PROYEKSI BERBEDA
    plot_axes: tuple[str, str] = ("x", "y")

    animation_frame_step: int = 4
    animation_interval_ms: int = 40
    animation_fps: int = 20
    animation_blit_interactive: bool = False
    animation_blit_offscreen: bool = True
    figure_dpi: int = 140
    show_plots: bool = os.environ.get("ARTEMIS_SHOW_PLOTS", "1").lower() not in {"0", "false", "no"}
    save_static_plot: bool = True
    save_animation: bool = True
    save_animation_when_interactive: bool = (
        os.environ.get("ARTEMIS_SAVE_ANIMATION_IN_INTERACTIVE", "0").lower()
        in {"1", "true", "yes"}
    )

    prefer_timebase_from: str = "artemis"
    strict_metadata_checks: bool = True

    manual_parser: ManualParserConfig = field(default_factory=ManualParserConfig)


CONFIG = AppConfig()

AXIS_LABELS = {
    "x": "X (km)",
    "y": "Y (km)",
    "z": "Z (km)",
}


def read_text_lines(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def extract_metadata(lines: Iterable[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("*") or line.startswith("$$"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def normalize_name(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def make_unique(names: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    unique_names: list[str] = []
    for name in names:
        counts[name] = counts.get(name, 0) + 1
        if counts[name] == 1:
            unique_names.append(name)
        else:
            unique_names.append(f"{name}_{counts[name]}")
    return unique_names


def find_horizons_data_bounds(lines: list[str]) -> tuple[int, int]:
    start_index = next((i for i, line in enumerate(lines) if "$$SOE" in line), None)
    end_index = next((i for i, line in enumerate(lines) if "$$EOE" in line), None)
    if start_index is None or end_index is None or end_index <= start_index:
        raise EphemerisParseError(
            "Marker data Horizons `$$SOE` / `$$EOE` tidak ditemukan atau tidak valid."
        )
    return start_index + 1, end_index


def find_horizons_header_line(lines: list[str], soe_index: int) -> str:
    for idx in range(soe_index - 1, -1, -1):
        line = lines[idx].strip()
        if not line or line.startswith("*"):
            continue
        if "calendar date" in line.lower() and "x" in line.lower() and "y" in line.lower():
            return line
    raise EphemerisParseError("Baris header kolom Horizons tidak ditemukan.")


def split_csv_like_line(line: str) -> list[str]:
    tokens = next(csv.reader([line], skipinitialspace=True))
    return [token.strip() for token in tokens if token.strip()]


def fix_horizons_header_tokens(tokens: list[str]) -> list[str]:
    if not tokens:
        return tokens

    first = tokens[0]
    if "calendar date" in first.lower():
        split_first = re.split(r"\s{2,}", first.strip(), maxsplit=1)
        if len(split_first) == 2:
            return [split_first[0], split_first[1], *tokens[1:]]
    return tokens


def parse_horizons_table(lines: list[str]) -> tuple[pd.DataFrame, dict[str, str]]:
    metadata = extract_metadata(lines)
    soe_line_index = next((i for i, line in enumerate(lines) if "$$SOE" in line), None)
    if soe_line_index is None:
        raise EphemerisParseError("Marker `$$SOE` tidak ditemukan.")

    header_line = find_horizons_header_line(lines, soe_line_index)
    raw_columns = fix_horizons_header_tokens(split_csv_like_line(header_line))
    normalized_columns = make_unique([normalize_name(col) for col in raw_columns])

    data_start, data_end = find_horizons_data_bounds(lines)
    rows: list[list[str]] = []
    for raw_line in lines[data_start:data_end]:
        line = raw_line.strip()
        if not line:
            continue
        tokens = split_csv_like_line(line)
        if len(tokens) < len(normalized_columns):
            raise EphemerisParseError(
                f"Jumlah kolom data lebih sedikit dari header. Line: {raw_line}"
            )
        rows.append(tokens[: len(normalized_columns)])

    if not rows:
        raise EphemerisParseError("Blok data Horizons ditemukan, tetapi tidak berisi record.")

    table = pd.DataFrame(rows, columns=normalized_columns)
    metadata["_raw_columns"] = ", ".join(raw_columns)
    metadata["_normalized_columns"] = ", ".join(normalized_columns)
    metadata["_parser"] = "horizons"
    return table, metadata


def parse_generic_table(path: Path, cfg: ManualParserConfig) -> tuple[pd.DataFrame, dict[str, str]]:
    if not cfg.enabled:
        raise EphemerisParseError(
            "Auto-parser Horizons gagal dan parser manual belum diaktifkan."
        )

    read_kwargs = {
        "sep": cfg.delimiter,
        "skiprows": cfg.skiprows,
        "comment": cfg.comment,
        "engine": "python",
    }
    if cfg.column_names is not None:
        read_kwargs["names"] = cfg.column_names
        read_kwargs["header"] = None

    table = pd.read_csv(path, **read_kwargs)
    table.columns = make_unique([normalize_name(str(col)) for col in table.columns])

    metadata = {
        "_parser": "generic",
        "_normalized_columns": ", ".join(table.columns),
    }
    return table, metadata


def load_raw_ephemeris(path: Path, config: AppConfig) -> tuple[pd.DataFrame, dict[str, str]]:
    lines = read_text_lines(path)
    try:
        return parse_horizons_table(lines)
    except EphemerisParseError as exc:
        print(f"[WARN] Auto-parser Horizons gagal untuk {path.name}: {exc}")
        return parse_generic_table(path, config.manual_parser)


def choose_column(columns: Iterable[str], patterns: Iterable[str]) -> str | None:
    normalized_columns = list(columns)
    cleaned_patterns = [pattern for pattern in patterns if pattern]
    for pattern in cleaned_patterns:
        for column in normalized_columns:
            if column == pattern:
                return column
    for pattern in cleaned_patterns:
        for column in normalized_columns:
            if pattern in column:
                return column
    return None


def parse_single_datetime(value: str) -> pd.Timestamp:
    cleaned = str(value).strip()
    cleaned = cleaned.replace("A.D.", "").strip()
    if cleaned.startswith("B.C."):
        raise EphemerisParseError(
            "Tanggal B.C. tidak didukung oleh parser datetime sederhana ini."
        )

    formats = [
        "%Y-%b-%d %H:%M:%S.%f",
        "%Y-%b-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    ]
    for fmt in formats:
        parsed = pd.to_datetime(cleaned, format=fmt, errors="coerce")
        if pd.notna(parsed):
            return parsed

    parsed = pd.to_datetime(cleaned, errors="coerce")
    if pd.isna(parsed):
        raise EphemerisParseError(f"Gagal mem-parse waktu: {value!r}")
    return parsed


def detect_time_scale(table: pd.DataFrame, metadata: dict[str, str]) -> str:
    raw_columns = metadata.get("_raw_columns", "").lower()
    calendar_match = re.search(r"calendar date \(([^)]+)\)", raw_columns)
    if calendar_match:
        return calendar_match.group(1).strip().upper()

    jd_column = choose_column(table.columns, ["jdtdb", "jdutc", "jdut", "jd"])
    if jd_column:
        upper = jd_column.upper()
        if upper.startswith("JD") and len(upper) > 2:
            return upper[2:]

    start_time = metadata.get("Start time", "")
    match = re.search(r"\b([A-Z]{2,4})$", start_time)
    if match:
        return match.group(1).upper()

    return "UNKNOWN"


def standardize_ephemeris(
    table: pd.DataFrame,
    metadata: dict[str, str],
    label: str,
    config: AppConfig,
) -> tuple[pd.DataFrame, dict[str, str]]:
    columns = list(table.columns)

    x_col = choose_column(columns, ["x"])
    y_col = choose_column(columns, ["y"])
    z_col = choose_column(columns, ["z"])

    time_col = choose_column(
        columns,
        [
            "calendar_date_tdb",
            "calendar_date_utc",
            "calendar_date",
            "time",
            "epoch",
            normalize_name(config.manual_parser.time_column),
        ],
    )
    jd_col = choose_column(
        columns,
        [
            "jdtdb",
            "jdutc",
            "jdut",
            normalize_name(config.manual_parser.julian_day_column or ""),
            "jd",
        ],
    )
    vx_col = choose_column(columns, ["vx"])
    vy_col = choose_column(columns, ["vy"])
    vz_col = choose_column(columns, ["vz"])

    missing = [name for name, col in [("x", x_col), ("y", y_col), ("z", z_col)] if col is None]
    if missing:
        raise EphemerisParseError(
            f"Kolom koordinat wajib tidak ditemukan pada {label}: {', '.join(missing)}"
        )

    if time_col is None and jd_col is None:
        raise EphemerisParseError(
            f"Tidak ditemukan kolom waktu string atau Julian day pada {label}."
        )

    standardized = pd.DataFrame()
    standardized["x"] = pd.to_numeric(table[x_col], errors="coerce")
    standardized["y"] = pd.to_numeric(table[y_col], errors="coerce")
    standardized["z"] = pd.to_numeric(table[z_col], errors="coerce")

    if vx_col is not None:
        standardized["vx"] = pd.to_numeric(table[vx_col], errors="coerce")
    if vy_col is not None:
        standardized["vy"] = pd.to_numeric(table[vy_col], errors="coerce")
    if vz_col is not None:
        standardized["vz"] = pd.to_numeric(table[vz_col], errors="coerce")

    if jd_col is not None:
        standardized["jd"] = pd.to_numeric(table[jd_col], errors="coerce")

    if time_col is not None:
        standardized["time"] = table[time_col].map(parse_single_datetime)
    elif "jd" in standardized.columns:
        standardized["time"] = pd.to_datetime(
            standardized["jd"],
            unit="D",
            origin="julian",
            errors="coerce",
        )
        warnings.warn(
            f"{label}: waktu dibangun dari Julian day karena kolom calendar date tidak ada."
        )

    standardized = standardized.dropna(subset=["time", "x", "y", "z"]).copy()
    standardized = standardized.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)

    if standardized.empty:
        raise EphemerisParseError(f"Data standar untuk {label} kosong setelah pembersihan.")

    standardized["t_seconds"] = standardized["time"].astype("int64") / 1e9

    metadata["_time_scale"] = detect_time_scale(table, metadata)
    standardized.attrs["label"] = label
    standardized.attrs["time_scale"] = metadata["_time_scale"]
    metadata["_x_col"] = x_col
    metadata["_y_col"] = y_col
    metadata["_z_col"] = z_col
    metadata["_time_col"] = time_col or ""
    metadata["_jd_col"] = jd_col or ""

    required_output_columns = ["time", "x", "y", "z"]
    standardized = standardized[required_output_columns + [col for col in standardized.columns if col not in required_output_columns]]
    return standardized, metadata


def print_dataframe_report(label: str, df: pd.DataFrame, metadata: dict[str, str]) -> None:
    print(f"\n===== {label.upper()} =====")
    print(f"Parser            : {metadata.get('_parser', 'unknown')}")
    print(f"Time scale        : {metadata.get('_time_scale', 'UNKNOWN')}")
    print(f"Reference frame   : {metadata.get('Reference frame', 'UNKNOWN')}")
    print(f"Center body       : {metadata.get('Center body name', 'UNKNOWN')}")
    print(f"Output units      : {metadata.get('Output units', 'UNKNOWN')}")
    print(f"Normalized columns: {list(df.columns)}")
    print("Head:")
    print(df.head(3).to_string(index=False))


def sanity_check_ephemeris(label: str, df: pd.DataFrame) -> None:
    xyz_are_numeric = all(pd.api.types.is_numeric_dtype(df[col]) for col in ["x", "y", "z"])
    if not xyz_are_numeric:
        raise EphemerisParseError(f"{label}: kolom x/y/z tidak terbaca sebagai numerik.")
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        raise EphemerisParseError(f"{label}: kolom time gagal diparse menjadi datetime.")
    if not df["time"].is_monotonic_increasing:
        raise EphemerisParseError(f"{label}: kolom time tidak terurut naik.")


def compare_metadata(
    moon_meta: dict[str, str],
    artemis_meta: dict[str, str],
    strict: bool,
) -> None:
    checks = {
        "Center body name": "pusat koordinat",
        "Reference frame": "reference frame",
        "Output units": "satuan output",
        "_time_scale": "skala waktu",
    }

    mismatches: list[str] = []
    for key, description in checks.items():
        left = moon_meta.get(key, "UNKNOWN")
        right = artemis_meta.get(key, "UNKNOWN")
        if left != right:
            mismatches.append(f"- {description}: Moon={left!r}, Artemis={right!r}")

    if mismatches:
        message = "Ketidaksesuaian metadata terdeteksi:\n" + "\n".join(mismatches)
        if strict:
            raise EphemerisParseError(message)
        print(f"[WARN] {message}")


def interpolation_time_axis(df: pd.DataFrame) -> np.ndarray:
    if "jd" in df.columns and df["jd"].notna().all():
        return df["jd"].to_numpy(dtype=float)
    return df["t_seconds"].to_numpy(dtype=float)


def interpolate_to_target_timebase(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    source_t = interpolation_time_axis(source_df)
    target_t = interpolation_time_axis(target_df)

    if target_t.min() < source_t.min() or target_t.max() > source_t.max():
        raise EphemerisParseError(
            f"Interpolasi {label} gagal karena range waktunya tidak menutup target timebase."
        )

    interpolated = pd.DataFrame({"time": target_df["time"].copy()})
    for column in ["x", "y", "z"]:
        interpolated[column] = np.interp(target_t, source_t, source_df[column].to_numpy(dtype=float))

    for optional_column in ["vx", "vy", "vz"]:
        if optional_column in source_df.columns:
            interpolated[optional_column] = np.interp(
                target_t,
                source_t,
                source_df[optional_column].to_numpy(dtype=float),
            )

    if "jd" in target_df.columns:
        interpolated["jd"] = target_df["jd"].to_numpy(dtype=float)
    interpolated["t_seconds"] = target_df["t_seconds"].to_numpy(dtype=float)
    interpolated.attrs["label"] = source_df.attrs.get("label", label)
    interpolated.attrs["time_scale"] = source_df.attrs.get("time_scale", "")
    return interpolated


def align_timebases(
    moon_df: pd.DataFrame,
    artemis_df: pd.DataFrame,
    prefer: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    moon_t = interpolation_time_axis(moon_df)
    artemis_t = interpolation_time_axis(artemis_df)

    same_count = len(moon_df) == len(artemis_df)
    same_time = same_count and np.allclose(moon_t, artemis_t, rtol=0, atol=1e-9)

    print(f"\nJumlah timestamp Moon    : {len(moon_df)}")
    print(f"Jumlah timestamp Artemis : {len(artemis_df)}")

    if same_time:
        print("Status timebase          : Sama, interpolasi tidak diperlukan.")
        return moon_df.copy(), artemis_df.copy()

    if prefer.lower() == "moon":
        print("Status timebase          : Berbeda, Artemis diinterpolasi ke timebase Moon.")
        aligned_artemis = interpolate_to_target_timebase(artemis_df, moon_df, "Artemis")
        return moon_df.copy(), aligned_artemis

    print("Status timebase          : Berbeda, Moon diinterpolasi ke timebase Artemis.")
    aligned_moon = interpolate_to_target_timebase(moon_df, artemis_df, "Moon")
    return aligned_moon, artemis_df.copy()


def add_distance_columns(artemis_df: pd.DataFrame, moon_df: pd.DataFrame) -> pd.DataFrame:
    result = artemis_df.copy()
    artemis_xyz = result[["x", "y", "z"]].to_numpy(dtype=float)
    moon_xyz = moon_df[["x", "y", "z"]].to_numpy(dtype=float)

    result["distance_to_earth_km"] = np.linalg.norm(artemis_xyz, axis=1)
    result["distance_to_moon_km"] = np.linalg.norm(artemis_xyz - moon_xyz, axis=1)
    return result


def print_distance_summary(artemis_df: pd.DataFrame) -> None:
    print("\nRingkasan jarak Artemis:")
    print(
        "Distance to Earth (km): "
        f"min={artemis_df['distance_to_earth_km'].min():,.3f} | "
        f"max={artemis_df['distance_to_earth_km'].max():,.3f}"
    )
    print(
        "Distance to Moon  (km): "
        f"min={artemis_df['distance_to_moon_km'].min():,.3f} | "
        f"max={artemis_df['distance_to_moon_km'].max():,.3f}"
    )


def plane_values(df: pd.DataFrame, axes: tuple[str, str]) -> tuple[np.ndarray, np.ndarray]:
    ax1, ax2 = axes
    return df[ax1].to_numpy(dtype=float), df[ax2].to_numpy(dtype=float)


def compute_plot_limits(
    moon_df: pd.DataFrame,
    artemis_df: pd.DataFrame,
    axes: tuple[str, str],
) -> tuple[tuple[float, float], tuple[float, float]]:
    moon_x, moon_y = plane_values(moon_df, axes)
    art_x, art_y = plane_values(artemis_df, axes)

    all_x = np.concatenate([moon_x, art_x, np.array([0.0])])
    all_y = np.concatenate([moon_y, art_y, np.array([0.0])])
    pad_x = 0.08 * max(all_x.max() - all_x.min(), 1.0)
    pad_y = 0.08 * max(all_y.max() - all_y.min(), 1.0)

    return (all_x.min() - pad_x, all_x.max() + pad_x), (all_y.min() - pad_y, all_y.max() + pad_y)


def plot_static_2d(
    moon_df: pd.DataFrame,
    artemis_df: pd.DataFrame,
    config: AppConfig,
    metadata: dict[str, str],
) -> tuple[plt.Figure, Path | None]:
    axes = config.plot_axes
    xlim, ylim = compute_plot_limits(moon_df, artemis_df, axes)

    fig, ax = plt.subplots(figsize=(10, 10))
    moon_x, moon_y = plane_values(moon_df, axes)
    art_x, art_y = plane_values(artemis_df, axes)

    ax.plot(moon_x, moon_y, color="slateblue", lw=1.8, label="Moon trajectory")
    ax.plot(art_x, art_y, color="tomato", lw=1.8, label="Artemis II trajectory")
    ax.scatter(0.0, 0.0, color="royalblue", s=140, label="Earth", zorder=5)
    ax.scatter(moon_x[0], moon_y[0], color="navy", s=30, label="Moon start", zorder=6)
    ax.scatter(art_x[0], art_y[0], color="darkred", s=30, label="Artemis start", zorder=6)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel(AXIS_LABELS[axes[0]])
    ax.set_ylabel(AXIS_LABELS[axes[1]])
    ax.set_title(
        "Artemis II Mission Geometry\n"
        f"{metadata.get('Reference frame', 'Unknown frame')} | "
        f"{metadata.get('_time_scale', 'Unknown time scale')}"
    )
    ax.legend(loc="best")
    fig.tight_layout()

    saved_path: Path | None = None
    if config.save_static_plot:
        saved_path = config.output_dir / config.static_plot_name
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(saved_path, dpi=config.figure_dpi, bbox_inches="tight")
        print(f"Static plot tersimpan   : {saved_path}")

    return fig, saved_path


def save_animation(anim: FuncAnimation, output_path: Path, fps: int, dpi: int) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix == ".mp4" and not FFMPEG_PATH:
        fallback_path = output_path.with_suffix(".gif")
        print(
            f"[WARN] {FFMPEG_WARNING} "
            "Output MP4 diminta tetapi ffmpeg tidak usable. "
            f"Menyimpan GIF sebagai {fallback_path.name}"
        )
        writer = PillowWriter(fps=fps)
        anim.save(fallback_path, writer=writer, dpi=dpi)
        return fallback_path

    try:
        if suffix == ".gif":
            writer = PillowWriter(fps=fps)
            anim.save(output_path, writer=writer, dpi=dpi)
            return output_path
        if suffix == ".mp4":
            print(f"FFmpeg path             : {FFMPEG_PATH}")
            writer = FFMpegWriter(fps=fps)
            anim.save(output_path, writer=writer, dpi=dpi)
            return output_path
        raise ValueError(f"Ekstensi animasi tidak didukung: {suffix}")
    except Exception as exc:
        if suffix == ".mp4":
            fallback_path = output_path.with_suffix(".gif")
            print(
                f"[WARN] Gagal menyimpan MP4 ({exc}). "
                f"Mencoba fallback GIF: {fallback_path.name}"
            )
            writer = PillowWriter(fps=fps)
            anim.save(fallback_path, writer=writer, dpi=dpi)
            return fallback_path
        raise


def animate_2d(
    moon_df: pd.DataFrame,
    artemis_df: pd.DataFrame,
    config: AppConfig,
) -> tuple[plt.Figure, FuncAnimation, Path | None]:
    axes = config.plot_axes
    xlim, ylim = compute_plot_limits(moon_df, artemis_df, axes)

    moon_x, moon_y = plane_values(moon_df, axes)
    art_x, art_y = plane_values(artemis_df, axes)
    frames = list(range(0, len(artemis_df), max(1, config.animation_frame_step)))
    if frames[-1] != len(artemis_df) - 1:
        frames.append(len(artemis_df) - 1)
    use_blit = config.animation_blit_interactive if config.show_plots else config.animation_blit_offscreen
    print(f"Animation blit          : {use_blit}")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel(AXIS_LABELS[axes[0]])
    ax.set_ylabel(AXIS_LABELS[axes[1]])
    ax.set_title("Artemis II 2D Animation")

    ax.plot(moon_x, moon_y, color="slateblue", lw=1.0, alpha=0.35, label="Moon trajectory")
    ax.scatter(0.0, 0.0, color="royalblue", s=140, label="Earth", zorder=5)

    moon_marker, = ax.plot([], [], marker="o", color="navy", markersize=8, linestyle="None", label="Moon")
    artemis_marker, = ax.plot([], [], marker="o", color="tomato", markersize=7, linestyle="None", label="Artemis II")
    artemis_trail, = ax.plot([], [], color="tomato", lw=2.0, alpha=0.9, label="Artemis trail")

    timestamp_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=10)
    distance_text = ax.text(0.02, 0.90, "", transform=ax.transAxes, va="top", fontsize=10)
    ax.legend(loc="best")

    def init() -> tuple:
        moon_marker.set_data([], [])
        artemis_marker.set_data([], [])
        artemis_trail.set_data([], [])
        timestamp_text.set_text("")
        distance_text.set_text("")
        return moon_marker, artemis_marker, artemis_trail, timestamp_text, distance_text

    def update(frame_index: int) -> tuple:
        moon_marker.set_data([moon_x[frame_index]], [moon_y[frame_index]])
        artemis_marker.set_data([art_x[frame_index]], [art_y[frame_index]])
        artemis_trail.set_data(art_x[: frame_index + 1], art_y[: frame_index + 1])

        current_time = artemis_df["time"].iloc[frame_index]
        time_scale = artemis_df.attrs.get("time_scale", "").strip()
        time_suffix = f" {time_scale}" if time_scale else ""
        timestamp_text.set_text(f"Epoch: {current_time:%Y-%m-%d %H:%M:%S}{time_suffix}")
        distance_text.set_text(
            "Distance to Earth: "
            f"{artemis_df['distance_to_earth_km'].iloc[frame_index]:,.1f} km\n"
            "Distance to Moon:  "
            f"{artemis_df['distance_to_moon_km'].iloc[frame_index]:,.1f} km"
        )
        return moon_marker, artemis_marker, artemis_trail, timestamp_text, distance_text

    anim = FuncAnimation(
        fig,
        update,
        frames=frames,
        init_func=init,
        interval=config.animation_interval_ms,
        blit=use_blit,
        repeat=False,
    )

    saved_path: Path | None = None
    should_save_animation = config.save_animation and (
        not config.show_plots or config.save_animation_when_interactive
    )

    if should_save_animation:
        saved_path = save_animation(
            anim,
            config.output_dir / config.animation_name,
            fps=config.animation_fps,
            dpi=config.figure_dpi,
        )
        print(f"Animasi tersimpan       : {saved_path}")
    elif config.save_animation and config.show_plots:
        print(
            "Animasi file tidak disimpan pada mode interaktif agar jendela tampil segera. "
            "Set `ARTEMIS_SAVE_ANIMATION_IN_INTERACTIVE=1` jika tetap ingin menyimpan saat mode interaktif."
        )

    return fig, anim, saved_path


def build_relative_frame(target_df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    """
    Menggeser origin ke pusat objek referensi tanpa mengubah orientasi sumbu.

    Untuk kasus Artemis vs Moon, hasilnya adalah koordinat Moon-centered
    translation-only: origin di pusat Bulan, tetapi sumbu tetap sumbu inertial
    semula (mis. Ecliptic of J2000.0), bukan frame lokal Bulan yang berotasi.
    """
    relative = target_df.copy()
    relative[["x", "y", "z"]] = (
        target_df[["x", "y", "z"]].to_numpy(dtype=float)
        - reference_df[["x", "y", "z"]].to_numpy(dtype=float)
    )
    return relative


def main(config: AppConfig) -> None:
    print("Membaca file ephemeris...")
    runtime_backend = prepare_runtime_backend(config.show_plots)
    print(f"Matplotlib backend      : {runtime_backend}")
    moon_raw, moon_meta = load_raw_ephemeris(config.moon_file, config)
    artemis_raw, artemis_meta = load_raw_ephemeris(config.artemis_file, config)

    moon_df, moon_meta = standardize_ephemeris(moon_raw, moon_meta, "Moon", config)
    artemis_df, artemis_meta = standardize_ephemeris(artemis_raw, artemis_meta, "Artemis II", config)

    print_dataframe_report("Moon", moon_df, moon_meta)
    print_dataframe_report("Artemis II", artemis_df, artemis_meta)

    sanity_check_ephemeris("Moon", moon_df)
    sanity_check_ephemeris("Artemis II", artemis_df)
    compare_metadata(moon_meta, artemis_meta, strict=config.strict_metadata_checks)

    moon_df, artemis_df = align_timebases(
        moon_df,
        artemis_df,
        prefer=config.prefer_timebase_from,
    )
    artemis_df = add_distance_columns(artemis_df, moon_df)
    print_distance_summary(artemis_df)

    static_fig, _ = plot_static_2d(moon_df, artemis_df, config, artemis_meta)
    animation_fig, animation_obj, _ = animate_2d(moon_df, artemis_df, config)

    moon_relative_artemis = build_relative_frame(artemis_df, moon_df)
    print(
        "\nKoordinat Moon-centered (translation only) sudah dibuat untuk pengembangan lanjutan. "
        f"Jarak Artemis ke pusat Bulan pada epoch pertama: "
        f"{np.linalg.norm(moon_relative_artemis[['x', 'y', 'z']].iloc[0]):,.3f} km"
    )

    if config.show_plots:
        print("Menampilkan jendela interaktif...")
        plt.show()
    else:
        plt.close(static_fig)
        plt.close(animation_fig)

    # Menjaga referensi agar animasi tidak terhapus sebelum plt.show() selesai.
    _ = animation_obj


if __name__ == "__main__":
    try:
        main(CONFIG)
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        print(
            "Periksa konfigurasi file, delimiter, nama kolom, dan metadata ephemeris. "
            "Jika file Anda bukan format Horizons standar, aktifkan `ManualParserConfig`."
        )
        raise
