#!/usr/bin/env python3
"""Prepare publication figures without recreating experimental results.

Python >= 3.10. Install dependencies: python -m pip install Pillow matplotlib
Place this script beside the exact source PNGs and run:
    python prepare_final_paper_figures.py

Saved numerical figure/table modes (manual execution only):
    python prepare_final_paper_figures.py --final-test-results
    python prepare_final_paper_figures.py --complete-paper-figures
    python prepare_final_paper_figures.py --project-final-test-figures
The complete mode also requires NumPy (a Matplotlib dependency). New numerical
plots use solid fills and borderless imshow heatmaps. Complete mode draws a new
vector methodology; default raster preparation remains available separately.
LaTeX complete-library tables need booktabs and longtable.

Outputs go beside the script, in final_paper_figures/. Existing outputs with
these names are replaced; source files are never modified. No OCR, tracing,
metric extraction, or scientific text replacement is performed.
"""

from __future__ import annotations

import base64
import argparse
import csv
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP, localcontext
import hashlib
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image, ImageChops, ImageDraw, ImageFont
import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager, rc_context
from matplotlib.figure import Figure


TRIM_PADDING = 24
# Exact white is the safest default: even faint near-white marks are retained.
# Lowering this threshold explicitly permits removal of near-white pixels.
WHITE_THRESHOLD = 255
PNG_DPI = 600
PAPER_WIDTH_INCHES = 7.16
FIG06_COLUMNS = 2  # Default 2 x 3; set to 3 after inspecting the preview.
SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)
WARNINGS: list[str] = []
FONT_PATH = font_manager.findfont(
    font_manager.FontProperties(family="DejaVu Sans", weight="bold")
)

# Order in each tuple is the panel order; no alternate filename lookup.
SPECS = [
    ("Fig01_Complete_Methodology_Architecture", "Complete Methodology",
     ("fig01_complete_methodology_architecture.png",)),
    ("Fig02_Flow_Extraction_and_Preparation", "Flow Preparation",
     ("flow_extraction_combined(2).png",)),
    ("Fig03_PhaseI_Model_Comparison", "Model Comparison",
     ("phase1_model_comparison.png",)),
    ("Fig04_Selected_LightGBM_Confusion_Matrices", "LightGBM Confusions",
     ("phase1_best_model_confusions.png",)),
    ("Fig05_PhaseII_Provenance_Aware_RAG_Pipeline", "Phase-II Pipeline",
     ("phase2_overall_pipeline(1).png",)),
    ("Fig06_PhaseII_Evidence_Processing", "Evidence Processing",
     ("S2_closed_test(1).png", "S3_closed_test(1).png", "S4_closed_test.png",
      "S5_closed_test.png", "S6_closed_test.png", "S7_closed_test.png")),
    ("Fig07_PhaseII_Evidence_and_Decision_Analysis", "Evidence & Decision Analysis",
     ("F_test_subset_evidence(1).png", "F_test_subset_correctness(1).png",
      "phase2_stage8_to_stage10_summary.png")),
    ("Fig08_PhaseI_Combined_ROC", "ROC Comparison",
     ("phase1_combined_roc.png",)),
]


def warn(message: str) -> None:
    WARNINGS.append(message)
    print(f"[WARNING] {message}")


def load_image(path: Path) -> Image.Image:
    """Decode PNG and composite transparency onto white without rescaling."""
    with Image.open(path) as source:
        if source.format != "PNG" or getattr(source, "n_frames", 1) != 1:
            raise ValueError("Expected a single-frame PNG")
        source.load()
        rgba = source.convert("RGBA")
        return Image.alpha_composite(
            Image.new("RGBA", rgba.size, "white"), rgba
        ).convert("RGB")


def trim_whitespace(image: Image.Image, padding: int = TRIM_PADDING):
    """Bound every non-white pixel, retain padding, never remove internal space."""
    if padding < 0 or not 1 <= WHITE_THRESHOLD <= 255:
        raise ValueError("Invalid trim configuration")
    channels = image.split()
    darkest = ImageChops.darker(ImageChops.darker(channels[0], channels[1]), channels[2])
    mask = darkest.point([255 if value < WHITE_THRESHOLD else 0 for value in range(256)])
    bounds = mask.getbbox()
    if bounds is None:
        return image.copy(), (0, 0, *image.size), False
    x0, y0, x1, y1 = bounds
    edge_content = min(x0, y0, image.width - x1, image.height - y1) < padding
    box = (max(0, x0 - padding), max(0, y0 - padding),
           min(image.width, x1 + padding), min(image.height, y1 + padding))
    return image.crop(box), box, edge_content


@dataclass
class Panel:
    image: Image.Image
    x: float
    y: float
    scale: float = 1.0


@dataclass
class Composition:
    width: int
    height: int
    panels: list[Panel]
    labels: list[tuple[str, float, float]]
    font_size: int = 1


def combine_panels(images: list[Image.Image], figure_number: int) -> Composition:
    """Use uniform scaling only; never downsample panels in manuscript exports."""
    if len(images) == 1:
        im = images[0]
        return Composition(im.width, im.height, [Panel(im, 0, 0)], [])
    columns = FIG06_COLUMNS if figure_number == 6 else 2
    if columns not in (2, 3):
        raise ValueError("FIG06_COLUMNS must be 2 or 3")
    cell_images = images[:2] if figure_number == 7 else images
    cell = max(im.width for im in cell_images)
    gap = max(32, round(cell * 0.04))
    if figure_number == 7:
        cell = max(cell, math.ceil((images[2].width - gap) / 2))
    width = columns * cell + (columns + 1) * gap
    font_size = max(24, round(width * 10 / (PAPER_WIDTH_INCHES * 72)))
    header = math.ceil(font_size * 1.8)
    panels, labels = [], []
    rows = [[0, 1], [2]] if figure_number == 7 else [list(range(i, min(i + columns, len(images)))) for i in range(0, len(images), columns)]
    y = gap
    for row in rows:
        row_height = 0
        for col, index in enumerate(row):
            im = images[index]
            available = width - 2 * gap if figure_number == 7 and index == 2 else cell
            scale = available / im.width
            x = gap + col * (cell + gap)
            labels.append((f"({chr(97 + index)})", x, y))
            panels.append(Panel(im, x, y + header, scale))
            row_height = max(row_height, math.ceil(im.height * scale))
        y += header + row_height + gap
    return Composition(width, y, panels, labels, font_size)


def create_canvas(figure: Composition) -> Image.Image:
    canvas = Image.new("RGB", (figure.width, figure.height), "white")
    for panel in figure.panels:
        size = (round(panel.image.width * panel.scale), round(panel.image.height * panel.scale))
        image = panel.image if size == panel.image.size else panel.image.resize(size, Image.Resampling.LANCZOS)
        canvas.paste(image, (round(panel.x), round(panel.y)))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT_PATH, figure.font_size)
    for label, x, y in figure.labels:
        draw.text((x, y), label, font=font, fill="black", anchor="lt")
    return canvas


def save_pdf(figure: Composition, path: Path) -> None:
    """Embed each native raster losslessly; retain vector panel labels."""
    height_inches = PAPER_WIDTH_INCHES * figure.height / figure.width
    plot = Figure(figsize=(PAPER_WIDTH_INCHES, height_inches), facecolor="white")
    for panel in figure.panels:
        w, h = panel.image.width * panel.scale, panel.image.height * panel.scale
        axes = plot.add_axes([panel.x / figure.width,
                              1 - (panel.y + h) / figure.height,
                              w / figure.width, h / figure.height])
        axes.imshow(panel.image, interpolation="none", aspect="auto")
        axes.set_axis_off()
    for label, x, y in figure.labels:
        plot.text(x / figure.width, 1 - y / figure.height, label,
                  ha="left", va="top", family="DejaVu Sans", weight="bold",
                  fontsize=figure.font_size * PAPER_WIDTH_INCHES * 72 / figure.width)
    with rc_context({"pdf.compression": 9, "pdf.fonttype": 42,
                     "image.composite_image": False}):
        plot.savefig(path, format="pdf", metadata={"CreationDate": None, "ModDate": None})
    plot.clear()


def save_svg(figure: Composition, path: Path) -> None:
    root = ET.Element(f"{{{SVG_NS}}}svg", {
        "version": "1.1", "width": f"{PAPER_WIDTH_INCHES}in",
        "height": f"{PAPER_WIDTH_INCHES * figure.height / figure.width:.8f}in",
        "viewBox": f"0 0 {figure.width} {figure.height}",
        "preserveAspectRatio": "xMidYMid meet"})
    ET.SubElement(root, f"{{{SVG_NS}}}rect", {
        "width": "100%", "height": "100%", "fill": "white"})
    for panel in figure.panels:
        buffer = io.BytesIO()
        panel.image.save(buffer, format="PNG")
        ET.SubElement(root, f"{{{SVG_NS}}}image", {
            "x": str(panel.x), "y": str(panel.y),
            "width": str(panel.image.width * panel.scale),
            "height": str(panel.image.height * panel.scale),
            "preserveAspectRatio": "xMidYMid meet",
            f"{{{XLINK_NS}}}href": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")})
    for label, x, y in figure.labels:
        node = ET.SubElement(root, f"{{{SVG_NS}}}text", {
            "x": str(x), "y": str(y + figure.font_size),
            "font-family": "DejaVu Sans, sans-serif", "font-weight": "bold",
            "font-size": str(figure.font_size), "fill": "black"})
        node.text = label
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def validate_output(path: Path, expected_size: tuple[int, int]) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError("Missing or empty output")
    if path.suffix == ".png":
        with Image.open(path) as image:
            image.load()
            if image.format != "PNG" or image.size != expected_size or min(image.size) < 1:
                raise ValueError("Invalid PNG dimensions or format")
    elif path.suffix == ".pdf":
        with path.open("rb") as stream:
            if stream.read(5) != b"%PDF-":
                raise ValueError("Invalid PDF header")
    else:
        root = ET.parse(path).getroot()
        if root.tag != f"{{{SVG_NS}}}svg" or root.get("viewBox") != f"0 0 {expected_size[0]} {expected_size[1]}":
            raise ValueError("Invalid SVG structure")
        if not root.findall(f"{{{SVG_NS}}}image"):
            raise ValueError("SVG contains no source images")


def save_all_formats(figure: Composition, basename: Path) -> dict[str, bool]:
    """Validate temporary exports before atomically replacing each final file."""
    status = {}
    for extension in ("png", "pdf", "svg"):
        destination = basename.with_suffix("." + extension)
        temporary = destination.with_name(destination.stem + ".pending." + extension)
        try:
            if destination.is_symlink() or temporary.is_symlink():
                raise ValueError("Refusing a symbolic-link output")
            if extension == "png":
                create_canvas(figure).save(temporary, format="PNG", dpi=(PNG_DPI, PNG_DPI))
            elif extension == "pdf":
                save_pdf(figure, temporary)
            else:
                save_svg(figure, temporary)
            validate_output(temporary, (figure.width, figure.height))
            temporary.replace(destination)
            status[extension] = True
        except Exception as exc:
            status[extension] = False
            warn(f"{basename.name}: {extension.upper()} export failed: {exc}")
        finally:
            if temporary.is_file() and not temporary.is_symlink():
                temporary.unlink()
    print(f"[{'OK' if all(status.values()) else 'FAILED'}] {basename.name}")
    for extension, ok in status.items():
        print(f"     {extension.upper():3} : {'OK' if ok else 'FAILED'}")
    print(f"     Size: {figure.width} x {figure.height} px")
    return status


def create_preview_sheet(entries, output_dir: Path) -> None:
    if not entries:
        warn("No successful figures available for the preview sheet.")
        return
    tile_w, tile_h, margin = 900, 650, 25
    canvas = Image.new("RGB", (2 * tile_w, math.ceil(len(entries) / 2) * tile_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT_PATH, 25)
    for index, (number, title, path) in enumerate(entries):
        x, y = (index % 2) * tile_w, (index // 2) * tile_h
        draw.text((x + margin, y + margin), f"Fig. {number} — {title}", fill="black", font=font)
        with Image.open(path) as source:
            image = source.convert("RGB")
        image.thumbnail((tile_w - 2 * margin, tile_h - 90), Image.Resampling.LANCZOS)
        canvas.paste(image, (x + (tile_w - image.width) // 2, y + 70))
    path = output_dir / "Final_Paper_Figures_Preview.png"
    if path.is_symlink():
        raise ValueError("Refusing a symbolic-link preview output")
    canvas.save(path, dpi=(150, 150))
    validate_output(path, canvas.size)


def main() -> int:
    WARNINGS.clear()
    source_dir = Path(__file__).resolve().parent
    output_dir = source_dir / "final_paper_figures"
    if output_dir.is_symlink():
        raise ValueError("Output directory must not be a symbolic link")
    output_dir.mkdir(exist_ok=True)
    missing = set()
    for _, _, names in SPECS:
        for name in names:
            if not (source_dir / name).is_file():
                missing.add(name)
                warn(f"Missing source: {name}; dependent figure will be skipped.")
    warn("Manual wording review required for Figures 1, 5 and 7: raster text is not "
         "automatically verified. If Stage 8-10 is described as a development-validation "
         "subset, manually correct the source before submission to: "
         "Prediction-stratified held-out TEST subset; n = 1,200; seed = 42. "
         "These are unweighted subset results, not full-TEST population LLM metrics. "
         "Figure 5 must distinguish the full held-out TEST population (n = 129,978), "
         "frozen split-independent corpus/index resources, and the later subset.")
    warn("Manual full-width inspection is required: source text readability and "
         "scientific wording cannot be guaranteed by pixel dimensions or DPI metadata. "
         "LightGBM selection must be described as development/validation selection.")
    summary, previews, audit = [], [], []
    for number, (basename, title, names) in enumerate(SPECS, 1):
        status = dict.fromkeys(("png", "pdf", "svg"), False)
        if any(name in missing for name in names):
            summary.append((number, status, "SKIPPED"))
            continue
        try:
            images = []
            for name in names:
                path = source_dir / name
                original = load_image(path)
                cropped, box, edge = trim_whitespace(original)
                if edge:
                    warn(f"{name}: content approaches the original edge; inspect for pre-existing clipping.")
                if min(original.size) < 600:
                    warn(f"{name}: unusually small source resolution {original.size}.")
                audit.append({"source": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                              "original_size": original.size, "crop_box": box})
                images.append(cropped)
            figure = combine_panels(images, number)
            for name, panel in zip(names, figure.panels):
                effective_dpi = figure.width / (PAPER_WIDTH_INCHES * panel.scale)
                if effective_dpi < 300:
                    warn(f"{name}: native resolution at intended placement is {effective_dpi:.0f} DPI; "
                         "upscaling or 600-DPI metadata cannot add source detail.")
            if figure.height / figure.width * PAPER_WIDTH_INCHES > 9:
                warn(f"{basename}: taller than 9 inches at full width; inspect page fit/readability.")
            status = save_all_formats(figure, output_dir / basename)
            if all(status.values()):
                previews.append((number, title, output_dir / (basename + ".png")))
            summary.append((number, status, "SUCCESS" if all(status.values()) else "FAILED"))
        except Exception as exc:
            warn(f"{basename}: preparation failed: {exc}")
            summary.append((number, status, "FAILED"))
    if not any(number == 8 for number, _, _ in previews):
        warn("Optional Figure 8 unavailable or incomplete; omitted from preview.")
    preview_ok = False
    try:
        create_preview_sheet(previews, output_dir)
        preview_ok = bool(previews)
    except Exception as exc:
        warn(f"Preview export failed: {exc}")
    manifest = output_dir / "Preparation_Report.json"
    if not manifest.is_symlink():
        manifest.write_text(json.dumps({"sources": audit, "warnings": WARNINGS,
            "summary": summary, "figure6_columns": FIG06_COLUMNS,
            "trim_padding": TRIM_PADDING, "white_threshold": WHITE_THRESHOLD}, indent=2), encoding="utf-8")
    print("\nFINAL PAPER FIGURE EXPORT SUMMARY\nFigure   PNG    PDF    SVG    Status")
    for number, status, result in summary:
        flags = "  ".join(f"{'OK' if status[ext] else '--':5}" for ext in ("png", "pdf", "svg"))
        print(f"Fig{number:02}    {flags}  {result}")
    required_ok = all(result == "SUCCESS" for number, _, result in summary if number <= 7)
    optional_ok = summary[-1][2] == "SUCCESS" or "phase1_combined_roc.png" in missing
    if required_ok and optional_ok and preview_ok:
        print("\nFinal paper figures generated successfully.")
    else:
        print("\nExport incomplete; see skipped/failed figures above.")
    print(f"Output directory: {output_dir}")
    if WARNINGS:
        print("\nWarnings:")
        for message in WARNINGS:
            print(f"- {message}")
    print("Only SUCCESS rows refer to complete exports from this run; older outputs may remain.")
    return 0 if required_ok and optional_ok and preview_ok else 1


# Isolated, read-only saved-result publication mode. The raster workflow above
# remains unchanged. No research modules, models, or inference clients are used.
STATUSES = ("SUPPORTED", "CONTRADICTED", "INSUFFICIENT", "CONFLICTING")
STATUS_COLORS = ("#009E73", "#D55E00", "#7F7F7F", "#E69F00")
DATASETS = ("USTC-TFC2016", "CIC-Darknet2020")
CLASS_KEYS = tuple((DATASETS[0], c) for c in ("benign", "malware")) + tuple(
    (DATASETS[1], c) for c in ("Non-Tor", "NonVPN", "Tor", "VPN"))
TEST_NAMESPACE = Path("results/rag/test/final_4300_v5_contract_v2")
FIGURE_NAMES = (
    "R1_Final_TEST_Confusion_Matrices", "R2_Final_TEST_Per_Class_Performance",
    "R3_TEST_Evidence_Status_and_Correctness", "R4_V4_V5_Matched_Evidence_Comparison",
    "R5_TEST_Provenance_Coverage", "S1_Validation_vs_TEST_Evidence_Status",
    "S2_TEST_Class_Evidence_Matrix")
TABLE_NAMES = (
    "Table_R1_Final_TEST_Performance", "Table_R2_Per_Class_TEST_Performance",
    "Table_R3_Evidence_Status_Correctness", "Table_R4_Provenance_Coverage",
    "Table_R5_V4_V5_Matched_Comparison", "Table_S1_TEST_Cohort",
    "Table_S2_Validation_TEST_Evidence")
CAPTIONS = (
    "Final TEST confusion matrices for frozen Phase-I predictions. Cell colors are "
    "row-normalized; annotations show raw counts only. Phase-II evidence "
    "reasoning does not override these predictions.",
    "Per-class precision, recall and F1 on the fixed historical TEST cohort; support "
    "is taken from saved evaluation artifacts. The performance axis begins at 90%.",
    "Evidence-status distribution and classifier accuracy conditioned on status. "
    "SUPPORTED / CONTRADICTED / INSUFFICIENT / CONFLICTING describe compatibility "
    "with qualifying supplied evidence. They do not denote prediction correctness. "
    "Evidence states are subject to the response schema and validator constraints.",
    "V4 and V5 are compared on the exact same 4,300 historical TEST IDs. Final "
    "classifier predictions remain unchanged. The figure reflects evidence-"
    "characterization changes, not classifier accuracy improvement.",
    "Final provenance availability is distinct from retrieval/context coverage; "
    "all queries received trusted contexts, while 3,425 produced nonempty final "
    "provenance mappings. Context chunk occurrences and selected citation "
    "occurrences are separate quantities with separate denominators.",
    "Evidence distributions for different historical validation and TEST cohorts; "
    "observations are not paired.",
    "Final predicted class by evidence status in the historical TEST cohort. "
    "Colors show within-class proportions; annotations show counts and percentages.")


def require(condition, message):
    """An assertion that remains active even when Python is run with -O."""
    if not condition:
        raise ValueError("Frozen TEST publication validation failed: " + message)


def format_percent(value, decimals=2):
    """Display only: never round a genuinely sub-one value to 100%."""
    d = Decimal(str(value))
    require(d.is_finite(), "non-finite percentage")
    require(isinstance(decimals, int) and decimals >= 0, "invalid decimal places")
    with localcontext() as context:
        context.prec = max(50, len(d.as_tuple().digits) + decimals + 4)
        quantum = Decimal(1).scaleb(-decimals)
        rounded = (d * 100).quantize(quantum, rounding=ROUND_HALF_UP)
        if d < 1 and rounded >= 100:
            rounded = Decimal(100) - quantum
        return f"{rounded:.{decimals}f}%"


def fraction(numerator, denominator):
    require(denominator > 0, "zero denominator")
    with localcontext() as context:
        context.prec = 50
        return Decimal(numerator) / Decimal(denominator)


def read_saved_csv(path, audit):
    """Keep stored numeric strings verbatim; open sources only for reading."""
    raw = path.read_bytes()
    audit.append({"source": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    require(bool(rows), f"empty source: {path}")
    return rows


def unique_rows(rows, columns, label):
    result = {}
    for row in rows:
        key = tuple(row[c] for c in columns)
        require(all(key) and key not in result, f"blank/duplicate key in {label}: {key}")
        result[key] = row
    return result


def load_final_metrics(base, audit):
    rows = read_saved_csv(base / "stage10_final/test_evaluation_metrics.csv", audit)
    indexed = unique_rows(rows, ("source_dataset", "metric"), "metrics")
    for dataset, support in (("OVERALL", 4300), (DATASETS[0], 1700), (DATASETS[1], 2600)):
        subset = [r for r in rows if r["source_dataset"] == dataset]
        require(bool(subset) and all(int(r["sample_size"]) == support for r in subset),
                f"{dataset} support")
        for metric in ("accuracy", "macro_precision", "macro_recall", "macro_f1",
                       "weighted_precision", "weighted_recall", "weighted_f1"):
            require((dataset, metric) in indexed, f"missing {dataset}/{metric}")
            require(0 <= Decimal(indexed[dataset, metric]["value"]) <= 1,
                    f"invalid {dataset}/{metric}")
    require(Decimal(indexed["OVERALL", "total_support"]["value"]) == 4300, "total support")
    return indexed


def load_per_class_metrics(base, audit):
    rows = read_saved_csv(base / "stage10_final/test_evaluation_per_class_metrics.csv", audit)
    indexed = unique_rows(rows, ("source_dataset", "class_label"), "per-class metrics")
    require(set(indexed) == set(CLASS_KEYS), "per-class keys")
    for row in rows:
        for metric in ("precision", "recall", "f1"):
            require(0 <= Decimal(row[metric]) <= 1, "invalid per-class metric")
    return [indexed[key] for key in CLASS_KEYS]


def load_confusion_matrices(base, audit):
    rows = read_saved_csv(base / "stage10_final/test_evaluation_confusion_matrix.csv", audit)
    indexed = unique_rows(rows, ("source_dataset", "actual_class", "predicted_class"), "confusions")
    expected = ([[850, 1], [0, 849]],
                [[848, 2, 0, 2], [0, 800, 2, 36], [0, 4, 48, 1], [2, 44, 0, 811]])
    matrices = []
    for dataset, check in zip(DATASETS, expected):
        labels = [c for ds, c in CLASS_KEYS if ds == dataset]
        matrix = [[int(indexed[dataset, a, p]["count"]) for p in labels] for a in labels]
        require(matrix == check, f"unexpected confusion matrix for {dataset}")
        matrices.append(matrix)
    require(len(indexed) == 20, "unexpected confusion cells")
    require([sum(map(sum, m)) for m in matrices] == [1700, 2600], "confusion totals")
    return matrices


def load_evidence_status_counts(base, audit):
    rows = read_saved_csv(base / "stage10_final/test_final_evidence_status_distribution.csv", audit)
    indexed = unique_rows(rows, ("evidence_status",), "status distribution")
    require(set(indexed) == {(s,) for s in STATUSES}, "unknown evidence status")
    overall = [int(indexed[s,]["count"]) for s in STATUSES]
    rows = read_saved_csv(base / "stage08_llm/test_llm_by_dataset.csv", audit)
    indexed = unique_rows(rows, ("source_dataset", "evidence_status"), "dataset status")
    datasets = [[int(indexed[ds, s]["count"]) for s in STATUSES] for ds in DATASETS]
    require(overall == [2216, 561, 875, 648], "overall evidence counts")
    require(datasets == [[887, 357, 4, 452], [1329, 204, 871, 196]], "dataset evidence counts")
    require([sum(v) for v in [overall] + datasets] == [4300, 1700, 2600], "evidence totals")
    require([sum(v[i] for v in datasets) for i in range(4)] == overall, "status aggregation")
    return [overall] + datasets


def load_correctness_by_status(base, audit):
    rows = read_saved_csv(base / "stage10_final/test_evaluation_correctness_by_evidence_status.csv", audit)
    unique_rows(rows, ("source_dataset", "evidence_status", "correct"), "correctness")
    counts = Counter()
    for row in rows:
        require(row["source_dataset"] in DATASETS and row["evidence_status"] in STATUSES
                and row["correct"] in ("0", "1"), "unknown correctness key")
        counts[row["evidence_status"], int(row["correct"])] += int(row["count"])
    pairs = [(counts[s, 1], counts[s, 0]) for s in STATUSES]
    require(pairs == [(2157, 59), (558, 3), (852, 23), (639, 9)], "correctness counts")
    require(tuple(map(sum, zip(*pairs))) == (4206, 94), "correctness totals")
    return pairs


def build_matched_v4_v5_records(root, base, audit):
    v5 = read_saved_csv(base / "stage10_final/test_final_classification_records.csv", audit)
    v4 = read_saved_csv(root / "results/rag/test/stage10_final/final_disjoint_4319_v4/test_final_classification_records.csv", audit)
    new = unique_rows(v5, ("query_id",), "V5 records")
    old = unique_rows(v4, ("query_id",), "V4 records")
    common = set(new) & set(old)
    require(len(new) == len(common) == 4300, "exact matched IDs and V5 unique IDs must equal 4300")
    transition = [[0] * 4 for _ in STATUSES]
    for key in sorted(common):
        for field in ("final_class", "classifier_prediction", "source_dataset"):
            require(new[key][field] == old[key][field], f"changed {field} at {key}")
        require(new[key]["final_class"] == new[key]["classifier_prediction"], "classifier override")
        require(new[key]["evidence_status"] in STATUSES and old[key]["evidence_status"] in STATUSES,
                "unknown historical status")
        transition[STATUSES.index(old[key]["evidence_status"])][STATUSES.index(new[key]["evidence_status"])] += 1
    require(transition == [[0, 0, 0, 0], [0, 0, 0, 0], [2049, 544, 825, 617], [167, 17, 50, 31]],
            "historical transition matrix")
    require([sum(row) for row in transition] == [0, 0, 4035, 265], "matched V4 counts")
    return v5, transition


def load_provenance_summary(base, records, audit):
    rows = read_saved_csv(base / "stage10_final/test_final_provenance_summary.csv", audit)
    require(sum(int(r["count"]) for r in rows) == 4300, "provenance total")
    provenance = sum(int(r["count"]) for r in rows if r["provenance_available"] == "YES")
    explanation = sum(int(r["count"]) for r in rows if r["explanation_available"] == "YES")
    require((provenance, explanation) == (3425, 4300), "provenance/explanation availability")
    require(sum(bool(json.loads(r["provenance"])) for r in records) == provenance,
            "record provenance disagrees with summary")
    require(sum(bool(r["explanation"].strip()) for r in records) == explanation, "record explanations")
    ids = {r["query_id"] for r in records}
    chunks = read_saved_csv(base / "stage07_trusted_contexts/trusted_context_chunks.csv", audit)
    require({r["query_id"] for r in chunks} == ids and len(chunks) == 21500, "context chunk coverage")
    diversity = read_saved_csv(base / "stage07_trusted_contexts/trusted_context_document_diversity.csv", audit)
    documents = {qid: set() for qid in ids}
    for row in chunks:
        documents[row["query_id"]].add(row["document_id"])
    require(Counter(map(len, documents.values())) == Counter({int(r["distinct_document_count"]): int(r["count"]) for r in diversity}),
            "context document diversity")
    ready_rows = read_saved_csv(base / "stage07_trusted_contexts/trusted_context_summary.csv", audit)
    ready = sum(int(r["count"]) for r in ready_rows if r["context_status"] == "READY")
    response_path = base / "stage08_llm/test_llm_responses.jsonl"
    raw = response_path.read_bytes()
    audit.append({"source": str(response_path), "sha256": hashlib.sha256(raw).hexdigest()})
    responses = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
    response_ids = [r["query_id"] for r in responses]
    require(len(response_ids) == len(set(response_ids)) == 4300 and set(response_ids) == ids, "Stage-8 response IDs")
    valid = sum(r["validation_status"] == "PASS" for r in responses)
    require(ready == valid == 4300, "ready contexts / valid Stage-8 responses")
    citations = [c for r in records for c in json.loads(r["cited_chunk_ids"])]
    claims = {c for r in records for c in json.loads(r["claim_ids_json"])}
    require((len(citations), len(set(citations)), len(claims)) == (5772, 51, 51), "selected citations/claims")
    retrieved_rows = read_saved_csv(base / "stage04_retrieval/retrieval_method_summary.csv", audit)
    retrieved = sum(int(r["count"]) for r in retrieved_rows)
    verified_rows = read_saved_csv(base / "plotting/stage05_provenance_status_counts.csv", audit)
    verified = sum(int(r["count"]) for r in verified_rows
                   if r["verification_status"] == "VERIFIED" and r["authorization_status"] == "YES")
    require(retrieved == verified == 129000, "retrieved / verified authorized occurrences")
    # None means a count only, not a query-coverage percentage.
    return [
        ("Retrieved candidate occurrences", retrieved, None, "Candidate occurrences across retrieval methods"),
        ("Verified authorized candidate occurrences", verified, retrieved, "Denominator: retrieved candidate occurrences"),
        ("Ready trusted contexts", ready, len(ids), "Denominator: TEST queries"),
        ("Context chunk occurrences", len(chunks), None, "Chunk occurrences supplied in trusted contexts"),
        ("Valid Stage-8 responses", valid, len(ids), "Saved validator PASS; denominator: TEST queries"),
        ("Final provenance available", provenance, len(ids), "Nonempty final mappings; denominator: TEST queries"),
        ("Explanation available", explanation, len(ids), "Nonempty explanation; denominator: TEST queries"),
        ("Selected citation occurrences", len(citations), None, "Selected citations, separate from context occurrences"),
        ("Distinct selected chunks", len(set(citations)), None, "Unique selected chunk identifiers"),
        ("Distinct selected claims", len(claims), None, "Unique selected claim identifiers")]


def load_final_test_data(root):
    """Complete every source/consistency assertion before creating any output."""
    audit = []
    base = root / TEST_NAMESPACE
    require(format_percent(1.0) == "100.00%", "exact-one formatter")
    require(format_percent("0.9999474047714945") == "99.99%", "near-one formatter")
    require(format_percent("0.9994117647058823") == "99.94%", "percentage rounding")
    metrics = load_final_metrics(base, audit)
    classes = load_per_class_metrics(base, audit)
    matrices = load_confusion_matrices(base, audit)
    require([int(r["support"]) for r in classes] == [sum(row) for m in matrices for row in m], "per-class supports")
    evidence = load_evidence_status_counts(base, audit)
    correctness = load_correctness_by_status(base, audit)
    require([sum(pair) for pair in correctness] == evidence[0], "correctness/status totals")
    records = read_saved_csv(base / "stage10_final/test_final_classification_records.csv", audit)
    final_ids = unique_rows(records, ("query_id",), "final TEST records")
    require(len(final_ids) == 4300, "final TEST unique IDs")
    require(all(r["final_class"] == r["classifier_prediction"] for r in records), "final classifier overrides")
    fusion = read_saved_csv(base / "stage09_fusion/test_fusion_records.csv", audit)
    fusion_ids = unique_rows(fusion, ("query_id",), "final TEST fusion")
    require(set(fusion_ids) == set(final_ids), "fusion/final query IDs")
    require(all(r["decision_source"] == "retain_classifier" and
                r["final_class"] == r["classifier_prediction"] == final_ids[key]["classifier_prediction"]
                for key, r in fusion_ids.items()), "fusion must retain every frozen prediction")
    rejected_path = base / "stage08_llm/test_llm_rejected_responses.jsonl"
    rejected_raw = rejected_path.read_bytes()
    audit.append({"source": str(rejected_path), "sha256": hashlib.sha256(rejected_raw).hexdigest()})
    require(not rejected_raw.decode("utf-8-sig").strip(), "Stage-8 rejected response count must be zero")
    record_counts = Counter((r["source_dataset"], r["final_class"], r["evidence_status"]) for r in records)
    cohort = Counter((r["source_dataset"], r["classifier_prediction"]) for r in records)
    require(set(cohort) == set(CLASS_KEYS), "unexpected cohort classes")
    require([cohort[k] for k in CLASS_KEYS] == [850, 850, 850, 850, 50, 850], "frozen prediction cohort")
    matrix_rows = read_saved_csv(base / "stage10_final/test_final_class_evidence_matrix.csv", audit)
    indexed = unique_rows(matrix_rows, ("source_dataset", "final_class", "evidence_status"), "class evidence")
    require(Counter({key: int(r["count"]) for key, r in indexed.items()}) == record_counts, "class evidence records")
    class_evidence = [[int(indexed.get((ds, c, s), {"count": "0"})["count"]) for s in STATUSES] for ds, c in CLASS_KEYS]
    require([sum(row) for row in class_evidence] == [cohort[k] for k in CLASS_KEYS], "class evidence totals")
    for ds, expected in zip(DATASETS, evidence[1:]):
        require([sum(n for (dataset, _, status), n in record_counts.items() if dataset == ds and status == s)
                 for s in STATUSES] == expected, "record/dataset evidence totals")
    provenance = load_provenance_summary(base, records, audit)
    return dict(metrics=metrics, classes=classes, matrices=matrices, evidence=evidence,
                correctness=correctness, provenance=provenance, records=records,
                cohort=cohort, class_evidence=class_evidence,
                audit=audit)


def export_figure(fig, base_path):
    from matplotlib import pyplot as plt
    prepare_visible_text(fig)
    _finalize_figure_layout(fig)
    try:
        for extension in ("pdf", "svg", "png"):
            fig.savefig(base_path.with_suffix("." + extension),
                        bbox_inches=fig.bbox_inches if getattr(fig, "_publication_exact_size", False) else "tight",
                        facecolor="white", **({"dpi": 600} if extension == "png" else {}))
    finally:
        plt.close(fig)


def percentage_axis(ax, axis="x"):
    from matplotlib.ticker import FuncFormatter
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_formatter(FuncFormatter(lambda x, pos: format_percent(Decimal(str(x)) / 100, 0)))


def light_heatmap(ax, counts, row_labels, column_labels, percentages=True):
    """Borderless imshow; annotations and axes remain vector text."""
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    array = np.asarray(counts, dtype=int)
    totals = array.sum(axis=1)
    normalized = np.divide(array, totals[:, None], out=np.zeros(array.shape, dtype=float), where=totals[:, None] != 0)
    cmap = LinearSegmentedColormap.from_list("publication_light_blues", plt.cm.Blues(np.linspace(.05, .60, 256)))
    mesh = ax.imshow(normalized, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(column_labels)), column_labels)
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.tick_params(length=0)
    for i, row in enumerate(array):
        for j, count in enumerate(row):
            label = str(count)
            if percentages and totals[i]:
                label += "\n" + format_percent(fraction(int(count), int(totals[i])))
            ax.text(j, i, label, ha="center", va="center", color="black", fontsize=7)
    return mesh


def stacked_status_bars(ax, groups, labels):
    for x, counts in enumerate(groups):
        bottom = 0.0
        total = sum(counts)
        for i, count in enumerate(counts):
            proportion = fraction(count, total)
            height = float(proportion * 100)
            ax.bar(x, height, bottom=bottom, width=.57, color=STATUS_COLORS[i],
                   edgecolor="none",
                   label=STATUSES[i] if x == 0 else None)
            if height >= 5:
                ax.text(x, bottom + height / 2, format_percent(proportion), ha="center", va="center", fontsize=7)
            bottom += height
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of queries (%)")
    percentage_axis(ax, "y")
    ax.spines[["top", "right"]].set_visible(False)


def status_legend(fig, ax):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=7,
               bbox_to_anchor=(.5, .005))


def plot_test_confusions(data):
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.1), layout="constrained")
    for i, (ax, ds, matrix) in enumerate(zip(axes, DATASETS, data["matrices"])):
        labels = [c for dataset, c in CLASS_KEYS if dataset == ds]
        mesh = light_heatmap(ax, matrix, labels, labels, percentages=False)
        ax.set_title(f"({chr(97+i)}) {ds}", loc="left", weight="bold")
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Actual class")
    colorbar = fig.colorbar(mesh, ax=axes, fraction=.035, pad=.03)
    colorbar.set_label("Row-normalized proportion")
    return fig


def plot_per_class_performance(data):
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(7.16, 4.0), layout="constrained")
    for j, (metric, color, marker) in enumerate(zip(("precision", "recall", "f1"),
                                                   ("#0072B2", "#D55E00", "#009E73"), ("o", "s", "D"))):
        for i, row in enumerate(data["classes"]):
            x = float(row[metric]) * 100
            y = i + (j - 1) * .25
            ax.plot(x, y, marker=marker, color=color, markeredgecolor="black", markeredgewidth=.5,
                    markersize=3.5, linestyle="none", label=metric.capitalize() if i == 0 else None)
            # Separate vertical lanes and inward labels avoid near-perfect overlap.
            ax.annotate(format_percent(row[metric]), (x, y), xytext=(-5 if x > 98 else 5, 0),
                        textcoords="offset points", ha="right" if x > 98 else "left", va="center", fontsize=6.5)
    ax.set_yticks(range(6), [f"{'USTC' if r['source_dataset'] == DATASETS[0] else 'CIC'} / {r['class_label']} (n={r['support']})" for r in data["classes"]])
    ax.set_ylim(5.6, -.6)
    ax.set_xlim(90, 100.2)
    ax.set_xticks([90, 92, 94, 96, 98, 100])
    percentage_axis(ax)
    ax.set_xlabel("Performance (%) — axis begins at 90%")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    place_external_legend(ax, ncol=3)
    return fig


def plot_evidence_and_correctness(data):
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 4.2), gridspec_kw={"width_ratios": [1, 1.1]})
    fig.subplots_adjust(left=.075, right=.97, bottom=.28, top=.80, wspace=.70)
    safe_suptitle(fig, "Evidence Status and Correctness")
    labels = [f"{name}\n(n={sum(counts):,})" for name, counts in zip(("Overall", "USTC", "CIC"), data["evidence"])]
    stacked_status_bars(axes[0], data["evidence"], labels)
    axes[0].set_title("(a) Evidence status", loc="left", weight="bold")
    small = data["evidence"][1][2]
    axes[0].text(.5, -.38, f"USTC INSUFFICIENT: {small} / {sum(data['evidence'][1]):,} ({format_percent(fraction(small, sum(data['evidence'][1])))})",
                 transform=axes[0].transAxes, ha="center", fontsize=7)
    ax = axes[1]
    for i, (correct, errors) in enumerate(data["correctness"]):
        value = fraction(correct, correct + errors)
        ax.scatter(float(value * 100), i, s=35, color=STATUS_COLORS[i], edgecolor="black", linewidth=.6)
        ax.annotate(f"{format_percent(value)}  ({errors} errors)", (float(value * 100), i),
                    xytext=(0, 10), textcoords="offset points", ha="center", fontsize=7)
    ax.set_yticks(range(4), STATUSES)
    ax.set_ylim(3.6, -.7)
    ax.set_xlim(96.5, 100.1)
    ax.set_xticks([97, 98, 99, 100])
    percentage_axis(ax)
    ax.set_title("(b) Accuracy within status", loc="left", weight="bold")
    ax.set_xlabel("Accuracy (%) — axis begins at 96.5%")
    ax.text(.5, -.38, "Evidence status ≠ classifier correctness", transform=ax.transAxes, ha="center", fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    status_legend(fig, axes[0])
    return fig


def plot_v4_v5_comparison(data):
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.5), gridspec_kw={"width_ratios": [.8, 1.3]})
    fig.subplots_adjust(left=.075, right=.985, top=.86, bottom=.26, wspace=.65)
    stacked_status_bars(axes[0], [[sum(r) for r in data["transition"]], data["evidence"][0]], ["V4", "V5"])
    axes[0].set_title("(a) Evidence-status composition", loc="left", weight="bold")
    # Full status names on rows; abbreviated columns are defined by the legend.
    light_heatmap(axes[1], data["transition"], STATUSES, ["SUP", "CONTR", "INSUFF", "CONFL"])
    axes[1].set_title("(b) Exact status transitions", loc="left", weight="bold")
    axes[1].set_xlabel("V5 evidence status")
    axes[1].set_ylabel("V4 evidence status")
    fig.text(.5, .08, "Same 4,300 query IDs; final-class changes = 0", ha="center", fontsize=8)
    status_legend(fig, axes[0])
    return fig


def plot_provenance_coverage(data):
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(7.16, 2.4))
    fig.subplots_adjust(left=.25, right=.83, bottom=.36, top=.98)
    rows = [data["provenance"][i] for i in (2, 4, 6, 5)]
    for i, (label, count, denominator, _) in enumerate(rows):
        value = fraction(count, denominator)
        ax.barh(i, float(value * 100), height=.6, color="#8FB5CE" if i < 3 else "#B0B0B0", edgecolor="none")
        ax.text(2, i, format_percent(value), va="center", fontsize=8)
        ax.annotate(f"{count:,} / {denominator:,}", (float(value * 100), i), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=8)
    ax.set_yticks(range(4), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 101)
    percentage_axis(ax)
    ax.set_xlabel("Coverage of TEST queries (%)")
    ax.spines[["top", "right"]].set_visible(False)
    p = data["provenance"]
    fig.text(.25, .04, f"Context chunk occurrences: {p[3][1]:,}     Selected citation occurrences: {p[7][1]:,}\n"
             f"Distinct selected chunks: {p[8][1]:,}     Distinct selected claims: {p[9][1]:,}", fontsize=8)
    return fig


def plot_supplementary_results(data):
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    fig.subplots_adjust(bottom=.24, top=.87, left=.18, right=.97)
    stacked_status_bars(ax, [data["validation"], data["evidence"][0]],
                        [f"Validation\n(n={sum(data['validation']):,})", f"TEST\n(n={sum(data['evidence'][0]):,})"])
    ax.set_title("Different cohorts — descriptive comparison", fontsize=9)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=7)
    matrix_fig, matrix_ax = plt.subplots(figsize=(7.16, 3.2), layout="constrained")
    mesh = light_heatmap(matrix_ax, data["class_evidence"],
                        [f"{'USTC' if ds == DATASETS[0] else 'CIC'} {c}" for ds, c in CLASS_KEYS], STATUSES)
    matrix_ax.set_xlabel("Evidence status")
    matrix_ax.set_ylabel("Final predicted class")
    matrix_fig.colorbar(mesh, ax=matrix_ax, label="Row-normalized proportion")
    return fig, matrix_fig


def latex_escape(value):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
                    "^": r"\textasciicircum{}"}
    return "".join(replacements.get(c, c) for c in str(value))


def write_publication_table(base, headers, rows, percent_columns=(), note="", latex_rows=None):
    """CSV stores original strings/unrounded ratios; only LaTeX is formatted."""
    with base.with_suffix(".csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)
    display = latex_rows if latex_rows is not None else [
        [format_percent(value) if i in percent_columns else str(value) for i, value in enumerate(row)]
        for row in rows]
    # Wrapping text columns keeps provenance descriptions suitable for a page.
    layout = r"p{0.27\linewidth}rp{0.20\linewidth}p{0.31\linewidth}" if base.name == TABLE_NAMES[3] else "l" + "r" * (len(headers) - 1)
    lines = [r"% Requires \usepackage{booktabs}", r"\begingroup", r"\small",
             r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{" + layout + "}", r"\toprule",
             " & ".join(map(latex_escape, headers)) + r" \\", r"\midrule"]
    lines += [" & ".join(map(latex_escape, row)) + r" \\" for row in display]
    lines += [r"\bottomrule", r"\end{tabular}"]
    if note:
        lines += [r"\par\smallskip", r"\begin{minipage}{\linewidth}\footnotesize " + latex_escape(note) + r"\end{minipage}"]
    lines += [r"\endgroup", ""]
    base.with_suffix(".tex").write_text("\n".join(lines), encoding="utf-8")


def write_performance_table(data, output):
    names = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_precision", "weighted_recall", "weighted_f1")
    labels = ("Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted Precision", "Weighted Recall", "Weighted F1")
    rows = [[label] + [data["metrics"][ds, metric]["value"] for ds in ("OVERALL",) + DATASETS] for label, metric in zip(labels, names)]
    write_publication_table(output / TABLE_NAMES[0], ["Metric", "Overall", *DATASETS], rows, (1, 2, 3),
                            "Fixed historical prediction-stratified TEST cohort; unweighted, not population estimates. Overall macro metrics use six dataset-qualified classes. CSV metrics are fractions with original stored precision.")


def write_per_class_table(data, output):
    rows = [[r[k] for k in ("source_dataset", "class_label", "precision", "recall", "f1", "support")] for r in data["classes"]]
    write_publication_table(output / TABLE_NAMES[1], ["Dataset", "Class", "Precision", "Recall", "F1", "Support"], rows, (2, 3, 4), "Support is from saved evaluation artifacts. CSV metrics are fractions.")


def write_evidence_correctness_table(data, output):
    pairs = data["correctness"]
    rows = []
    for status, (correct, errors) in list(zip(STATUSES, pairs)) + [("Overall", tuple(map(sum, zip(*pairs))))]:
        count = correct + errors
        rows.append([status, count, fraction(count, sum(data["evidence"][0])), correct, errors, fraction(correct, count)])
    write_publication_table(output / TABLE_NAMES[2], ["Evidence status", "Count", "Share", "Correct", "Incorrect", "Accuracy within status"], rows, (2, 5), "Evidence compatibility does not denote classifier correctness. CSV share and accuracy are fractions derived from saved counts.")


def write_provenance_table(data, output):
    rows, display = [], []
    for name, count, denominator, interpretation in data["provenance"]:
        coverage = "" if denominator is None else f"{count}/{denominator}"
        rows.append([name, count, coverage, interpretation])
        display.append([name, str(count), "Count only" if denominator is None else f"{format_percent(fraction(count, denominator))} ({coverage})", interpretation])
    write_publication_table(output / TABLE_NAMES[3], ["Quantity", "Count", "Coverage / denominator", "Interpretation"], rows,
                            note="Context chunk occurrences and selected citation occurrences have distinct meanings; counts without a denominator are not coverage percentages.", latex_rows=display)


def write_v4_v5_table(data, output):
    old = [sum(row) for row in data["transition"]]
    new = data["evidence"][0]
    rows = [[s, a, fraction(a, sum(old)), b, fraction(b, sum(new))] for s, a, b in zip(STATUSES, old, new)]
    write_publication_table(output / TABLE_NAMES[4], ["Status", "V4 count", "V4 share", "V5 count", "V5 share"], rows, (2, 4),
                            "Exact matched historical query IDs; final-class differences = 0. Frozen-prediction differences = 0. CSV shares are fractions.")


def write_supplementary_tables(data, output):
    rows = [[ds, c, data["cohort"][ds, c], fraction(data["cohort"][ds, c], sum(data["cohort"].values()))] for ds, c in CLASS_KEYS]
    write_publication_table(output / TABLE_NAMES[5], ["Dataset", "Frozen prediction", "Selected", "Share of 4300"], rows, (3,), "Prediction-stratified historical TEST cohort; CSV shares are fractions.")
    old, new = data["validation"], data["evidence"][0]
    rows = [[s, a, fraction(a, sum(old)), b, fraction(b, sum(new))] for s, a, b in zip(STATUSES, old, new)]
    write_publication_table(output / TABLE_NAMES[6], ["Status", "Validation count", "Validation share", "TEST count", "TEST share"], rows, (2, 4), "Different cohorts; not paired observations. CSV shares are fractions.")


def prepare_final_test_results():
    root = Path(__file__).resolve().parent
    data = load_final_test_data(root)  # All numeric checks precede ALL output writes.
    output = root / "final_paper_figures/final_test_results"
    figures, tables = output / "figures", output / "tables"
    # Check every intended destination before writing, including linked parents.
    directories = [root / "final_paper_figures", output, figures, tables]
    destinations = [figures / (FIGURE_NAMES[i] + ext) for i in (0, 1, 2, 4) for ext in (".pdf", ".png", ".svg")]
    destinations += [tables / (name + ext) for name in TABLE_NAMES[:4] for ext in (".csv", ".tex")]
    destinations.append(output / "Preparation_Report.json")
    for path in directories + destinations:
        require(not path.is_symlink() and path.resolve().is_relative_to(root), f"unsafe output: {path}")
        require(not path.resolve().is_relative_to((root / "results").resolve()), f"output points to results: {path}")
    for directory in directories:
        directory.mkdir(exist_ok=True)
    style = {"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
             "axes.labelsize": 8.5, "axes.titlesize": 9, "xtick.labelsize": 7.5,
             "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "text.color": "black",
             "axes.labelcolor": "black", "axes.titlecolor": "black", "xtick.color": "black",
             "ytick.color": "black", "figure.facecolor": "white", "axes.facecolor": "white",
             "axes.grid": False, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}
    with rc_context(style):
        for index, plot in ((0, plot_test_confusions), (1, plot_per_class_performance),
                            (2, plot_evidence_and_correctness), (4, plot_provenance_coverage)):
            export_figure(plot(data), figures / FIGURE_NAMES[index])
    for writer in (write_performance_table, write_per_class_table, write_evidence_correctness_table,
                   write_provenance_table):
        writer(data, tables)
    # Reuse the existing Preparation_Report.json convention for this isolated mode.
    recommendations = ("MAIN — REQUIRED", "MAIN — IF SPACE", "MAIN — REQUIRED", "MAIN — REQUIRED",
                       "SUPPLEMENT / MAIN TABLE PREFERRED", "SUPPLEMENT", "SUPPLEMENT")
    reasons = ("Final predictive performance", "Overlaps the per-class table",
               "Evidence compatibility is distinct from correctness", "Matched evidence changes without prediction changes",
               "Often clearer as a compact table", "Different cohorts", "Supplementary class composition")
    table_recommendations = ("MAIN", "MAIN if R2 omitted; otherwise SUPPLEMENT", "MAIN",
                            "Compact MAIN if space; otherwise SUPPLEMENT", "MAIN if R4 omitted; otherwise SUPPLEMENT",
                            "SUPPLEMENT / METHODOLOGY", "SUPPLEMENT")
    report = {"mode": "final-test-results", "sources": data["audit"],
              "summary": [{"name": name, "recommendation": rec, "reason": reason, "caption": caption,
                           "files": [f"figures/{name}.{ext}" for ext in ("pdf", "png", "svg")]}
                          for i, (name, rec, reason, caption) in enumerate(zip(FIGURE_NAMES, recommendations, reasons, CAPTIONS)) if i in (0, 1, 2, 4)],
              "tables": [{"name": name, "recommendation": rec, "files": [f"tables/{name}.{ext}" for ext in ("csv", "tex")]}
                         for name, rec in zip(TABLE_NAMES[:4], table_recommendations[:4])],
              "result_scope": "FINAL_TEST",
              "V4/V5 comparison figures excluded": "YES",
              "validation-vs-TEST result figures excluded": "YES",
              "validation": {"support": sum(data["evidence"][0]), "classifier_overrides": 0},
              "notes": ["Fixed historical TEST comparison; not a pristine unseen cohort or population-weighted estimate.",
                        "All source files opened read-only; no inference or evaluation pipeline invoked.",
                        "Stored metrics preserved verbatim in CSV; derived ratios use 50-digit Decimal arithmetic.",
                        "Percentage display uses shared anti-100% rounding; exact one remains 100.00%.",
                        "Empirical TRAIN associations are not causal; evidence states use schema/validator constraints."]}
    (output / "Preparation_Report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Final TEST publication artifacts written to: {output}")
    return 0


# Complete-library adapters deliberately live in this file. They consume saved
# summaries only and never import the research implementation or deserialize models.
USTC_DATA = "results/datasets/USTC-TFC2016/20260919T121151Z/"
USTC_PRE = "results/datasets/USTC-TFC2016/preprocessing/20260919T123108Z/"
SPLITS = ("train", "validation", "test")
SPLIT_COLORS = ("#0072B2", "#E69F00", "#009E73")
MODEL_PALETTE = ("#0072B2", "#8064A2", "#78909C", "#B07D42", "#CC79A7")
SOURCE_PALETTE = ("#4477AA", "#66CCEE", "#AA3377", "#BBBBBB", "#332288", "#88CCEE")
PUBLICATION_STYLE = {
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
    "axes.labelsize": 8.5, "axes.titlesize": 10, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "text.color": "black",
    "axes.labelcolor": "black", "axes.titlecolor": "black", "xtick.color": "black",
    "ytick.color": "black", "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": False, "patch.edgecolor": "none", "pdf.fonttype": 42,
    "ps.fonttype": 42, "svg.fonttype": "none"}


class SourceNotFound(ValueError):
    """A requested scientific quantity has no authoritative saved source."""


class PublicationSources:
    """Read-only CSV cache with per-candidate lineage, including cached reads."""
    def __init__(self, root):
        self.root = root
        self.cache = {}
        self.used = set()
        self.hashes = {}
        self.final_data = None

    def touch(self, relative):
        relative = str(relative).replace("\\", "/")
        self.used.add(relative)
        path = self.root / relative
        if not path.is_file():
            raise SourceNotFound(f"Required saved artifact: {relative}")
        require(path.resolve().is_relative_to(self.root), "source outside project")
        if relative not in self.hashes:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            self.hashes[relative] = digest.hexdigest()
        return path

    def rows(self, relative):
        path = self.touch(relative)
        key = str(relative)
        if key not in self.cache:
            csv.field_size_limit(100_000_000)
            with path.open(encoding="utf-8-sig", newline="") as stream:
                self.cache[key] = list(csv.DictReader(stream))
            require(bool(self.cache[key]), f"empty CSV: {relative}")
        return self.cache[key]

    def stage(self, relative):
        return self.rows(TEST_NAMESPACE / relative)

    def final(self):
        if self.final_data is None:
            self.final_data = load_final_test_data(self.root)
        for item in self.final_data["audit"]:
            relative = Path(item["source"]).relative_to(self.root).as_posix()
            self.used.add(relative)
            self.hashes[relative] = item["sha256"]
        return self.final_data


def checked_counts(rows, keys, value="count", total=None):
    counts = Counter()
    for row in rows:
        number = int(row[value]) if value else 1
        require(number >= 0, f"negative count for {keys}")
        counts[tuple(row[k] for k in keys)] += number
    require(bool(counts), f"missing categories: {keys}")
    if total is not None:
        require(sum(counts.values()) == total, f"{keys} total: {sum(counts.values())} != {total}")
    return counts


def finite_number(value, unit_interval=False):
    number = Decimal(str(value))
    require(number.is_finite(), "non-finite saved number")
    if unit_interval:
        require(0 <= number <= 1, "metric outside [0,1]")
    return float(number)


def load_phase1_dataset_stats(s):
    closures = [s.rows(f"metadata/traffic/{prefix}_experiment_closure.csv")[0] for prefix in ("ustc", "cic")]
    for ds, row in zip(DATASETS, closures):
        require(row["dataset"] == ds and row["stage_status"] == "CLOSED_FROZEN"
                and row["integrity_audit_status"] == row["finalization_status"] == "PASS", "Phase-I closure")
    summary = s.rows(USTC_DATA + "dataset_summary.csv")[0]
    ustc = checked_counts(s.rows(USTC_DATA + "class_distribution.csv"), ("class",), "flow_count", int(summary["total_flows"]))
    contract = s.rows("metadata/traffic/cic_phase1_contract.csv")[0]
    cic = {(label,): sum(int(contract[f"{split}_{field}_rows"]) for split in SPLITS)
           for label, field in zip(("Non-Tor", "NonVPN", "Tor", "VPN"), ("non_tor", "nonvpn", "tor", "vpn"))}
    counts = [ustc, cic]
    require(set(ustc) == {("benign",), ("malware",)}, "USTC project targets")
    for i, row in enumerate(closures):
        require(sum(counts[i].values()) == sum(int(row[f"{split}_rows"]) for split in SPLITS), "dataset / closure totals")
    require(sum(cic.values()) == int(contract["raw_rows"]) and len(cic) == int(contract["class_count"]), "CIC population")
    return closures, counts, summary, contract


def load_split_stats(s):
    closures, counts, _, contract = load_phase1_dataset_stats(s)
    rows = s.rows(USTC_PRE + "split_distribution.csv")
    ustc = checked_counts(rows, ("split",), "flow_count", sum(counts[0].values()))
    # The CIC manifest supplies saved aggregate group counts; no raw labels read.
    cic_rows = s.rows("metadata/traffic/cic_split_manifest.csv")
    cic = checked_counts(cic_rows, ("split",), "group_row_count", sum(counts[1].values()))
    values = [[ustc[split,] for split in SPLITS], [cic[split.upper(),] for split in SPLITS]]
    for i in range(2):
        require(values[i] == [int(closures[i][f"{split}_rows"]) for split in SPLITS], "split manifest / closure mismatch")
    require(values[1] == [int(contract[f"{split}_rows"]) for split in SPLITS], "CIC contract / split mismatch")
    return values


def load_model_comparison(s):
    s.touch("RESEARCH_PROTOCOL.md")  # Records the validation-only selection policy.
    closures, _, _, _ = load_phase1_dataset_stats(s)
    all_rows = []
    for ds, closure in zip(DATASETS, closures):
        rows = s.rows(closure["validation_results_path"])
        require(len(rows) == len({r["model"] for r in rows}) == 5, "five distinct validation models required")
        keys = ("accuracy", "precision", "recall", "f1") if ds == DATASETS[0] else (
            "accuracy", "macro_precision", "macro_recall", "macro_f1")
        for row in rows:
            require(row["dataset"] == ds and row["authoritative_run"] == closure["authoritative_run"], "validation run identity")
            require(int(row.get("total_rows", row.get("row_count", 0))) == int(closure["validation_rows"]), "validation support")
            for key in keys:
                finite_number(row[key], True)
        all_rows.append(rows)
    require({r["model"] for r in all_rows[0]} == {r["model"] for r in all_rows[1]}, "model sets differ")
    return all_rows


def load_roc_curves(s, dataset_index, selected_only=False):
    """Post-hoc ROC from completed TEST evaluation; tied scores move together.

    No model, tuning, threshold selection, raw dataset labels, or validation ROC.
    AUC uses integer pair counts, retaining exact-one versus sub-one distinction.
    """
    try:
        rows = s.stage("stage10_final/test_evaluation_records.csv")
    except SourceNotFound as exc:
        raise SourceNotFound("TEST_ROC_SOURCE_NOT_FOUND: completed TEST evaluation records required") from exc
    required = {"query_id", "source_dataset", "true_class_label", "classifier_probabilities", "final_class"}
    if not rows or not required.issubset(rows[0]):
        raise SourceNotFound("TEST_ROC_SOURCE_NOT_FOUND: evaluation labels, IDs and frozen probabilities required")
    final = unique_rows(s.final()["records"], ("query_id",), "frozen TEST ROC records")
    evaluated = unique_rows(rows, ("query_id",), "TEST ROC evaluation records")
    require(set(evaluated) == set(final) and len(evaluated) == 4300, "TEST ROC exact ID match")
    probabilities = {}
    for key, row in evaluated.items():
        require(row["source_dataset"] == final[key]["source_dataset"] and row["final_class"] == final[key]["final_class"], "TEST ROC identity")
        expected_labels = {label for ds, label in CLASS_KEYS if ds == row["source_dataset"]}
        if not row["classifier_probabilities"] or not row["true_class_label"]:
            raise SourceNotFound("TEST_ROC_SOURCE_NOT_FOUND: missing saved TEST labels or probabilities")
        vector = json.loads(row["classifier_probabilities"], parse_float=Decimal)
        frozen = json.loads(final[key]["classifier_probabilities"], parse_float=Decimal)
        if not expected_labels.issubset(vector):
            raise SourceNotFound("TEST_ROC_SOURCE_NOT_FOUND: incomplete saved per-class TEST probabilities")
        require(vector == frozen and set(vector) == expected_labels, "TEST ROC frozen probability match")
        require(row["true_class_label"] in expected_labels, "TEST ROC evaluation class")
        require(all(Decimal(str(v)).is_finite() and 0 <= Decimal(str(v)) <= 1 for v in vector.values()), "TEST ROC score range")
        require(abs(sum(Decimal(str(v)) for v in vector.values()) - 1) < Decimal("1e-10"), "TEST ROC probability sum")
        probabilities[key] = vector
    subset = [(key, row) for key, row in evaluated.items() if row["source_dataset"] == DATASETS[dataset_index]]
    require(len(subset) == (1700 if dataset_index == 0 else 2600), "TEST ROC dataset support")
    labels = ("malware",) if dataset_index == 0 else ("Non-Tor", "NonVPN", "Tor", "VPN")
    curves = {}
    for label in labels:
        groups = {}
        for key, row in subset:
            score = Decimal(str(probabilities[key][label]))
            group = groups.setdefault(score, [0, 0])
            group[0 if row["true_class_label"] == label else 1] += 1
        positive = sum(pair[0] for pair in groups.values())
        negative = sum(pair[1] for pair in groups.values())
        require(positive > 0 and negative > 0, "ROC requires both classes")
        tp = fp = twice_area = 0
        points = [(0., 0.)]
        for score in sorted(groups, reverse=True):
            pos, neg = groups[score]
            twice_area += neg * (2 * tp + pos)
            tp += pos
            fp += neg
            points.append((float(fraction(fp, negative)), float(fraction(tp, positive))))
        require(points[-1] == (1., 1.), "TEST ROC endpoints")
        curves[label] = (points, fraction(twice_area, 2 * positive * negative))
    return curves


def load_stage1_query_stats(s):
    rows = s.stage("stage01_structured_queries/test_query_records.csv")
    require(len(unique_rows(rows, ("query_id",), "query records")) == 4300, "Stage-1 unique queries")
    counts = checked_counts(rows, ("source_dataset", "classifier_prediction"), None, 4300)
    summary = s.stage("stage01_structured_queries/test_query_class_distribution.csv")
    require(counts == checked_counts(summary, ("source_dataset", "classifier_prediction"), total=4300), "query composition")
    for row in rows:
        finite_number(row["classifier_confidence"], True)
        require(row["source_split"] == "test", "query split")
    return {"records": rows, "counts": counts}


def load_stage2_corpus_stats(s):
    rows = s.stage("stage02_corpus/corpus_document_summary.csv")
    ref = s.stage("stage02_corpus/corpus_reference_summary.csv")[0]
    require(len(unique_rows(rows, ("document_id",), "corpus")) == int(ref["document_count"]), "corpus document total")
    return rows


def load_stage3_index_stats(s):
    rows = s.stage("stage03_index/index_chunk_summary.csv")
    ref = s.stage("stage03_index/index_reference_summary.csv")[0]
    require(len(unique_rows(rows, ("chunk_id",), "index chunks")) == int(ref["chunk_count"]), "index chunk total")
    require(len({r["document_id"] for r in rows}) == int(ref["document_count"]), "index document total")
    require(all(0 < int(r["chunk_token_count"]) <= int(ref["token_limit"]) for r in rows), "index token limit")
    saved = s.stage("stage03_index/index_document_chunk_counts.csv")
    require(checked_counts(rows, ("document_id", "source_type"), None) == checked_counts(saved, ("document_id", "source_type")), "chunks per document")
    return rows, ref


def load_stage4_retrieval_stats(s):
    rows = s.stage("stage04_retrieval/retrieval_method_summary.csv")
    counts = checked_counts(rows, ("retrieval_method",), total=129000)
    require(len(counts) == 3 and set(counts.values()) == {43000}, "retrieval method totals")
    sources = s.stage("stage04_retrieval/retrieval_source_distribution.csv")
    require(checked_counts(sources, ("retrieval_method",)) == counts, "retrieval source totals")
    return counts, sources


def load_stage5_provenance_stats(s):
    rows = s.stage("stage05_provenance/provenance_status_summary.csv")
    counts = checked_counts(rows, ("verification_status", "authorization_status"), total=129000)
    require(counts == {("VERIFIED", "YES"): 129000}, "verification categories")
    return rows


def load_stage6_reranking_stats(s):
    rows = s.stage("stage06_reranking/reranked_candidates.csv")
    require(len(rows) == 43000, "reranked occurrence total")
    require(len(unique_rows(rows, ("query_id", "final_rank"), "reranked slots")) == 43000, "reranking slots")
    per_query = checked_counts(rows, ("query_id",), None)
    require(len(per_query) == 4300 and set(per_query.values()) == {10}, "reranking top-k")
    for row in rows:
        finite_number(row["reranking_score"])
    return rows


def load_stage7_context_stats(s):
    rows = s.stage("stage07_trusted_contexts/trusted_context_chunks.csv")
    counts = checked_counts(rows, ("query_id",), None, 21500)
    require(len(counts) == 4300 and set(counts.values()) == {5}, "context chunk coverage")
    return rows


def load_stage8_llm_stats(s):
    rows = s.stage("stage08_llm/test_llm_execution_summary.csv")
    counts = checked_counts(rows, ("response_valid",), total=4300)
    require(counts == {("True",): 4300}, "saved accepted responses")
    rejected = s.touch(TEST_NAMESPACE / "stage08_llm/test_llm_rejected_responses.jsonl")
    with rejected.open(encoding="utf-8-sig") as stream:
        rejected_count = sum(bool(line.strip()) for line in stream)
    require(rejected_count == 0, "saved rejected responses")
    return [("Accepted", sum(counts.values())), ("Rejected", rejected_count)]


def load_stage9_fusion_stats(s):
    rows = s.stage("stage09_fusion/test_fusion_summary.csv")
    counts = checked_counts(rows, ("decision_source",), total=4300)
    require(counts == {("retain_classifier",): 4300}, "fusion actions")
    return rows


def new_axes(title="", size=(7.16, 3.2)):
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=size, layout="constrained")
    if title:
        import textwrap
        ax.set_title(textwrap.fill(title, 66), loc="left", fontsize=9, pad=10)
    clean_axes(ax)
    return fig, ax


def clean_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
    ax.tick_params(colors="black")


def safe_suptitle(fig, title):
    """Constrained-layout callers reserve the title automatically."""
    return fig.suptitle(title, fontsize=10, fontweight="bold")


FIGURE_TITLES = {
    "P01": "Dataset Overview", "P02": "Dataset Class Composition",
    "P03": "USTC Family Distribution", "P04": "Dataset Split Counts",
    "P05": "Dataset Split Composition", "P05b": "USTC Family-Disjoint Split",
    "P06": "Flow Extraction and Preparation", "P07": "Preprocessing Retention",
    "P08": "Traffic Representation", "P09": "Five-Model Comparison",
    "P09a": "USTC Five-Model Comparison", "P09b": "CIC Five-Model Comparison",
    "P11": "Model Metric Matrix", "P12": "TEST ROC Curves",
    "P13": "USTC-TFC2016 TEST ROC", "P14": "CIC-Darknet2020 TEST ROC",
    "P19": "Confidence by Class", "P21": "Integrity Audit",
    "Q02": "Query Confidence Distribution", "Q03": "Query Probability Summary",
    "Q05": "Evidence Corpus", "Q06": "Corpus Source Distribution",
    "Q07": "Corpus Claim Counts", "Q09": "Chunks per Document",
    "Q10": "Index Token Distribution", "Q12": "Retrieval Method Distribution",
    "Q13": "Retrieval Source Distribution", "Q15": "Provenance Status",
    "Q16": "Verified Source Types", "Q17": "Evidence Composition After Reranking",
    "Q18": "Reranking Score Distribution", "Q19": "Retrieval and Reranking",
    "Q20": "Trusted Context Coverage", "Q21": "Trusted Context Source Distribution",
    "Q22": "Document Diversity", "Q25": "Evidence Status by Class",
    "Q28": "Evidence Status and Correctness", "Q29": "Decision Fusion",
    "Q30": "Evidence Status under Decision Fusion", "Q32": "Final Classification",
    "Q33": "Classification Performance", "Q34": "Per-Class Performance",
    "Q35": "Final Class Distribution", "Q37": "Provenance Coverage",
    "X01": "Query Record Progression", "X02": "Evidence Processing Progression",
    "M01": "Complete Methodology Pipeline",
}


def display_text(value):
    """Display-only vocabulary; saved fields, values and captions are untouched."""
    import re
    text = str(value)
    for raw, friendly in SOURCE_DISPLAY_NAMES.items():
        text = text.replace(raw, friendly)
    for phrase in ("fixed historical TEST cohort", "prediction-stratified cohort",
                   "unchanged classifier predictions", "same frozen classifier",
                   "fixed historical TEST", "historical TEST", "historical comparison"):
        text = re.sub(re.escape(phrase), "", text, flags=re.I)
    text = re.sub(r"frozen[_\s-]*|cohorts?[_\s-]*|historical[_\s-]*|\bV[45]\b|validation-vs-TEST", "", text, flags=re.I)
    return "\n".join(re.sub(r"[ \t]{2,}", " ", line).strip(" —;") for line in text.split("\n"))


def prepare_visible_text(fig, candidate=None):
    """Clean all Text artists, including legends, table cells and dynamic labels.

    Return displaced wording for external metadata. Never touches source data.
    This runs before savefig, including the separate final-test export mode.
    """
    from matplotlib.text import Text
    import re
    moved = []
    if candidate and candidate.figure_id in FIGURE_TITLES:
        title = FIGURE_TITLES[candidate.figure_id]
        axes = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]
        # Preserve dataset/panel titles, but never repeat a single-panel title.
        if len(axes) == 1:
            for location in ("left", "center", "right"):
                old = axes[0].get_title(loc=location)
                if old:
                    moved.append(old)
                    axes[0].set_title("", loc=location)
        if fig._suptitle is not None and fig._suptitle.get_text() != title:
            moved.append(fig._suptitle.get_text())
        safe_suptitle(fig, title)
        if not fig.get_constrained_layout() and not getattr(fig, "_publication_layout_locked", False):
            fig.subplots_adjust(top=min(fig.subplotpars.top, .86))
    for artist in fig.findobj(Text):
        old = artist.get_text()
        if not old:
            continue
        new = display_text(old)
        if old != new:
            moved.append(old)
            artist.set_text(new)
        require(not re.search(r"frozen|cohort|historical|\bV[45]\b|validation-vs-TEST", new, re.I), "forbidden visible figure wording")
    if candidate and candidate.figure_id == "M01":
        assert_methodology_text(fig)
    return sorted(set(moved))


def place_external_legend(ax, handles=None, labels=None, ncol=1):
    options = dict(loc="upper center", bbox_to_anchor=(.5, -.32), frameon=False,
                   ncol=ncol, fontsize=7, borderaxespad=0, labelspacing=.35,
                   columnspacing=1.1, handlelength=1.5)
    return ax.legend(handles, labels, **options) if handles is not None else ax.legend(**options)


def _place_bottom_legend(legend_ax, source_ax, ncol=1):
    """A separate layout row, never an anchor competing with axis labels."""
    legend_ax.set_axis_off()
    handles, labels = source_ax.get_legend_handles_labels()
    return legend_ax.legend(handles, labels, loc="upper center", frameon=False,
                            ncol=ncol, fontsize=6.5, borderaxespad=0,
                            labelspacing=.35, columnspacing=.9, handlelength=1.6)


def _finalize_figure_layout(fig):
    """Reserve padding without mixing tight_layout and constrained layout.

    Explicit ROC/methodology geometry is locked. Other constrained figures
    include external legends in their layout; manually reserved note lanes stay.
    """
    if getattr(fig, "_publication_layout_locked", False):
        return
    for ax in fig.axes:
        ax.xaxis.labelpad = max(ax.xaxis.labelpad, 8)
        ax.yaxis.labelpad = max(ax.yaxis.labelpad, 8)
        ax.grid(False, which="both")
        for legend in [ax.get_legend()]:
            if legend is not None:
                legend.set_in_layout(True)
    if fig.get_constrained_layout():
        fig.set_constrained_layout_pads(w_pad=.08, h_pad=.12, wspace=.08, hspace=.10)


def annotate_bars(ax, bars, labels, horizontal=False):
    for bar, label in zip(bars, labels):
        position = (bar.get_width(), bar.get_y()+bar.get_height()/2) if horizontal else (
            bar.get_x()+bar.get_width()/2, bar.get_height())
        ax.annotate(label, position, xytext=(4, 0) if horizontal else (0, 4),
                    textcoords="offset points", ha="left" if horizontal else "center",
                    va="center" if horizontal else "bottom", fontsize=7, color="black")


SOURCE_DISPLAY_NAMES = {
    "dataset_documentation": "Dataset documentation", "standards_rfc": "Standards / RFC",
    "official_protocol_documentation": "Official protocol documentation",
    "peer_reviewed_research": "Peer-reviewed research",
    "FROZEN_TRAIN_EMPIRICAL_EVIDENCE": "TRAIN empirical evidence"}


def draw_counts(ax, pairs, xlabel, colors=None, horizontal=True):
    import textwrap
    labels, values = zip(*pairs)
    values = [int(v) for v in values]
    require(all(v >= 0 for v in values), "negative plotted counts")
    colors = colors or "#4477AA"
    if horizontal:
        ax.barh(range(len(values)), values, color=colors, edgecolor="none")
        ax.set_yticks(range(len(values)), [textwrap.fill(str(label), 31) for label in labels])
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_xlim(0, max(max(values), 1) * 1.22)
        for i, value in enumerate(values):
            ax.annotate(f"{value:,}", (value, i), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7)
    else:
        ax.bar(range(len(values)), values, color=colors, edgecolor="none")
        ax.set_xticks(range(len(values)), [textwrap.fill(str(label), 17) for label in labels])
        ax.set_ylabel(xlabel)
        ax.set_ylim(0, max(max(values), 1) * 1.2)
        for i, value in enumerate(values):
            ax.annotate(f"{value:,}", (i, value), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.ticklabel_format(axis="x" if horizontal else "y", style="plain", useOffset=False)


def count_figure(counts, title, unit, colors=None, horizontal=True):
    import textwrap
    pairs = [(" / ".join(key) if isinstance(key, tuple) else str(key), value) for key, value in counts.items()]
    label_lines = sum(max(1, len(textwrap.wrap(label, 31))) for label, _ in pairs)
    height = max(2.8, .20 * label_lines + .14 * len(pairs) + 1.0) if horizontal else 3.2
    fig, ax = new_axes(title, (7.16, height))
    draw_counts(ax, pairs, unit, colors, horizontal)
    return fig


def safe_pie(ax, counts, colors, legend_ax):
    labels, values = zip(*counts.items())
    require(sum(values) > 0 and all(v >= 0 for v in values), "pie counts")
    labels = [" / ".join(k) if isinstance(k, tuple) else str(k) for k in labels]
    labels = [SOURCE_DISPLAY_NAMES.get(label, label) for label in labels]
    # Compute labels from exact counts, never Matplotlib's rounded wedge shares.
    proportions = [fraction(value, sum(values)) for value in values]
    label_iterator = iter([format_percent(value) if value >= Decimal("0.08") else ""
                           for value in proportions])
    wedges, _, texts = ax.pie(values, colors=colors, startangle=90, shadow=False,
                             autopct=lambda ignored: next(label_iterator), pctdistance=.67,
                             textprops={"color": "black", "fontsize": 8},
                             wedgeprops={"edgecolor": "none"})
    legend_ax.set_axis_off()
    legend_ax.legend(wedges, [f"{label} — {value:,}" +
                    (f" ({format_percent(proportion)})" if proportion < Decimal(".08") else "")
                    for label, value, proportion in zip(labels, values, proportions)],
                    loc="center", frameon=False, fontsize=7, ncol=1)


def plot_dataset_overview(s, pies=False, splits=False, combined=False):
    from matplotlib import pyplot as plt
    closures, counts, summary, _ = load_phase1_dataset_stats(s)
    split_values = load_split_stats(s) if splits or combined else None
    if pies:
        fig = plt.figure(figsize=(7.16, 4.8), layout="constrained")
        grid = fig.add_gridspec(3, 2, height_ratios=[3, 1.25, .45])
        safe_suptitle(fig, "Dataset Split Composition" if splits else "Dataset Class Composition")
        for i, ds in enumerate(DATASETS):
            ax = fig.add_subplot(grid[0, i])
            legend_ax = fig.add_subplot(grid[1, i])
            ax.set_title(f"({chr(97+i)}) {ds}", loc="left", fontsize=9, pad=8)
            values = dict(zip(SPLITS, split_values[i])) if splits else counts[i]
            safe_pie(ax, values, SPLIT_COLORS if splits else SOURCE_PALETTE, legend_ax)
        note_ax = fig.add_subplot(grid[2, :])
        note_ax.set_axis_off()
        note = "TRAIN / validation / TEST partitions" if splits else (
            f"USTC: {len(counts[0])} target classes; {summary['semantic_family_count']} original families     "
            f"CIC: {len(counts[1])} target classes")
        note_ax.text(.5, .5, note, ha="center", va="center", fontsize=7)
        return fig
    fig, axes = plt.subplots(2 if combined else 1, 2, figsize=(7.16, 5.6 if combined else 3.5),
                             layout="constrained", squeeze=False)
    for i, ds in enumerate(DATASETS):
        values = dict(zip(SPLITS, split_values[i])) if splits else counts[i]
        ax = axes[0, i]
        ax.set_title(f"({chr(97+i)}) {ds}", loc="left", weight="bold")
        draw_counts(ax, [(" / ".join(k) if isinstance(k, tuple) else k, v) for k, v in values.items()],
                    "Records / flows", SPLIT_COLORS if splits else ("#0072B2", "#D55E00")[i], horizontal=not splits)
        if not splits:
            suffix = f"{summary['semantic_family_count']} original families" if i == 0 else "fixed four-class target"
            ax.set_xlabel(f"{len(counts[i])} target classes; analysis n={sum(counts[i].values()):,}\n{suffix}", fontsize=7)
        if combined:
            lower = axes[1, i]
            draw_counts(lower, list(zip(SPLITS, split_values[i])), "Records / flows", SPLIT_COLORS, horizontal=False)
            lower.set_title(f"({chr(99+i)}) Dataset splits", loc="left", weight="bold")
    return fig


def plot_family_distribution(s, split_assignment=False):
    filename = USTC_PRE + "split_family_assignment.csv" if split_assignment else USTC_DATA + "family_distribution.csv"
    rows = s.rows(filename)
    require(len(unique_rows(rows, ("semantic_family",), "USTC families")) == 20, "USTC family metadata")
    require(sum(int(r["flow_count"]) for r in rows) == int(load_phase1_dataset_stats(s)[2]["total_flows"]), "family population")
    rows = sorted(rows, key=lambda r: (-int(r["flow_count"]), r["semantic_family"]))
    colors = [SPLIT_COLORS[SPLITS.index(r["split"])] for r in rows] if split_assignment else "#4477AA"
    labels = [(r["semantic_family"] + (" / " + r["split"] if split_assignment else ""), int(r["flow_count"])) for r in rows]
    fig, ax = new_axes("USTC families — descending flow count", (7.16, 6.4))
    draw_counts(ax, labels, "Flows", colors)
    return fig


def plot_preparation(s, retention=False):
    from matplotlib import pyplot as plt
    closures, _, _, contract = load_phase1_dataset_stats(s)
    extraction = s.rows(USTC_DATA + "processing_summary.csv")[0]
    pre = s.rows(USTC_PRE + "processing_summary.csv")[0]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.2), layout="constrained")
    for i, ax in enumerate(axes):
        before = int(pre["input_rows"]) if i == 0 else int(contract["raw_rows"])
        after = sum(int(closures[i][f"{split}_rows"]) for split in SPLITS)
        require(before >= after, "preprocessing population expansion")
        if retention:
            pairs = [("Before preprocessing", before), ("Removed records", before - after), ("Final usable records", after)]
        elif i == 0:
            pairs = [("Extracted flows", int(extraction["output_flow_count"])), ("Preprocessing input flows", before), ("Final usable records", after)]
        else:
            pairs = [("Source CSV records", before), ("Final usable records", after)]
        draw_counts(ax, pairs, "Records / flows", ("#0072B2", "#D55E00")[i])
        ax.set_title(DATASETS[i], loc="left", fontsize=9)
    if not retention:
        axes[0].text(.5, -.25, f"Input: {extraction['source_pcap_count']} PCAP files (separate unit)", transform=axes[0].transAxes, ha="center", fontsize=7)
    return fig


def plot_feature_roles(s):
    from matplotlib import pyplot as plt
    rows = [s.rows(USTC_DATA + "feature_summary.csv"), s.rows("metadata/traffic/cic_feature_contract.csv")]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.4), layout="constrained")
    for i, ax in enumerate(axes):
        counts = checked_counts(rows[i], ("role",), None)
        require(sum(counts.values()) == len(rows[i]), "feature role total")
        draw_counts(ax, [(k[0], v) for k, v in counts.items()], "Schema columns", ("#0072B2", "#D55E00")[i])
        ax.set_title(DATASETS[i] + " — saved schema roles", loc="left", fontsize=9)
    return fig


def baseline_metric_label(dataset, key):
    """Display the saved baseline definition explicitly, without changing values."""
    labels = {
        "accuracy": "Accuracy", "macro_f1": "Macro F1",
        "macro_precision": "Macro Precision", "macro_recall": "Macro Recall",
    }
    if key in ("f1", "precision", "recall"):
        require(dataset == DATASETS[0], "binary baseline metric requires USTC")
        return {"f1": "Binary F1", "precision": "Binary Precision", "recall": "Binary Recall"}[key]
    return labels[key]


def draw_model_metrics(ax, rows, metric_keys, colors):
    import numpy as np
    datasets = {row["dataset"] for row in rows}
    require(len(datasets) == 1, "baseline metric panels must not pool datasets")
    dataset = next(iter(datasets))
    width = .36
    model_colors = {name: MODEL_PALETTE[i] for i, name in enumerate(sorted(r["model"] for r in rows))}
    for j, key in enumerate(metric_keys):
        values = [finite_number(r[key], True) * 100 for r in rows]
        x = np.arange(len(rows)) + (j - .5) * width
        # Model identity uses the same solid color in every baseline/ROC figure.
        # Metric identity is explicitly left/right within each group, not a hatch.
        ax.bar(x, values, width=width, color=[model_colors[r["model"]] for r in rows], edgecolor="none")
        for pos, value, row in zip(x, values, rows):
            ax.annotate(format_percent(row[key]), (pos, value), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", rotation=90, fontsize=6.5)
    ax.set_xticks(range(len(rows)), [r["model"].replace("_", "\n") for r in rows])
    ax.set_ylim(0, 127)
    ax.set_yticks([0, 25, 50, 75, 100])
    percentage_axis(ax, "y")
    ax.set_ylabel("Validation performance (%)")
    ax.text(.5, -.30, "Left: " + baseline_metric_label(dataset, metric_keys[0]) + "   |   Right: " + baseline_metric_label(dataset, metric_keys[1]),
            transform=ax.transAxes, ha="center", va="top", fontsize=7, color="black")
    ax.spines[["top", "right"]].set_visible(False)


def plot_model_comparison(s, dataset_index=None, matrix=False):
    from matplotlib import pyplot as plt
    import numpy as np
    all_rows = load_model_comparison(s)
    if dataset_index is None and not matrix:
        fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.8), layout="constrained")
        safe_suptitle(fig, "Five-Model Comparison")
        for i, ax in enumerate(axes):
            keys = ("accuracy", "f1" if i == 0 else "macro_f1")
            draw_model_metrics(ax, all_rows[i], keys, ("#0072B2", "#009E73"))
            ax.set_title(f"({chr(97+i)}) {DATASETS[i]}\nAccuracy and "
                         + baseline_metric_label(DATASETS[i], keys[1]), loc="left", fontsize=9)
        return fig
    indices = [0, 1] if dataset_index is None else [dataset_index]
    fig, axes = plt.subplots(len(indices), 2, figsize=(7.16, 3.4 * len(indices)), layout="constrained", squeeze=False)
    safe_suptitle(fig, "Five-Model Comparison")
    for panel, i in enumerate(indices):
        rows = all_rows[i]
        keys = ("accuracy", "f1", "precision", "recall") if i == 0 else ("accuracy", "macro_f1", "macro_precision", "macro_recall")
        for j in range(2):
            ax = axes[panel, j]
            selected_keys = keys[j*2:j*2+2]
            if matrix:
                values = [[finite_number(row[key], True) for key in selected_keys] for row in rows]
                from matplotlib.colors import LinearSegmentedColormap
                cmap = LinearSegmentedColormap.from_list("light_metrics", plt.cm.Blues(np.linspace(.05, .6, 256)))
                ax.imshow(values, vmin=0, vmax=1, cmap=cmap, aspect="auto", interpolation="nearest")
                ax.set_xticks(range(2), [baseline_metric_label(DATASETS[i], k) for k in selected_keys])
                ax.set_xlabel("Validation metrics")
                ax.set_yticks(range(5), [r["model"] for r in rows])
                for y, row in enumerate(rows):
                    for x, key in enumerate(selected_keys):
                        ax.text(x, y, format_percent(row[key]), ha="center", va="center", fontsize=7)
            else:
                draw_model_metrics(ax, rows, selected_keys, ("#0072B2", "#009E73") if j == 0 else ("#56B4E9", "#D55E00"))
            ax.set_title(f"{DATASETS[i]}\n" + " and ".join(
                baseline_metric_label(DATASETS[i], key) for key in selected_keys), loc="left", fontsize=9)
    return fig


def draw_roc(s, ax, i, selected_only, legend_ax=None):
    curves = load_roc_curves(s, i, selected_only)
    for j, (label, (points, auc)) in enumerate(curves.items()):
        ax.plot([p[0] for p in points], [p[1] for p in points], linestyle="-", linewidth=1.7,
                color=SOURCE_PALETTE[j], label=f"{label}; AUC {format_percent(auc)}")
    ax.plot([0, 1], [0, 1], color=".6", linestyle="--", linewidth=.7)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="False-positive rate", ylabel="True-positive rate")
    ax.set_title(DATASETS[i] + " TEST ROC", loc="left", fontsize=9, pad=10)
    ax.xaxis.labelpad = 8
    ax.yaxis.labelpad = 8
    if legend_ax is None:
        place_external_legend(ax)
    else:
        _place_bottom_legend(legend_ax, ax, ncol=1 if i == 0 else 2)
    clean_axes(ax)


def plot_roc_curves(s, dataset_index=None):
    from matplotlib import pyplot as plt
    indices = [0, 1] if dataset_index is None else [dataset_index]
    fig = plt.figure(figsize=(7.16, 4.8), layout=None)
    fig._publication_layout_locked = True
    grid = fig.add_gridspec(2, len(indices), height_ratios=(3, .85),
                           left=.10, right=.98, top=.82, bottom=.055,
                           hspace=.55, wspace=.40)
    for column, i in enumerate(indices):
        ax = fig.add_subplot(grid[0, column])
        legend_ax = fig.add_subplot(grid[1, column])
        draw_roc(s, ax, i, selected_only=(dataset_index is not None or i == 1), legend_ax=legend_ax)
        if dataset_index is None:
            ax.set_title(f"({chr(97+i)}) {DATASETS[i]}", loc="left", fontsize=9)
    if dataset_index is None:
        safe_suptitle(fig, "TEST ROC Curves")
    return fig


def plot_model_roc_composition(s):
    from matplotlib import pyplot as plt
    models = load_model_comparison(s)
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 6.5), layout="constrained")
    for i in range(2):
        draw_model_metrics(axes[i, 0], models[i], ("accuracy", "f1" if i == 0 else "macro_f1"), ("#0072B2", "#009E73"))
        axes[i, 0].set_title(DATASETS[i] + " — Accuracy and "
                            + baseline_metric_label(DATASETS[i], "f1" if i == 0 else "macro_f1"),
                            loc="left", fontsize=9)
        draw_roc(s, axes[i, 1], i, selected_only=True)
    return fig


def plot_query_confidence(s, by_class=False, saved_histogram=False):
    from matplotlib import pyplot as plt
    import numpy as np
    rows = load_stage1_query_stats(s)["records"]
    if by_class:
        fig, ax = new_axes("Confidence by Class")
        values = [[finite_number(r["classifier_confidence"], True) for r in rows
                   if (r["source_dataset"], r["classifier_prediction"]) == key] for key in CLASS_KEYS]
        require(all(values), "confidence class records")
        box = ax.boxplot(values, vert=False, patch_artist=True, showfliers=False)
        for patch in box["boxes"]:
            patch.set_facecolor("#8FB5CE")
        for line in box["medians"]:
            line.set_color("black")
        ax.set_yticks(range(1, 7), [f"{'USTC' if ds == DATASETS[0] else 'CIC'} / {c}" for ds, c in CLASS_KEYS])
        ax.set_xlabel("Confidence (outliers omitted from box display)")
        return fig
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3), layout="constrained")
    hist = s.stage("stage01_structured_queries/test_query_confidence_histogram.csv") if saved_histogram else []
    for i, (ax, ds) in enumerate(zip(axes, DATASETS)):
        subset = [r for r in rows if r["source_dataset"] == ds]
        if saved_histogram:
            bins = [r for r in hist if r["source_dataset"] == ds]
            require(sum(int(r["count"]) for r in bins) == len(subset), "confidence histogram support")
            ax.bar([int(r["confidence_bin"]) for r in bins], [int(r["count"]) for r in bins],
                   width=1, align="edge", color=("#0072B2", "#D55E00")[i], edgecolor="none")
            ax.set_xlabel("Saved confidence bin (%)")
            ax.set_ylabel("Queries")
        else:
            values = sorted(finite_number(r["classifier_confidence"], True) for r in subset)
            ax.step(values, np.arange(1, len(values)+1) / len(values), where="post", color=("#0072B2", "#D55E00")[i])
            ax.set_xlabel("Classifier confidence")
            ax.set_ylabel("Empirical cumulative proportion")
        ax.set_title(ds, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    return fig


def plot_final_metrics(s):
    from matplotlib import pyplot as plt
    metrics = s.final()["metrics"]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.5), layout="constrained")
    safe_suptitle(fig, "Classification Performance")
    keys = ("accuracy", "macro_f1", "weighted_f1")
    for ax, ds, title in zip(axes, ("OVERALL",) + DATASETS, ("Overall",) + DATASETS):
        values = [metrics[ds, key]["value"] for key in keys]
        bars = ax.bar(range(3), [float(v)*100 for v in values], width=.65,
                      color=("#0072B2", "#009E73", "#8064A2"), edgecolor="none")
        annotate_bars(ax, bars, [format_percent(v) for v in values])
        ax.set_xticks(range(3), ["Accuracy", "Macro\nF1", "Weighted\nF1"])
        ax.set_title(title, fontsize=9, pad=10)
        ax.set_ylim(0, 110)
        ax.set_yticks([0, 25, 50, 75, 100])
        percentage_axis(ax, "y")
        clean_axes(ax)
    axes[0].set_ylabel("Performance (%)")
    return fig


def plot_integrity_audit(s):
    rows = s.rows("results/audits/USTC-TFC2016/integrity_audit/ustc_cross_split_duplicate_check.csv")
    subset = [r for r in rows if r["comparison"] == "predictors"]
    require(len(subset) == 3, "three duplicate comparisons")
    counts = {(r["left_split"], r["right_split"]): int(r["shared_distinct_rows"]) for r in subset}
    return count_figure(counts, "USTC audit: shared distinct predictor rows (not provenance leakage)", "Distinct numeric predictor rows")


def plot_query_probabilities(s):
    rows = s.stage("stage01_structured_queries/test_query_probability_summary.csv")
    indexed = unique_rows(rows, ("source_dataset", "class_label"), "probability summaries")
    require(set(indexed) == set(CLASS_KEYS), "probability summary classes")
    fig, ax = new_axes("Query Probability Summary")
    for i, key in enumerate(CLASS_KEYS):
        row = indexed[key]
        value = finite_number(row["mean_probability"], True)
        require(finite_number(row["min_probability"], True) <= value <= finite_number(row["max_probability"], True), "probability bounds")
        require(int(row["count"]) == (1700 if key[0] == DATASETS[0] else 2600), "probability summary denominator")
        ax.barh(i, value*100, color=("#0072B2" if key[0] == DATASETS[0] else "#D55E00"), edgecolor="none")
        ax.annotate(format_percent(row["mean_probability"]), (value*100, i), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7)
    for ds in DATASETS:
        require(abs(sum(Decimal(r["mean_probability"]) for r in rows if r["source_dataset"] == ds) - 1) < Decimal("1e-10"), "mean probability sum")
    ax.set_yticks(range(6), [f"{'USTC' if ds == DATASETS[0] else 'CIC'} / {c}" for ds, c in CLASS_KEYS])
    ax.invert_yaxis()
    ax.set_xlim(0, 65)
    ax.set_xlabel("Mean probability (%) — not per-class accuracy")
    return fig


def plot_corpus_composition(s, pie=False, claims=False):
    from matplotlib import pyplot as plt
    documents = load_stage2_corpus_stats(s)
    counts = checked_counts(documents, ("source_type",), None)
    if claims:
        records = s.rows("metadata/rag/v5f_validation/stage02_corpus/decision_evidence_claims.csv")
        unique_rows(records, ("claim_id",), "claims")
        document_sources = {r["document_id"]: r["source_type"] for r in documents}
        require(all(r["document_id"] in document_sources and r["source_type"] == document_sources[r["document_id"]] for r in records), "claim/document linkage")
        claim_counts = checked_counts(records, ("source_type",), None)
        # Document counts already appear in Q05/Q06; only claims are new here.
        return count_figure(claim_counts, "Corpus Claim Counts", "Decision claims")
    if pie:
        if len(counts) > 6:
            raise SourceNotFound("Corpus pie not applicable: more than six saved categories; use Q04 bars.")
        fig = plt.figure(figsize=(7.16, 3.8), layout="constrained")
        grid = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])
        safe_suptitle(fig, "Evidence Corpus Composition")
        safe_pie(fig.add_subplot(grid[0, 0]), counts, SOURCE_PALETTE, fig.add_subplot(grid[0, 1]))
        return fig
    return count_figure(counts, "Corpus Source Distribution", "Documents", SOURCE_PALETTE[:len(counts)])


def plot_index_stats(s, kind):
    rows, ref = load_stage3_index_stats(s)
    if kind == "source":
        return count_figure(checked_counts(rows, ("source_type",), None), "Indexed source composition", "Chunks")
    if kind == "documents":
        per_document = checked_counts(rows, ("document_id",), None)
        distribution = Counter(per_document.values())
        require(sum(distribution.values()) == int(ref["document_count"]), "index document distribution")
        return count_figure(dict(sorted(distribution.items())), "Chunks per indexed document", "Documents", horizontal=False)
    saved = s.stage("stage03_index/index_token_distribution.csv")
    require(checked_counts(rows, ("chunk_token_count",), None) == checked_counts(saved, ("chunk_token_count",)), "saved index token distribution")
    fig, ax = new_axes("Indexed chunk token distribution")
    ax.hist([int(r["chunk_token_count"]) for r in saved], weights=[int(r["count"]) for r in saved],
            bins=24, color="#4477AA", edgecolor="none")
    ax.set_xlabel("Tokens per chunk")
    ax.set_ylabel("Chunks")
    ax.text(.98, .95, f"Configured token limit: {ref['token_limit']}", transform=ax.transAxes, ha="right", va="top", fontsize=7)
    return fig


def plot_stacked_counts(rows, group, category, title, unit, total=None):
    counts = checked_counts(rows, (group, category), total=total)
    groups = sorted({key[0] for key in counts}, key=int if group == "rank" else str)
    categories = sorted({key[1] for key in counts})
    require(len(categories) <= len(SOURCE_PALETTE), "too many categories for source palette")
    fig, ax = new_axes(title, (7.16, 3.8))
    bottom = [0] * len(groups)
    for i, category_name in enumerate(categories):
        values = [counts[g, category_name] for g in groups]
        ax.bar(range(len(groups)), values, bottom=bottom, color=SOURCE_PALETTE[i], edgecolor="none", label=category_name)
        bottom = [a + b for a, b in zip(bottom, values)]
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylabel(unit)
    handles, labels = ax.get_legend_handles_labels()
    place_external_legend(ax, handles, [SOURCE_DISPLAY_NAMES.get(label, label) for label in labels], ncol=2)
    return fig


def plot_retrieval(s, kind):
    counts, rows = load_stage4_retrieval_stats(s)
    if kind == "methods":
        return count_figure(counts, "Retrieval candidates by saved method", "Candidate occurrences", horizontal=False)
    if kind == "sources":
        return plot_stacked_counts(rows, "retrieval_method", "source_type", "Retrieval source composition", "Candidate occurrences", 129000)
    rank_rows = s.stage("stage04_retrieval/retrieval_rank_distribution.csv")
    require(checked_counts(rank_rows, ("retrieval_method",), total=129000) == counts, "retrieval rank totals")
    return plot_stacked_counts(rank_rows, "rank", "retrieval_method", "Saved retrieval top-k rank composition", "Candidate occurrences", 129000)


def plot_provenance_verification(s, sources=False):
    rows = load_stage5_provenance_stats(s)
    if sources:
        rows = s.stage("stage05_provenance/provenance_source_type_summary.csv")
        counts = checked_counts(rows, ("source_type", "verification_status", "authorization_status"), total=129000)
        return count_figure(counts, "Verified evidence source distribution", "Candidate occurrences")
    counts = checked_counts(rows, ("verification_status", "authorization_status"), total=129000)
    return count_figure(counts, "Verification / downstream authorization", "Candidate occurrences")


def plot_reranking_comparison(s, kind):
    rows = load_stage6_reranking_stats(s)
    if kind == "sources":
        return count_figure(checked_counts(rows, ("source_type",), None, 43000), "Reranked evidence source composition", "Reranked candidate occurrences")
    if kind == "scores":
        fig, ax = new_axes("Saved reranking score distribution")
        ax.hist([finite_number(r["reranking_score"]) for r in rows], bins=30, color="#4477AA", edgecolor="none")
        ax.set_xlabel("Saved reranking score")
        ax.set_ylabel("Candidate occurrences")
        return fig
    from matplotlib import pyplot as plt
    _, retrieved = load_stage4_retrieval_stats(s)
    methods = {r["original_method"] for r in rows}
    before = checked_counts([r for r in retrieved if r["retrieval_method"] in methods], ("source_type",))
    after = checked_counts(rows, ("source_type",), None, 43000)
    require(sum(before.values()) > 0, "reranking input methods absent from retrieval")
    labels = sorted(set(before) | set(after))
    fig, ax = new_axes("Retrieval → reranking: within-stage source shares", (7.16, 4.3))
    import textwrap
    for j, (counts, name) in enumerate(((before, "Retrieval: " + ", ".join(sorted(methods))), (after, "Reranked"))):
        values = [fraction(counts.get(k, 0), sum(counts.values())) for k in labels]
        positions = [i + (j - .5) * .35 for i in range(len(labels))]
        ax.bar(positions, [float(v*100) for v in values], width=.35, color=("#4477AA", "#BBBBBB")[j], edgecolor="none", label=f"{name} (n={sum(counts.values()):,})")
        for x, value in zip(positions, values):
            ax.annotate(format_percent(value), (x, float(value*100)), xytext=(0, 3), textcoords="offset points", rotation=90, ha="center", fontsize=6.5)
    ax.set_xticks(range(len(labels)), [textwrap.fill(k[0], 20) for k in labels])
    ax.set_ylim(0, 125)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Share within each stage (%)")
    place_external_legend(ax, ncol=1)
    return fig


def plot_context_stats(s, kind):
    chunks = load_stage7_context_stats(s)
    if kind == "source":
        counts = checked_counts(chunks, ("source_type",), None, 21500)
        return count_figure(counts, "Trusted-context source composition", "Context chunk occurrences")
    if kind == "coverage":
        rows = s.stage("stage07_trusted_contexts/trusted_context_summary.csv")
        counts = checked_counts(rows, ("context_status",), total=4300)
        require(counts == {("READY",): 4300}, "ready context summary")
        return count_figure(counts, "Trusted-context coverage", "Queries")
    if kind == "diversity":
        rows = s.stage("stage07_trusted_contexts/trusted_context_document_diversity.csv")
        documents = {}
        for row in chunks:
            documents.setdefault(row["query_id"], set()).add(row["document_id"])
        counts = checked_counts(rows, ("distinct_document_count",), total=4300)
        require(Counter({int(k[0]): v for k, v in counts.items()}) == Counter(map(len, documents.values())), "context diversity summary")
        return count_figure(counts, "Distinct documents per trusted context", "Contexts", horizontal=False)
    rows = s.stage("stage07_trusted_contexts/trusted_context_chunk_distribution.csv")
    counts = checked_counts(rows, ("chunk_count",), total=4300)
    require(sum(int(k[0])*v for k, v in counts.items()) == len(chunks), "context chunks distribution")
    return count_figure(counts, "Chunks per trusted context", "Contexts", horizontal=False)


def plot_evidence_status(s, kind):
    data = s.final()
    if kind == "combined":
        return plot_evidence_and_correctness(data)
    if kind == "classes":
        fig, ax = new_axes("Evidence Status by Class", (7.16, 3.6))
        mesh = light_heatmap(ax, data["class_evidence"], [f"{'USTC' if ds == DATASETS[0] else 'CIC'} / {c}" for ds, c in CLASS_KEYS], STATUSES)
        fig.colorbar(mesh, ax=ax, label="Row-normalized proportion")
        return fig
    if kind == "correctness":
        fig, ax = new_axes("Evidence status ≠ classifier correctness")
        for i, (correct, errors) in enumerate(data["correctness"]):
            value = fraction(correct, correct+errors)
            ax.scatter(float(value*100), i, color=STATUS_COLORS[i], s=35)
            ax.annotate(f"{format_percent(value)} ({errors} errors)", (float(value*100), i),
                        xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8)
        ax.set_yticks(range(4), STATUSES)
        ax.set_ylim(3.6, -.6)
        ax.set_xlim(96.5, 100.1)
        ax.set_xlabel("Accuracy within status (%) — axis begins at 96.5%")
        return fig
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(7.16, 3.4))
    fig.subplots_adjust(bottom=.26, top=.9)
    stacked_status_bars(ax, data["evidence"], [f"{label} (n={sum(v):,})" for label, v in zip(("Overall", "USTC", "CIC"), data["evidence"])])
    ax.set_title("Evidence Status", loc="left", fontsize=9)
    small = data["evidence"][1][2]
    fig.text(.5, .11, f"USTC INSUFFICIENT: {small} / {sum(data['evidence'][1]):,} ({format_percent(fraction(small, sum(data['evidence'][1])))})", ha="center", fontsize=7)
    status_legend(fig, ax)
    return fig


def plot_fusion_summary(s, matrix=False):
    rows = load_stage9_fusion_stats(s)
    if not matrix:
        counts = checked_counts(rows, ("fusion_reason",), total=4300)
        fig = count_figure(counts, "retain_classifier for every query; zero overrides", "Queries")
        return fig
    rows = s.stage("stage09_fusion/test_fusion_by_evidence_status.csv")
    counts = checked_counts(rows, ("evidence_status", "decision_source"), total=4300)
    require({k[1] for k in counts} == {"retain_classifier"}, "fusion matrix actions")
    data = s.final()
    require([counts[status, "retain_classifier"] for status in STATUSES] == data["evidence"][0], "fusion status totals")
    fig, ax = new_axes("Evidence Status under Decision Fusion", (7.16, 3.5))
    draw_counts(ax, [(status, counts[status, "retain_classifier"]) for status in STATUSES],
                "Number of TEST queries", STATUS_COLORS)
    ax.set_xlabel("Number of TEST queries\nOverrides: 0", labelpad=10)
    return fig


def plot_progression(s, query_level=False):
    if query_level:
        query_count = len(load_stage1_query_stats(s)["records"])
        contexts = len({r["query_id"] for r in load_stage7_context_stats(s)})
        valid = dict(load_stage8_llm_stats(s))["Accepted"]
        fusion = sum(int(r["count"]) for r in load_stage9_fusion_stats(s))
        final = sum(s.final()["evidence"][0])
        values = {"1 Structured queries": query_count, "7 Trusted contexts": contexts, "8 Valid responses": valid,
                  "9 Fusion decisions": fusion, "10 Final records": final}
        require(set(values.values()) == {4300}, "query-stage coverage")
        return count_figure(values, "Phase-II query-level record progression", "Query-level records")
    retrieved = sum(load_stage4_retrieval_stats(s)[0].values())
    verified = sum(int(r["count"]) for r in load_stage5_provenance_stats(s))
    reranked = len(load_stage6_reranking_stats(s))
    chunks = len(load_stage7_context_stats(s))
    return count_figure({"Retrieved candidates\n(method occurrences)": retrieved,
                         "Verified / authorized\n(candidate occurrences)": verified,
                         "Reranked candidates\n(output occurrences)": reranked,
                         "Trusted-context chunks\n(context occurrences)": chunks},
                        "Evidence-processing progression — occurrence counts", "Occurrences (stage-specific units; not query counts)")


def plot_transition(s):
    data = s.final()
    fig, ax = new_axes("Exact matched historical V4 → V5 transitions", (7.16, 3.5))
    light_heatmap(ax, data["transition"], STATUSES, STATUSES, percentages=False)
    ax.set_xlabel("V5 evidence status")
    ax.set_ylabel("V4 evidence status")
    return fig


def plot_validation_test(s):
    from matplotlib import pyplot as plt
    data = s.final()
    fig, ax = plt.subplots(figsize=(7.16, 3.3))
    fig.subplots_adjust(bottom=.22, top=.9)
    stacked_status_bars(ax, [data["validation"], data["evidence"][0]],
                        [f"Validation (n={sum(data['validation']):,})", f"TEST (n={sum(data['evidence'][0]):,})"])
    ax.set_title("Different cohorts — descriptive comparison", fontsize=9)
    status_legend(fig, ax)
    return fig


METHODOLOGY_CAPTION = (
    "Complete two-phase methodology. Phase I performs dataset-specific "
    "encrypted-traffic preparation and classification on USTC-TFC2016 and "
    "CIC-Darknet2020 and produces the predicted class, confidence, traffic "
    "representation, and traffic provenance. Phase II converts these outputs "
    "into a structured security query and performs evidence retrieval, "
    "provenance verification, reranking, trusted-context construction, "
    "constrained reasoning, and retain-classifier decision fusion to produce "
    "the final class, evidence status, and provenance-linked explanation.")

METHODOLOGY_BRANCHES = (
    ("USTC-TFC2016", "NFStream Flow Extraction", "Leakage-Safe Preprocessing",
     "Family-Disjoint Split", "Five-Model Evaluation"),
    ("CIC-Darknet2020", "CIC-Specific Preprocessing", "Group-Aware Split",
     "Four-Class Classification", "Classifier Evaluation"))

METHODOLOGY_STAGES = (
    "Structured Security Query", "Evidence Corpus", "Evidence Indexing",
    "RAG Retrieval", "Provenance Verification", "Evidence Reranking",
    "Trusted Context Construction", "Constrained LLM Reasoning",
    "Decision Fusion", "Final Classification")
METHODOLOGY_INTERFACE = ("Predicted Class", "Confidence / Probabilities",
                         "Traffic Representation", "Traffic Provenance")
METHODOLOGY_OUTPUTS = ("Final Class", "Evidence Status", "Provenance + Explanation")


def draw_stage_box(ax, bounds, label, color, fontsize=6, wrap=20):
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.colors import to_rgb
    import textwrap
    x, y, width, height = bounds
    # A single pale solid fill, with a darker outline; no gradients or shadows.
    fill = tuple(.89 + .11 * component for component in to_rgb(color))
    ax.add_patch(FancyBboxPatch((x, y), width, height,
                 boxstyle="round,pad=0.002,rounding_size=0.006",
                 facecolor=fill, edgecolor=color, linewidth=.7))
    display = "\n".join(textwrap.fill(line, wrap, break_long_words=False,
                                     break_on_hyphens=False) for line in label.split("\n"))
    ax.text(x+width/2, y+height/2, display, ha="center", va="center",
            fontsize=fontsize, color="#16202A", linespacing=1.05)
    return bounds


def draw_arrow(ax, start, end, via=()):
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.path import Path as MplPath
    points = [start, *via, end]
    path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO]*(len(points)-1))
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=4.5,
                                linewidth=.75, color="#364152"))


def draw_phase_header(ax, x, y, phase, description, width):
    ax.text(x, y, phase, ha="left", va="top", fontsize=9, weight="bold", color="#16202A")
    ax.text(x, y-.033, description, ha="left", va="top", fontsize=8, color="#16202A")
    ax.plot([x, x+width], [y-.066, y-.066], color="#D7DFE5", linewidth=.7)


def draw_output_block(ax, bounds, fields, color, fontsize=7.5, gap=.004, wrap=40):
    x, y, width, height = bounds
    row_height = (height-gap*(len(fields)-1))/len(fields)
    for i, field in enumerate(fields):
        draw_stage_box(ax, (x, y+height-(i+1)*row_height-i*gap, width, row_height),
                       field, color, fontsize=fontsize, wrap=wrap)


def assert_methodology_text(fig):
    """Artist-text QA only: does not draw the canvas or write any files."""
    from matplotlib.text import Text
    import re
    texts = [" ".join(t.get_text().split()) for t in fig.findobj(Text) if t.get_text()]
    joined = " ".join(texts)
    branch_labels = tuple(label for branch in METHODOLOGY_BRANCHES for label in branch)
    require(tuple(branch[0] for branch in METHODOLOGY_BRANCHES) == DATASETS,
            "methodology dataset branches")
    require(all(len(branch) == 5 for branch in METHODOLOGY_BRANCHES), "five main steps per branch")
    for label in branch_labels + METHODOLOGY_INTERFACE + METHODOLOGY_OUTPUTS:
        require(texts.count(label) == 1, "methodology field must occur once: " + label)
    for number, label in enumerate(METHODOLOGY_STAGES, 1):
        require(texts.count(f"{number}. {label}") == 1, "methodology stage: " + label)
    require(sorted(int(re.match(r"^(\d+)\. ", t).group(1)) for t in texts
                   if re.match(r"^\d+\. ", t)) == list(range(1, 11)), "methodology stage numbers")
    require(texts.count("retain_classifier") == 1, "fusion policy label")
    for phrase in ("Final TEST Evaluation", "Integrity Audit", "Leakage Audit",
                   "TEST closure", "validation comparison", "experiment summary"):
        require(phrase.casefold() not in joined.casefold(), "closure step in methodology: " + phrase)
    require(not re.search(r"frozen|cohort|\bV[45]\b|injection|poison", joined, re.I),
            "obsolete methodology visible text")
    require(abs(fig.get_figwidth()-3.5) < 1e-9, "methodology must be native 3.5-inch width")
    require(all(t.get_fontsize() >= 7 for t in fig.findobj(Text) if t.get_text() and t.get_visible()),
            "methodology text below 7 pt")


def plot_methodology_pipeline(s):
    """Native IEEE single-column geometry; no downscaling or raster diagram."""
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(3.5, 7.87))
    fig.subplots_adjust(left=.025, right=.975, bottom=.025, top=.94)
    fig._publication_layout_locked = True
    fig._publication_exact_size = True
    safe_suptitle(fig, "Complete Methodology Pipeline")
    # Remove two Phase-I row pitches (.074) from the coordinate span and
    # physical height together, preserving printed box sizes and all font sizes.
    ax.set(xlim=(0, 1), ylim=(0, .926))
    ax.set_axis_off()
    draw_phase_header(ax, .04, .916, "PHASE I", "Encrypted Traffic Classification", .92)
    endpoints = []
    for column, (labels, color) in enumerate(zip(METHODOLOGY_BRANCHES, ("#0072B2", "#D55E00"))):
        x = .04 + column*.51
        for row, label in enumerate(labels):
            y = .835-row*.037-.031
            draw_stage_box(ax, (x, y, .41, .031), label, color, fontsize=7.2, wrap=22)
            if row:
                draw_arrow(ax, (x+.205, y+.037), (x+.205, y+.031))
        endpoints.append((x+.205, y))
    require(abs(endpoints[0][1]-endpoints[1][1]) < 1e-12, "dataset branches must align")
    # Independent dataset branches converge; neither dataset feeds the other.
    for x, y in endpoints:
        draw_arrow(ax, (x, y), (.5, .615), via=((x, .650), (.5, .650)))
    ax.text(.04, .635, "PHASE-I OUTPUT", fontsize=8.5, weight="bold", va="center")
    draw_output_block(ax, (.06, .530, .88, .085), METHODOLOGY_INTERFACE,
                      "#56B4E9", fontsize=7.5, gap=.003, wrap=40)
    draw_arrow(ax, (.5, .530), (.5, .516))
    draw_phase_header(ax, .04, .503, "PHASE II", "Provenance-Aware RAG", .92)
    boxes = []
    for i, label in enumerate(METHODOLOGY_STAGES):
        row, column = divmod(i, 2)
        x, y = .04+column*.51, .420-row*.060-.049
        color = "#009E73" if i in (1, 2, 4, 5, 6) else "#6F5AA8"
        if i == 8:
            draw_stage_box(ax, (x, y, .41, .049), "", color, fontsize=7.2)
            ax.text(x+.205, y+.033, f"{i+1}. {label}", ha="center", va="center", fontsize=7.2)
            ax.text(x+.205, y+.012, "retain_classifier", ha="center", va="center", fontsize=7)
        else:
            draw_stage_box(ax, (x, y, .41, .049), f"{i+1}. {label}", color, fontsize=7.2, wrap=22)
        boxes.append((x, y, .41, .049))
        if column:
            draw_arrow(ax, (.45, y+.0245), (.55, y+.0245))
        elif row:
            # Previous right box to next left box, entirely in the inter-row lane.
            prior = boxes[i-1]
            lane = y+.0545
            draw_arrow(ax, (.755, prior[1]), (.245, y+.049),
                       via=((.755, lane), (.245, lane)))
    ax.text(.04, .112, "FINAL OUTPUT", fontsize=8.5, weight="bold", va="center")
    draw_arrow(ax, (.755, boxes[-1][1]), (.5, .094), via=((.755, .101), (.5, .101)))
    draw_output_block(ax, (.06, .014, .88, .080), METHODOLOGY_OUTPUTS,
                      "#E69F00", fontsize=7.5, gap=.004, wrap=40)
    assert_methodology_text(fig)
    return fig



def plot_query_dataset_composition(s):
    rows = load_stage1_query_stats(s)["records"]
    counts = checked_counts(rows, ("source_dataset",), None, 4300)
    require([counts[ds,] for ds in DATASETS] == [1700, 2600], "TEST dataset composition")
    return count_figure({(ds,): counts[ds,] for ds in DATASETS}, "Final TEST query dataset composition",
                        "Queries", ["#0072B2", "#D55E00"], horizontal=False)


def plot_index_inventory(s):
    from matplotlib import pyplot as plt
    rows, ref = load_stage3_index_stats(s)
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8), layout="constrained")
    for ax, label, value in zip(axes, ("Indexed documents", "Indexed chunks"),
                                (int(ref["document_count"]), len(rows))):
        draw_counts(ax, [(label, value)], label, horizontal=False)
    return fig


def plot_fusion_actions(s):
    # s.final() validates all query-level fusion/final predictions before export.
    s.final()
    counts = checked_counts(load_stage9_fusion_stats(s), ("decision_source",), total=4300)
    fig, ax = new_axes("Decision Fusion", (7.16, 2.5))
    total = sum(counts.values())
    bars = ax.barh([0], [total], height=.45, color="#4477AA", edgecolor="none")
    ax.set_yticks([0], ["retain_classifier"])
    ax.set_xlim(0, total*1.5)
    annotate_bars(ax, bars, [f"{total:,} / {total:,}\n{format_percent(fraction(total, total))}"], horizontal=True)
    ax.set_xlabel("Number of TEST queries\nOverrides: 0", labelpad=10)
    return fig


def plot_final_class_distribution(s):
    counts = checked_counts(s.final()["records"], ("source_dataset", "final_class"), None, 4300)
    return count_figure(counts, "Final Class Distribution", "Queries")


ACTIVE_TABLE_NUMBERS = tuple(range(1, 14)) + (16,)
TABLE_FIGURE_NAMES = (
    "TF01_Dataset_Summary", "TF02_Dataset_Split_Summary", "TF03_Five_Model_Comparison",
    "TF04_Final_TEST_Performance", "TF05_Per_Class_TEST_Performance", "TF06_Confusion_Matrix_Values",
    "TF07_PhaseII_Corpus_Index_Summary", "TF08_Retrieval_Provenance_Reranking_Summary",
    "TF09_Trusted_Context_Summary", "TF10_Evidence_Status_Distribution", "TF11_Evidence_Status_Correctness",
    "TF12_Fusion_Summary", "TF13_Final_Provenance_Summary", "TF14_Integrity_Audit_Checklist")


def table_result_scope(number):
    return "DEVELOPMENT_VALIDATION_MODEL_SELECTION" if number == 3 else "FROZEN_METADATA" if number in (1, 2, 16) else "FINAL_TEST"


def render_table_figure(s, number):
    """Render the same canonical table cells, split into readable row panels."""
    from matplotlib import pyplot as plt
    import textwrap
    headers, rows, percentage_columns, note = complete_table_data(s, number)
    require(rows and all(len(row) == len(headers) for row in rows), "visual table shape")
    display = [[format_percent(value) if i in percentage_columns else str(value)
                for i, value in enumerate(row)] for row in rows]
    # Stable readable wrapping; long audit names and provenance notes get space.
    widths = [max(10, min(42, max(len(str(header)), max(len(row[i]) for row in display))))
              for i, header in enumerate(headers)]
    total_width = sum(widths)
    col_widths = [width / total_width for width in widths]
    wrap_cell = lambda value, width: textwrap.fill(value, width=width, break_long_words=True)
    wrapped = [[wrap_cell(value, widths[i]) for i, value in enumerate(row)] for row in display]
    wrapped_headers = [wrap_cell(str(value), widths[i]) for i, value in enumerate(headers)]
    panels = [wrapped[start:start+22] for start in range(0, len(wrapped), 22)]
    columns = min(2, len(panels))
    panel_heights = [max(2.0, .20 * sum(max(cell.count("\n")+1 for cell in row) for row in panel) + 1.1)
                     for panel in panels]
    grid_rows = math.ceil(len(panels) / columns)
    row_heights = [max(panel_heights[i:i+columns]) for i in range(0, len(panels), columns)]
    fig, axes = plt.subplots(grid_rows, columns, figsize=(7.16 * columns, sum(row_heights) + .7),
                             squeeze=False, gridspec_kw={"height_ratios": row_heights})
    fig.subplots_adjust(left=.025, right=.975, top=.94, bottom=.06, hspace=.22, wspace=.08)
    for index, ax in enumerate(axes.flat):
        ax.set_axis_off()
        if index >= len(panels):
            continue
        panel = panels[index]
        table = ax.table(cellText=panel, colLabels=wrapped_headers, colWidths=col_widths,
                         cellLoc="left", colLoc="left", loc="upper center")
        table.auto_set_font_size(False)
        table.set_fontsize(7.5)
        line_counts = [max(cell.count("\n")+1 for cell in wrapped_headers)] + [max(cell.count("\n")+1 for cell in row) for row in panel]
        line_total = sum(line_counts) + .8 * len(line_counts)
        for (row, col), cell in table.get_celld().items():
            cell.set_height((line_counts[row] + .8) / line_total * .93)
            cell.set_facecolor("#EEEEEE" if row == 0 else "white")
            cell.set_edgecolor("black")
            cell.visible_edges = "B" if row in (0, len(panel)) else ""
            cell.set_linewidth(.55 if row == 0 else .35)
            cell.get_text().set_color("black")
            if row == 0:
                cell.get_text().set_weight("bold")
        if len(panels) > 1:
            ax.set_title(f"({chr(97+index)}) Rows {index*22+1}–{index*22+len(panel)}", loc="left", fontsize=9)
    title = TABLE_FIGURE_NAMES[ACTIVE_TABLE_NUMBERS.index(number)].split("_", 1)[1].replace("_", " ")
    fig.suptitle(title, fontsize=10, weight="bold")
    scope_note = "Development/validation; USTC Binary F1; CIC Macro F1." if number == 3 else (
        "Final TEST run; evidence compatibility is distinct from classifier correctness." if table_result_scope(number) == "FINAL_TEST" else
        "Saved dataset / integrity metadata; no experimental values changed.")
    fig.text(.025, .015, scope_note, fontsize=7, color="black")
    return fig


def table_figure_candidates():
    return [FigureCandidate(name, "table_figures", f"table {number}", "SUPPLEMENT",
                            "Visual rendering of canonical CSV/LaTeX table; " + table_result_scope(number),
                            lambda s, number=number: render_table_figure(s, number))
            for name, number in zip(TABLE_FIGURE_NAMES, ACTIVE_TABLE_NUMBERS)]


def figure_result_scope(candidate):
    if candidate.name.split("_")[0] in ("P12", "P13", "P14"):
        return "FINAL_TEST_EVALUATION"
    if candidate.folder == "table_figures":
        return table_result_scope(int(candidate.stage.split()[-1]))
    if candidate.folder in ("phase2", "final_test") or candidate.name.startswith("X"):
        return "FINAL_TEST"
    if candidate.stage.startswith("validation") or candidate.stage == "selection":
        return "DEVELOPMENT_VALIDATION_MODEL_SELECTION"
    if candidate.stage in ("historical TEST", "frozen outputs"):
        return "FINAL_TEST"
    return "METHODOLOGY" if candidate.stage == "methodology" else "FROZEN_METADATA"


def verify_final_source_scope(s, scope):
    if scope not in ("FINAL_TEST", "FINAL_TEST_EVALUATION"):
        return
    # Corpus metadata is a shared frozen resource explicitly referenced by the
    # TEST corpus, not a validation outcome. No validation result files allowed.
    shared_corpus_metadata = "metadata/rag/v5f_validation/stage02_corpus/decision_evidence_claims.csv"
    prefix = TEST_NAMESPACE.as_posix() + "/"
    require(all(path.startswith(prefix) or path == shared_corpus_metadata for path in s.used),
            "FINAL_TEST figure/table accessed a source outside the final run or its frozen corpus metadata")


@dataclass
class FigureCandidate:
    name: str
    folder: str
    stage: str
    recommendation: str
    message: str
    build: object
    duplicate_of: str = ""
    canonical_id: str = ""
    source_signature: str = ""
    visual_signature: str = ""
    output_path: str = ""
    render: bool = True
    suppression_reason: str = ""

    @property
    def figure_id(self):
        return self.name if self.name == "M01_Complete_Methodology_Architecture" else self.name.split("_")[0]


# Explicit semantic audit: same observations, variables, aggregation and message.
# Requested component/composite plots and visual tables retain distinct purposes.
DUPLICATE_FIGURES = {
    "P15": "Q32", "R1": "Q32",
    "P17": "Q34", "R2": "Q34",
    "R3": "Q28", "R5": "Q37",
    "P16": "Q33", "P20": "Q35", "Q01": "Q35",
    "Q31": "Q30", "M03": "P09",
    "P09a": "TF03", "P09b": "TF03", "P11": "TF03",
    "P13": "P12", "P14": "P12", "P18": "Q02",
    "M02": "P01", "Q04": "Q35", "Q08": "TF07", "Q11": "TF07",
    "Q14": "Q12", "Q23": "TF09", "Q24": "Q28", "Q26": "Q28",
    "Q27": "X01", "Q36": "Q28",
    "M01_Complete_Methodology_Architecture": "M01",
}

# Coverage/curation is explicit: a component or alternate aggregation is not
# falsely described as an identical source/visual signature.
CURATION_REASONS = {
    "P09a": "Detailed USTC validation metrics retained in TF03; accuracy/Binary F1 in P09.",
    "P09b": "Detailed CIC validation metrics retained in TF03; accuracy/Macro F1 in P09.",
    "P11": "Alternate metric matrix suppressed; exact validation metrics retained in TF03.",
    "P13": "USTC ROC is panel (a) of P12.", "P14": "CIC ROC is panel (b) of P12.",
    "P18": "Alternate confidence aggregation suppressed in favor of the Q02 histogram.",
    "M02": "Repackages P01 and P04; use the separate dataset and split figures.",
    "M03": "Repackages P09 and P12 and mixes validation with TEST; keep them separate.",
    "Q04": "Dataset totals are the marginal counts of the dataset-qualified Q35 classes.",
    "Q08": "Document/chunk inventory retained in the compact TF07 table.",
    "Q11": "Source-wise indexed chunk counts retained in TF07.",
    "Q14": "All saved method/rank cells contain 4,300 queries; fixed top-k adds no selection message beyond Q12.",
    "Q23": "All contexts contain five chunks; exact ready-context/chunk totals retained in TF09.",
    "Q24": "Evidence-status distribution is panel (a) of Q28.",
    "Q26": "SUPPORT_CLASS = SUPPORTED + CONTRADICTED; ABSTAIN = INSUFFICIENT + CONFLICTING. Q28 retains the finer groups.",
    "Q27": "Accepted response count retained in X01; rejected count is zero.",
    "Q36": "Correctness by evidence status is panel (b) of Q28.",
    "Q31": "Retention reasons map one-to-one to the Q30 evidence groups.",
    "M01_Complete_Methodology_Architecture": "Replaced by the new vector methodology; old raster is never loaded.",
}


def registry_signatures(candidate):
    """Declarative scientific inputs, independent of titles and output folders.

    File patterns describe source families and are expanded to exact paths before
    rendering. Actual read-source hashes are recorded separately during rendering.
    Covered components keep their own signatures; only exact copies share both.
    """
    key = candidate.figure_id
    exact = {k: v for k, v in DUPLICATE_FIGURES.items() if k not in CURATION_REASONS}
    key = exact.get(key, key)
    specs = {
        "P01": ("dataset", "dataset,class,flow_count", "sum", "bar", "dataset", "count", "two datasets"),
        "P02": ("dataset", "dataset,class,flow_count", "sum", "pie", "dataset", "share", "two datasets"),
        "P03": ("family", "semantic_family,flow_count", "sum", "bar", "family", "count", "one"),
        "P04": ("split", "split,count", "sum", "bar", "dataset", "count", "two datasets"),
        "P05": ("split", "split,count", "sum", "pie", "dataset", "share", "two datasets"),
        "P05b": ("family_split", "semantic_family,split,flow_count", "sum", "bar", "family/split", "count", "one"),
        "P06": ("preparation", "input_rows,output_flow_count,raw_rows", "stage counts", "bar", "dataset", "count", "two datasets"),
        "P07": ("preparation", "input_rows,usable_rows,removed_rows", "difference", "bar", "dataset", "count", "two datasets"),
        "P08": ("schema", "role", "count", "bar", "dataset", "count", "two datasets"),
        "P09": ("models", "model,accuracy,f1,macro_f1", "saved validation metrics", "grouped bar", "dataset/model", "percent", "two datasets"),
        "P09a": ("models", "model,accuracy,f1,precision,recall", "saved validation metrics", "grouped bar", "USTC/model", "percent", "two metric pairs"),
        "P09b": ("models", "model,accuracy,macro_f1,macro_precision,macro_recall", "saved validation metrics", "grouped bar", "CIC/model", "percent", "two metric pairs"),
        "P10": ("models", "selection_metric,tie_break", "documented ranking", "bar", "dataset/model", "none", "two datasets"),
        "P11": ("models", "model,accuracy,f1,macro_f1,precision,recall,macro_precision,macro_recall", "saved validation metrics", "heatmap", "dataset/model", "percent", "four metric panels"),
        "P12": ("roc", "query_id,true_class_label,classifier_probabilities", "tied-score ROC/AUC", "line", "dataset/class", "FPR/TPR", "two datasets"),
        "P13": ("roc", "query_id,true_class_label,classifier_probabilities", "tied-score ROC/AUC", "line", "USTC/malware", "FPR/TPR", "one"),
        "P14": ("roc", "query_id,true_class_label,classifier_probabilities", "tied-score ROC/AUC", "line", "CIC/four classes", "FPR/TPR", "one"),
        "P18": ("query", "classifier_confidence", "ECDF", "step", "dataset", "cumulative share", "two datasets"),
        "P19": ("query", "classifier_confidence,classifier_prediction", "quartiles", "box", "dataset/class", "none", "one"),
        "P21": ("audit", "left_split,right_split,shared_distinct_rows", "saved counts", "bar", "split pair", "count", "one"),
        "Q02": ("query", "confidence_bin,count", "saved histogram", "bar", "dataset", "count", "two datasets"),
        "Q03": ("query", "class_label,mean_probability", "mean", "bar", "dataset/class", "percent", "one"),
        "Q04": ("query", "source_dataset", "count", "bar", "dataset", "count", "one"),
        "Q05": ("corpus", "source_type,document_id", "count", "bar", "source", "count", "one"),
        "Q06": ("corpus", "source_type,document_id", "count", "pie", "source", "share", "one"),
        "Q07": ("claims", "source_type,claim_id", "count", "bar", "source", "count", "one"),
        "Q08": ("index", "document_count,chunk_count", "count", "bar", "unit", "count", "two inventories"),
        "Q09": ("index", "document_id,chunk_id", "chunks per document histogram", "bar", "chunk count", "count", "one"),
        "Q10": ("index", "chunk_token_count,count", "weighted histogram 24 bins", "histogram", "tokens", "count", "one"),
        "Q11": ("index", "source_type,chunk_id", "count", "bar", "source", "count", "one"),
        "Q12": ("retrieval", "retrieval_method,count", "sum", "bar", "method", "count", "one"),
        "Q13": ("retrieval", "retrieval_method,source_type,count", "sum", "stacked bar", "method/source", "count", "one"),
        "Q14": ("retrieval", "retrieval_method,rank,count", "sum", "stacked bar", "rank/method", "count", "one"),
        "Q15": ("provenance", "verification_status,authorization_status,count", "sum", "bar", "status", "count", "one"),
        "Q16": ("provenance", "source_type,verification_status,authorization_status,count", "sum", "bar", "source/status", "count", "one"),
        "Q17": ("rerank", "source_type", "count", "bar", "source", "count", "one"),
        "Q18": ("rerank", "reranking_score", "histogram 30 bins", "histogram", "score", "count", "one"),
        "Q19": ("retrieval+rerank", "original_method,source_type,count", "before/after shares", "grouped bar", "stage/source", "within-stage share", "one"),
        "Q20": ("context", "context_status,count", "sum", "bar", "status", "count", "one"),
        "Q21": ("context", "source_type", "count", "bar", "source", "count", "one"),
        "Q22": ("context", "distinct_document_count,count", "histogram", "bar", "document count", "count", "one"),
        "Q23": ("context", "chunk_count,count", "histogram", "bar", "chunk count", "count", "one"),
        "Q24": ("evaluation", "evidence_status,source_dataset", "count", "stacked bar", "dataset/status", "share", "one"),
        "Q25": ("evaluation", "evidence_status,final_class,source_dataset", "count", "heatmap", "dataset/class/status", "row share", "one"),
        "Q26": ("reasoning", "reasoning_decision,count", "sum", "bar", "decision", "count", "one"),
        "Q27": ("reasoning", "accepted,rejected", "count", "bar", "validity", "count", "one"),
        "Q28": ("evaluation", "evidence_status,source_dataset,correct", "count and accuracy", "stacked bar/scatter", "dataset/status", "share/accuracy", "distribution/correctness"),
        "Q29": ("fusion", "decision_source,count", "sum", "bar", "action", "count", "one"),
        "Q30": ("fusion", "evidence_status,decision_source,count", "sum", "bar", "status", "count", "one"),
        "Q31": ("fusion", "fusion_reason,count", "sum", "bar", "reason", "count", "one"),
        "Q32": ("evaluation", "source_dataset,true_class_label,final_class", "confusion counts", "imshow", "dataset/class", "raw annotations", "two datasets"),
        "Q33": ("evaluation", "accuracy,macro_f1,weighted_f1", "saved metrics", "bar", "dataset/metric", "percent", "overall and two datasets"),
        "Q34": ("evaluation", "precision,recall,f1,support", "saved class metrics", "dot", "dataset/class", "percent", "one"),
        "Q35": ("evaluation", "source_dataset,final_class", "count", "bar", "dataset/class", "count", "one"),
        "Q36": ("evaluation", "evidence_status,correct", "accuracy", "scatter", "status", "percent", "one"),
        "Q37": ("evaluation", "provenance_available,evidence_status", "count", "bar", "coverage", "share", "one"),
        "X01": ("query+context+reasoning+fusion+evaluation", "query_id,accepted,count", "stage coverage", "bar", "stage", "count", "one"),
        "X02": ("retrieval+provenance+rerank+context", "count,chunk_id", "stage occurrences", "bar", "stage", "count", "one"),
        "M01": ("methodology", "dataset branches,interface,10 stages,outputs", "workflow", "vector diagram", "phase", "none", "single column/two dataset branches/five paired stage rows"),
        "M01_Complete_Methodology_Architecture": ("legacy_methodology", "architecture raster", "none", "raster", "phase", "none", "legacy"),
        "M02": ("dataset+split", "dataset,class,split,count", "sum", "bar", "dataset/split", "count", "four panels"),
        "M03": ("models+roc", "validation metrics,TEST ROC", "mixed scope", "bar/line", "dataset", "percent/FPR/TPR", "four panels"),
    }
    groups = {
        "dataset": [USTC_DATA+"class_distribution.csv", USTC_DATA+"dataset_summary.csv", "metadata/traffic/*experiment_closure.csv", "metadata/traffic/cic_phase1_contract.csv"],
        "family": [USTC_DATA+"family_distribution.csv"],
        "family_split": [USTC_PRE+"split_family_assignment.csv"],
        "split": [USTC_PRE+"split_distribution.csv", "metadata/traffic/cic_split_manifest.csv"],
        "preparation": [USTC_DATA+"processing_summary.csv", USTC_PRE+"processing_summary.csv", "metadata/traffic/*experiment_closure.csv", "metadata/traffic/cic_phase1_contract.csv"],
        "schema": [USTC_DATA+"feature_summary.csv", "metadata/traffic/cic_feature_contract.csv"],
        "models": ["metadata/traffic/*experiment_closure.csv"],
        "audit": ["results/audits/*/integrity_audit/*.csv"],
        "methodology": ["prepare_final_paper_figures.py#METHODOLOGY_STAGES"],
        "legacy_methodology": ["fig01_complete_methodology_architecture.png"],
    }
    stage_dirs = dict(query="stage01_structured_queries", corpus="stage02_corpus", index="stage03_index",
                      retrieval="stage04_retrieval", provenance="stage05_provenance", rerank="stage06_reranking",
                      context="stage07_trusted_contexts", reasoning="stage08_llm", fusion="stage09_fusion", evaluation="stage10_final")
    for group, directory in stage_dirs.items():
        groups[group] = [(TEST_NAMESPACE / directory / "*.csv").as_posix()]
    groups["reasoning"].append((TEST_NAMESPACE / "stage08_llm/test_llm_rejected_responses.jsonl").as_posix())
    groups["claims"] = groups["corpus"] + ["metadata/rag/v5f_validation/stage02_corpus/decision_evidence_claims.csv"]
    groups["roc"] = [(TEST_NAMESPACE / "stage10_final" / filename).as_posix() for filename in
                     ("test_evaluation_records.csv", "test_final_classification_records.csv")]
    if key.startswith("TF"):
        table = int(key[2:])
        group = {1:"dataset", 2:"split", 3:"models", 7:"corpus+index", 8:"retrieval+provenance+rerank",
                 9:"context", 12:"fusion", 14:"audit"}.get(table, "evaluation")
        # The table number identifies an exact canonical column/row contract.
        related = {1:"P01", 2:"P04", 3:"P11", 4:"Q33", 5:"Q34", 6:"Q32", 7:"Q08",
                   8:"X02", 9:"Q22", 10:"Q24", 11:"Q36", 12:"Q31", 13:"Q37", 14:"P21"}[table]
        fields = specs[related][1]
        spec = (group, fields, f"canonical table {table} exact cells", "table", "rows/columns", "exact values", "paginated rows")
    else:
        require(key in specs, "missing registry scientific specification: " + key)
        spec = specs[key]
    root = Path(__file__).resolve().parent
    paths = set()
    for group in spec[0].split("+"):
        for pattern in groups[group]:
            matches = sorted(path.relative_to(root).as_posix() for path in root.glob(pattern)) if "*" in pattern else []
            paths.update(matches or [pattern])
        if group == "models":
            # Resolve authoritative model-result paths from the saved closures.
            for path in root.glob("metadata/traffic/*experiment_closure.csv"):
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    for row in csv.DictReader(stream):
                        if row.get("validation_results_path"):
                            paths.add(row["validation_results_path"])
    source = dict(source_paths=sorted(paths), plotted_fields=spec[1].split(","), aggregation=spec[2])
    visual = dict(chart_type=spec[3], x_y_variables=spec[1].split(","), grouping=spec[4], normalization=spec[5], panels=spec[6])
    return json.dumps(source, sort_keys=True), json.dumps(visual, sort_keys=True)


def deduplicate_figure_registry(candidates):
    indexed = {candidate.figure_id: candidate for candidate in candidates}
    require(len(indexed) == len(candidates), "duplicate registry IDs")
    for alias, canonical in DUPLICATE_FIGURES.items():
        require(alias in indexed and canonical in indexed and canonical not in DUPLICATE_FIGURES,
                "invalid canonical figure mapping")
        indexed[alias].duplicate_of = canonical
        indexed[alias].suppression_reason = CURATION_REASONS.get(alias, "Identical scientific result; one canonical physical output.")
        if indexed[alias].recommendation == "MAIN_REQUIRED":
            indexed[canonical].recommendation = "MAIN_REQUIRED"
    seen = {}
    for candidate in candidates:
        candidate.source_signature, candidate.visual_signature = registry_signatures(candidate)
        signature = (candidate.source_signature, candidate.visual_signature)
        if not candidate.duplicate_of and signature in seen:
            candidate.duplicate_of = seen[signature].figure_id
            candidate.suppression_reason = "Equivalent source, fields, aggregation and visual contract."
        if not candidate.duplicate_of:
            seen[signature] = candidate
        canonical = indexed[candidate.duplicate_of] if candidate.duplicate_of else candidate
        candidate.canonical_id = canonical.name
        candidate.output_path = str(Path(canonical.folder) / canonical.name)
        candidate.render = not bool(candidate.duplicate_of)
    require(len({(c.source_signature, c.visual_signature) for c in candidates if c.render}) ==
            sum(c.render for c in candidates), "duplicate physical scientific signatures")
    indexed["M01_Complete_Methodology_Architecture"].duplicate_of = "M01_Complete_Methodology_Pipeline"
    return candidates


def phase1_candidates():
    def candidate(name, stage, message, build, recommendation="SUPPLEMENT"):
        return FigureCandidate(name, "phase1", stage, recommendation, message, build)
    return [
        candidate("P01_Dataset_Overview", "dataset", "Project-level populations; original USTC families distinct from target labels.", lambda s: plot_dataset_overview(s), "MAIN_REQUIRED"),
        candidate("P02_Dataset_Class_Composition", "dataset", "Actual saved class counts and shares.", lambda s: plot_dataset_overview(s, pies=True)),
        candidate("P03_USTC_Family_Distribution", "dataset", "Original families, sorted by descending saved flow count.", plot_family_distribution),
        candidate("P04_Dataset_Split_Counts", "split", "Frozen partition counts; no new splitting.", lambda s: plot_dataset_overview(s, splits=True)),
        candidate("P05_Dataset_Split_Composition", "split", "Frozen partition shares; requested pie version.", lambda s: plot_dataset_overview(s, pies=True, splits=True)),
        candidate("P05b_USTC_Family_Disjoint_Split", "split", "Each saved semantic family occupies one partition.", lambda s: plot_family_distribution(s, True)),
        candidate("P06_Flow_Extraction_Preparation", "preparation", "PCAP files and flow records have different units; no invented reconstruction stages.", plot_preparation),
        candidate("P07_Preprocessing_Retention", "preprocessing", "Saved input and final populations; removed records are their exact difference.", lambda s: plot_preparation(s, True)),
        candidate("P08_Traffic_Representation", "schema", "Explicit saved schema roles, not invented feature groups.", plot_feature_roles),
        candidate("P09_Five_Model_Baseline_Comparison", "validation", "Frozen development/validation comparison. USTC Binary F1 and CIC Macro F1 are distinct; no pooling or TEST-based selection.", plot_model_comparison, "MAIN_REQUIRED"),
        candidate("P09a_USTC_Five_Model_Comparison", "validation", "Frozen development/validation comparison. USTC Binary F1.", lambda s: plot_model_comparison(s, 0)),
        candidate("P09b_CIC_Model_Comparison", "validation", "Frozen development/validation comparison. CIC Macro F1.", lambda s: plot_model_comparison(s, 1)),
        candidate("P10_Five_Model_Ranking", "selection", "A ranking must use the documented selection criterion, not an assumed macro F1.", missing_selection_rule),
        candidate("P11_Five_Model_Metric_Matrix", "validation", "Frozen development/validation comparison. Binary F1 and Macro F1 definitions retained.", lambda s: plot_model_comparison(s, matrix=True)),
        candidate("P12_PhaseI_Combined_ROC", "final TEST ROC", "Post-hoc ROC from completed TEST evaluation labels and exact matched frozen probabilities. No tuning or threshold selection; unweighted 4,300-query cohort.", plot_roc_curves, "MAIN_REQUIRED"),
        candidate("P13_USTC_Selected_Model_ROC", "final TEST ROC", "USTC positive-class TEST ROC from saved evaluation labels/probabilities; no inference.", lambda s: plot_roc_curves(s, 0)),
        candidate("P14_CIC_Multiclass_ROC", "final TEST ROC", "CIC four-class TEST OvR from saved evaluation labels/probabilities, not AUC summaries.", lambda s: plot_roc_curves(s, 1)),
        candidate("P15_Final_Confusion_Matrices", "historical TEST", "4,300-query historical cohort, not full Phase-I TEST population; raw cell counts only.", lambda s: plot_test_confusions(s.final()), "MAIN_REQUIRED"),
        candidate("P16_Final_Dataset_Performance", "historical TEST", "Unweighted 4,300-query cohort; accuracy, macro F1, weighted F1.", plot_final_metrics),
        candidate("P17_Per_Class_Performance", "historical TEST", "Per-class saved evaluation metrics; truncated axis disclosed.", lambda s: plot_per_class_performance(s.final()), "MAIN_OPTIONAL"),
        candidate("P18_Prediction_Confidence", "frozen outputs", "ECDF of actual saved historical cohort confidences, not full-population confidence.", plot_query_confidence),
        candidate("P19_Confidence_by_Class", "frozen outputs", "Saved individual confidence records grouped by frozen prediction.", lambda s: plot_query_confidence(s, by_class=True)),
        candidate("P20_Predicted_Class_Distribution", "frozen outputs", "Frozen classifier predictions; not ground-truth composition.", lambda s: count_figure(load_stage1_query_stats(s)["counts"], "Query Class Distribution", "Queries")),
        candidate("P21_Integrity_Audit", "integrity", "Shared numeric predictor rows are not automatically provenance leakage; categorical checks supplied as table.", plot_integrity_audit, "DIAGNOSTIC")]


def missing_selection_rule(s):
    load_model_comparison(s)
    s.touch("RESEARCH_PROTOCOL.md")
    raise SourceNotFound("P10 requires a frozen model-selection manifest naming the actual ranking metric and tie-break/selection rule. Protocol establishes validation-only selection but no single ranking metric; USTC saves binary F1, not macro F1.")


def phase2_candidates():
    def c(name, stage, message, build, recommendation="SUPPLEMENT"):
        return FigureCandidate(name, "phase2", str(stage), recommendation, message, build)
    return [
        c("Q01_TEST_Query_Class_Distribution", 1, "Frozen prediction composition.", lambda s: count_figure(load_stage1_query_stats(s)["counts"], "Final TEST structured query composition", "Queries")),
        c("Q02_TEST_Query_Confidence_Distribution", 1, "Saved TEST confidence histogram.", lambda s: plot_query_confidence(s, saved_histogram=True)),
        c("Q03_TEST_Query_Probability_Summary", 1, "Mean probability, not accuracy.", plot_query_probabilities),
        c("Q04_TEST_Query_Dataset_Composition", 1, "Query counts by dataset in the final run.", plot_query_dataset_composition),
        c("Q05_TEST_Evidence_Corpus_Composition", 2, "Frozen corpus referenced by this TEST run.", plot_corpus_composition),
        c("Q06_TEST_Evidence_Corpus_Source_Pie", 2, "Source composition; exact shares inside sufficiently large slices.", lambda s: plot_corpus_composition(s, pie=True)),
        c("Q07_TEST_Corpus_Document_Claim_Summary", 2, "Decision claims by source; document counts are confined to Q05/Q06. Frozen corpus metadata only.", lambda s: plot_corpus_composition(s, claims=True)),
        c("Q08_TEST_Index_Chunk_Summary", 3, "Saved index inventory; documents and chunks have distinct units.", plot_index_inventory),
        c("Q09_TEST_Chunks_Per_Document", 3, "Saved per-document chunk distribution.", lambda s: plot_index_stats(s, "documents")),
        c("Q10_TEST_Index_Token_Distribution", 3, "Saved token counts and configured limit.", lambda s: plot_index_stats(s, "tokens")),
        c("Q11_TEST_Index_Source_Composition", 3, "Indexed chunks by source type.", lambda s: plot_index_stats(s, "source")),
        c("Q12_TEST_Retrieval_Methods", 4, "Saved retrieval method occurrences.", lambda s: plot_retrieval(s, "methods")),
        c("Q13_TEST_Retrieval_Source_Distribution", 4, "Source composition within method.", lambda s: plot_retrieval(s, "sources")),
        c("Q14_TEST_Retrieval_Rank_Distribution", 4, "Saved rank-level counts.", lambda s: plot_retrieval(s, "ranks")),
        c("Q15_TEST_Provenance_Status", 5, "Actual verification/authorization categories.", plot_provenance_verification),
        c("Q16_TEST_Provenance_Source_Types", 5, "Verified evidence occurrences by source.", lambda s: plot_provenance_verification(s, True)),
        c("Q17_TEST_Reranked_Evidence_Composition", 6, "Final-run reranking output composition.", lambda s: plot_reranking_comparison(s, "sources")),
        c("Q18_TEST_Reranking_Score_Distribution", 6, "Saved reranking scores.", lambda s: plot_reranking_comparison(s, "scores")),
        c("Q19_TEST_Retrieval_vs_Reranking", 6, "Same TEST run, recorded input methods; separate stage denominators.", lambda s: plot_reranking_comparison(s, "comparison")),
        c("Q20_TEST_Trusted_Context_Coverage", 7, "Query-level ready contexts.", lambda s: plot_context_stats(s, "coverage")),
        c("Q21_TEST_Trusted_Context_Source_Composition", 7, "Context chunk occurrences, not selected citations.", lambda s: plot_context_stats(s, "source"), "MAIN_OPTIONAL"),
        c("Q22_TEST_Context_Document_Diversity", 7, "Distinct documents per context.", lambda s: plot_context_stats(s, "diversity")),
        c("Q23_TEST_Context_Chunk_Distribution", 7, "Chunks per context.", lambda s: plot_context_stats(s, "chunks")),
        c("Q24_TEST_Evidence_Status_Distribution", 8, CAPTIONS[2], lambda s: plot_evidence_status(s, "distribution")),
        c("Q25_TEST_Evidence_Status_By_Class", 8, "Compatibility by frozen prediction, not correctness.", lambda s: plot_evidence_status(s, "classes"), "MAIN_OPTIONAL"),
        c("Q26_TEST_Reasoning_Decision_Distribution", 8, "SUPPORT_CLASS does not override the classifier; schema-constrained reasoning.", lambda s: count_figure(checked_counts(s.stage("stage08_llm/test_llm_reasoning_decision_summary.csv"), ("reasoning_decision",), total=4300), "Final TEST constrained reasoning decisions", "Queries")),
        c("Q27_TEST_LLM_Response_Validity", 8, "Structured response validity, not narrative semantic validation.", lambda s: count_figure(dict(load_stage8_llm_stats(s)), "Final TEST Stage-8 response validity", "Responses")),
        c("Q28_TEST_Evidence_Status_and_Correctness", 8, CAPTIONS[2], lambda s: plot_evidence_status(s, "combined"), "MAIN_REQUIRED"),
        c("Q29_TEST_Fusion_Action_Distribution", 9, "retain_classifier; zero overrides.", plot_fusion_actions),
        c("Q30_TEST_Evidence_Status_vs_Fusion", 9, "Evidence states do not override predictions.", lambda s: plot_fusion_summary(s, True)),
        c("Q31_TEST_Fusion_Summary", 9, "Detailed saved retention reasons, not classifier corrections.", plot_fusion_summary),
        c("Q32_TEST_Final_Confusion_Matrices", 10, CAPTIONS[0], lambda s: plot_test_confusions(s.final())),
        c("Q33_TEST_Final_Overall_Performance", 10, "Saved final TEST performance.", plot_final_metrics),
        c("Q34_TEST_Per_Class_Performance", 10, "Saved per-class evaluation metrics.", lambda s: plot_per_class_performance(s.final())),
        c("Q35_TEST_Final_Class_Distribution", 10, "Final classes equal frozen predictions.", plot_final_class_distribution),
        c("Q36_TEST_Correctness_By_Evidence_Status", 10, "Compatibility is distinct from correctness.", lambda s: plot_evidence_status(s, "correctness")),
        c("Q37_TEST_Final_Provenance_Coverage", 10, CAPTIONS[4], lambda s: plot_provenance_coverage(s.final()), "MAIN_REQUIRED")]


EXPECTED_CANONICAL_NAMES = (
    'P01_Dataset_Overview',
    'P02_Dataset_Class_Composition',
    'P03_USTC_Family_Distribution',
    'P04_Dataset_Split_Counts',
    'P05_Dataset_Split_Composition',
    'P05b_USTC_Family_Disjoint_Split',
    'P06_Flow_Extraction_Preparation',
    'P07_Preprocessing_Retention',
    'P08_Traffic_Representation',
    'P09_Five_Model_Baseline_Comparison',
    'P12_PhaseI_Combined_ROC',
    'P19_Confidence_by_Class',
    'P21_Integrity_Audit',
    'Q02_TEST_Query_Confidence_Distribution',
    'Q03_TEST_Query_Probability_Summary',
    'Q05_TEST_Evidence_Corpus_Composition',
    'Q06_TEST_Evidence_Corpus_Source_Pie',
    'Q07_TEST_Corpus_Document_Claim_Summary',
    'Q09_TEST_Chunks_Per_Document',
    'Q10_TEST_Index_Token_Distribution',
    'Q12_TEST_Retrieval_Methods',
    'Q13_TEST_Retrieval_Source_Distribution',
    'Q15_TEST_Provenance_Status',
    'Q16_TEST_Provenance_Source_Types',
    'Q17_TEST_Reranked_Evidence_Composition',
    'Q18_TEST_Reranking_Score_Distribution',
    'Q19_TEST_Retrieval_vs_Reranking',
    'Q20_TEST_Trusted_Context_Coverage',
    'Q21_TEST_Trusted_Context_Source_Composition',
    'Q22_TEST_Context_Document_Diversity',
    'Q25_TEST_Evidence_Status_By_Class',
    'Q28_TEST_Evidence_Status_and_Correctness',
    'Q29_TEST_Fusion_Action_Distribution',
    'Q30_TEST_Evidence_Status_vs_Fusion',
    'Q32_TEST_Final_Confusion_Matrices',
    'Q33_TEST_Final_Overall_Performance',
    'Q34_TEST_Per_Class_Performance',
    'Q35_TEST_Final_Class_Distribution',
    'Q37_TEST_Final_Provenance_Coverage',
    'X01_TEST_PhaseII_Record_Progression',
    'X02_TEST_Evidence_Processing_Progression',
    'M01_Complete_Methodology_Pipeline',
    'TF01_Dataset_Summary',
    'TF02_Dataset_Split_Summary',
    'TF03_Five_Model_Comparison',
    'TF04_Final_TEST_Performance',
    'TF05_Per_Class_TEST_Performance',
    'TF06_Confusion_Matrix_Values',
    'TF07_PhaseII_Corpus_Index_Summary',
    'TF08_Retrieval_Provenance_Reranking_Summary',
    'TF09_Trusted_Context_Summary',
    'TF10_Evidence_Status_Distribution',
    'TF11_Evidence_Status_Correctness',
    'TF12_Fusion_Summary',
    'TF13_Final_Provenance_Summary',
    'TF14_Integrity_Audit_Checklist',
)


def complete_figure_candidates():
    candidates = phase1_candidates() + phase2_candidates()
    for name, message, function, rec in (
        ("X01_TEST_PhaseII_Record_Progression", "Same-run query coverage; candidate counts excluded.", lambda s: plot_progression(s, True), "SUPPLEMENT"),
        ("X02_TEST_Evidence_Processing_Progression", "Same TEST run; stage-specific occurrence units, not query counts.", plot_progression, "MAIN_REQUIRED")):
        candidates.append(FigureCandidate(name, "cross_phase", "cross-stage", rec, message, function))
    for i, function in ((0, lambda s: plot_test_confusions(s.final())),
                         (1, lambda s: plot_per_class_performance(s.final())),
                         (2, lambda s: plot_evidence_and_correctness(s.final())),
                         (4, lambda s: plot_provenance_coverage(s.final()))):
        rec = "MAIN_OPTIONAL" if i == 1 else "MAIN_REQUIRED"
        candidates.append(FigureCandidate(FIGURE_NAMES[i], "final_test", "final TEST", rec, CAPTIONS[i], function))
    candidates += [
        FigureCandidate("M01_Complete_Methodology_Pipeline", "cross_phase", "methodology", "MAIN_REQUIRED",
                        METHODOLOGY_CAPTION, plot_methodology_pipeline),
        FigureCandidate("M01_Complete_Methodology_Architecture", "cross_phase", "methodology", "SUPPLEMENT",
                        "Legacy architecture alias; replaced by the canonical vector pipeline.", None),
        FigureCandidate("M02_Dataset_and_Split_Overview", "cross_phase", "dataset / split", "MAIN_REQUIRED",
                        "Combined saved dataset populations and frozen partitions.", lambda s: plot_dataset_overview(s, combined=True)),
        FigureCandidate("M03_Model_Comparison_and_ROC", "cross_phase", "validation", "MAIN_OPTIONAL",
                        "Frozen development/validation comparison plus saved backbone ROC; USTC Binary F1 and CIC Macro F1 are distinct.", plot_model_roc_composition)]
    candidates += table_figure_candidates()
    require(len({c.name for c in candidates}) == len(candidates), "duplicate figure IDs")
    return deduplicate_figure_registry(candidates)


def audit_complete_registry(candidates):
    """Fail explicitly before cleanup if a required canonical entry disappears."""
    by_name = {candidate.name: candidate for candidate in candidates}
    by_id = {candidate.figure_id: candidate for candidate in candidates}
    require(len(by_name) == len(candidates), "duplicate registry basename")
    expected = set(EXPECTED_CANONICAL_NAMES)
    require(expected.issubset(by_name), "missing canonical registrations: " + ", ".join(sorted(expected-set(by_name))))
    for name in EXPECTED_CANONICAL_NAMES:
        candidate = by_name[name]
        require(candidate.render and not candidate.duplicate_of, "required canonical accidentally suppressed: " + name)
        require(callable(candidate.build), "missing plotting callable: " + name)
        require(candidate.canonical_id == name and candidate.output_path == str(Path(candidate.folder)/name),
                "canonical output path mismatch: " + name)
        require(candidate.source_signature and candidate.visual_signature, "missing scientific signatures: " + name)
    physical = [candidate for candidate in candidates if candidate.render]
    require({c.name for c in physical} == expected | {"P10_Five_Model_Ranking"}, "unexpected physical figure registry")
    require(len({c.output_path for c in physical}) == len(physical), "duplicate physical output path")
    for alias in candidates:
        if alias.duplicate_of:
            canonical = by_name.get(alias.duplicate_of) or by_id.get(alias.duplicate_of)
            require(canonical is not None and canonical.render and not alias.render, "invalid physical alias: " + alias.name)
    return {"expected_physical_figures": len(expected),
            "table_figures": sum(by_name[name].folder == "table_figures" for name in expected),
            "suppressed_aliases": sum(not c.render for c in candidates),
            "registered_missing": [], "known_unavailable": {"P10_Five_Model_Ranking":
            "Authoritative selection metric and tie-break rule are not saved."}}


def complete_table_data(s, number):
    """Return headers, exact/raw cells, fractional display columns, and scope note."""
    require(number in ACTIVE_TABLE_NUMBERS, "cross-version and validation/TEST comparison tables are excluded")
    if number in (1, 2):
        closures, counts, _, _ = load_phase1_dataset_stats(s)
        if number == 1:
            return ["Dataset", "Target class", "Count", "Population"], [
                [ds, key[0], n, sum(counts[i].values())] for i, ds in enumerate(DATASETS) for key, n in counts[i].items()], (), "Project-level analysis populations; USTC targets are distinct from original families."
        splits = load_split_stats(s)
        return ["Dataset", "Partition", "Count", "Share"], [
            [ds, split, n, fraction(n, sum(splits[i]))] for i, ds in enumerate(DATASETS) for split, n in zip(SPLITS, splits[i])], (3,), "Frozen manifests; CSV shares are fractions."
    if number == 3:
        rows = load_model_comparison(s)
        result = []
        for i, dataset_rows in enumerate(rows):
            keys = ("accuracy", "precision", "recall", "f1") if i == 0 else ("accuracy", "macro_precision", "macro_recall", "macro_f1")
            for row in dataset_rows:
                for key in keys:
                    result.append([DATASETS[i], "validation", row["model"], key,
                                   baseline_metric_label(DATASETS[i], key), row[key]])
        return ["Dataset", "Split", "Model", "Source metric", "Metric definition", "Value"], result, (5,), "Frozen development/validation comparison. USTC: Binary F1; CIC: Macro F1. No pooling, averaging, or cross-dataset model ranking using these different definitions. No TEST model selection. CSV values preserve original metric strings."
    if number in (4, 5, 6, 10, 11, 13, 14, 15):
        data = s.final()
        if number == 4:
            keys = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_precision", "weighted_recall", "weighted_f1")
            return ["Dataset", "Metric", "Value"], [[ds, key, data["metrics"][ds, key]["value"]] for ds in ("OVERALL",) + DATASETS for key in keys], (2,), "Selected frozen backbone on the 4,300-query historical TEST cohort, not full Phase-I TEST population; CSV metrics are fractions."
        if number == 5:
            return ["Dataset", "Class", "Precision", "Recall", "F1", "Support"], [[r[k] for k in ("source_dataset", "class_label", "precision", "recall", "f1", "support")] for r in data["classes"]], (2, 3, 4), "Historical TEST cohort; support from saved evaluation artifacts."
        if number == 6:
            rows = s.stage("stage10_final/test_evaluation_confusion_matrix.csv")
            return ["Dataset", "Actual", "Predicted", "Count"], [[r[k] for k in ("source_dataset", "actual_class", "predicted_class", "count")] for r in rows], (), "Raw saved confusion cells, no percentages."
        if number == 10:
            return ["Dataset", "Evidence status", "Count", "Share"], [[ds, status, count, fraction(count, sum(counts))] for ds, counts in zip(("OVERALL",) + DATASETS, data["evidence"]) for status, count in zip(STATUSES, counts)], (3,), CAPTIONS[2]
        if number == 11:
            pairs = list(zip(STATUSES, data["correctness"])) + [("Overall", tuple(map(sum, zip(*data["correctness"]))))]
            return ["Status", "Count", "Share", "Correct", "Incorrect", "Accuracy"], [[status, a+b, fraction(a+b, sum(data["evidence"][0])), a, b, fraction(a, a+b)] for status, (a, b) in pairs], (2, 5), "Evidence compatibility is distinct from classifier correctness."
        if number == 13:
            return ["Quantity", "Count", "Denominator", "Interpretation"], [[label, count, denominator if denominator is not None else "Count only", note] for label, count, denominator, note in data["provenance"]], (), CAPTIONS[4]
        old = [sum(row) for row in data["transition"]] if number == 14 else data["validation"]
        new = data["evidence"][0]
        return ["Status", "V4 count" if number == 14 else "Validation count", "V4 share" if number == 14 else "Validation share", "V5 count" if number == 14 else "TEST count", "V5 share" if number == 14 else "TEST share"], [[status, a, fraction(a, sum(old)), b, fraction(b, sum(new))] for status, a, b in zip(STATUSES, old, new)], (2, 4), CAPTIONS[3] if number == 14 else CAPTIONS[5]
    if number == 7:
        documents = load_stage2_corpus_stats(s)
        chunks, ref = load_stage3_index_stats(s)
        doc_counts = checked_counts(documents, ("source_type",), None)
        chunk_counts = checked_counts(chunks, ("source_type",), None)
        return ["Source category", "Documents", "Indexed chunks"], [[key[0], value, chunk_counts.get(key, 0)] for key, value in doc_counts.items()], (), "Frozen corpus and index; counts refer to distinct document/chunk identifiers, not query occurrences."
    if number == 8:
        retrieval, _ = load_stage4_retrieval_stats(s)
        verified = checked_counts(load_stage5_provenance_stats(s), ("verification_status", "authorization_status"))
        reranked = checked_counts(load_stage6_reranking_stats(s), ("source_type",), None)
        rows = [["Retrieval", key[0], n, "method candidate occurrences"] for key, n in retrieval.items()]
        rows += [["Verified", " / ".join(key), n, "verified candidate occurrences"] for key, n in verified.items()]
        rows += [["Reranked", key[0], n, "reranked output occurrences"] for key, n in reranked.items()]
        return ["Stage", "Category", "Count", "Unit"], rows, (), "Stage-specific occurrence counts; not query counts."
    if number == 9:
        chunks = load_stage7_context_stats(s)
        diversity = s.stage("stage07_trusted_contexts/trusted_context_document_diversity.csv")
        rows = [["Ready contexts", len({r["query_id"] for r in chunks}), "queries"], ["Context chunks", len(chunks), "occurrences"]]
        rows += [["Contexts with " + r["distinct_document_count"] + " documents", r["count"], "contexts"] for r in diversity]
        return ["Quantity", "Count", "Unit"], rows, (), "Context coverage differs from final selected provenance."
    if number == 12:
        rows = load_stage9_fusion_stats(s)
        return ["Decision source", "Fusion reason", "Count"], [[r[k] for k in ("decision_source", "fusion_reason", "count")] for r in rows], (), "Every classifier prediction retained; zero overrides."
    if number == 16:
        rows = []
        for ds, prefix in zip(DATASETS, ("ustc", "cic")):
            for r in s.rows(f"results/audits/{ds}/integrity_audit/{prefix}_integrity_audit_checklist.csv"):
                rows.append([ds, r.get("check_name", r.get("check")), r["status"]])
        require(all(r[1] and r[2] for r in rows), "audit check names/statuses")
        return ["Dataset", "Saved audit check", "Status"], rows, (), "Categorical statuses reproduced verbatim, not converted into numeric scores. Warnings are retained."
    raise ValueError("Unknown publication table")


COMPLETE_TABLE_NAMES = (
    "T1_Dataset_Summary", "T2_Dataset_Split_Summary", "T3_Five_Model_Comparison",
    "T4_Selected_Model_Final_Performance", "T5_Per_Class_TEST_Performance", "T6_Confusion_Matrix_Values",
    "T7_PhaseII_Corpus_Index_Summary", "T8_Retrieval_Provenance_Reranking_Summary",
    "T9_Trusted_Context_Summary", "T10_Evidence_Status_Distribution", "T11_Evidence_Status_Correctness",
    "T12_Fusion_Summary", "T13_Final_Provenance_Summary", "T14_V4_V5_Matched_Comparison",
    "T15_Validation_TEST_Comparison", "T16_Integrity_Audit_Checklist")


def write_complete_table(base, headers, rows, percentages, note):
    """Longtable permits multipage tables; CSV is always unformatted source data."""
    with base.with_suffix(".csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)
    width = .94 / len(headers)
    layout = "".join(f"p{{{width:.3f}\\linewidth}}" for _ in headers)
    head = " & ".join(map(latex_escape, headers)) + r" \\"
    lines = [r"% Requires \usepackage{booktabs,longtable}", r"\begingroup\small",
             r"\setlength{\tabcolsep}{2pt}", r"\begin{longtable}{" + layout + "}",
             r"\toprule", head, r"\midrule", r"\endfirsthead", r"\toprule", head,
             r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot"]
    for row in rows:
        display = [format_percent(v) if i in percentages else str(v) for i, v in enumerate(row)]
        lines.append(" & ".join(latex_escape(v).replace(r"\_", r"\_\allowbreak ") for v in display) + r" \\")
    lines += [r"\end{longtable}", r"\noindent\footnotesize " + latex_escape(note), r"\endgroup", ""]
    base.with_suffix(".tex").write_text("\n".join(lines), encoding="utf-8")


def safe_publication_directory(root, output):
    """Reject symlink/junction redirection, including preexisting leaf outputs."""
    expected = root / "final_paper_figures" / "complete_paper_figures"
    require(output == expected, "complete output root changed")
    paths = [root / "final_paper_figures", output] + [output / name for name in
             ("phase1", "phase2", "cross_phase", "final_test", "table_figures", "tables")]
    for path in paths:
        require(not path.is_symlink() and path.resolve() == path, f"redirected output directory: {path}")
    for path in paths:
        path.mkdir(exist_ok=True)


GENERATED_FIGURE_FOLDERS = ("phase1", "phase2", "cross_phase", "final_test", "table_figures")


def clear_generated_figure_folders(root, output):
    """Manual complete-mode only. Preflight every target before any deletion.

    Never follow links/junctions, and never clear tables or source directories.
    """
    import os
    import shutil
    import stat
    expected = root.resolve() / "final_paper_figures" / "complete_paper_figures"
    require(output == expected and output.resolve() == expected, "cleanup output root")
    targets = [output / name for name in GENERATED_FIGURE_FOLDERS]
    for path in [output.parent, output, *targets]:
        require(path.resolve() == path and not path.is_symlink(), "redirected cleanup path")
        require(not (getattr(path.lstat(), "st_file_attributes", 0) &
                     getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)), "cleanup reparse point")
    for target in targets:
        require(target.parent == expected and target.resolve().is_relative_to(expected), "cleanup escaped output")
        for current, directories, files in os.walk(target, followlinks=False):
            for name in directories + files:
                path = Path(current) / name
                require(path.resolve().is_relative_to(target) and not path.is_symlink(), "linked cleanup child")
                require(not (getattr(path.lstat(), "st_file_attributes", 0) &
                             getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)), "cleanup child reparse point")
    # All absolute targets and descendants are verified before recursive removal.
    for target in targets:
        shutil.rmtree(target)
        target.mkdir()


def verify_curated_source_relationships(s):
    """Check the source facts supporting conditional suppressions before cleanup."""
    final = s.final()
    queries = unique_rows(load_stage1_query_stats(s)["records"], ("query_id",), "curation queries")
    results = unique_rows(final["records"], ("query_id",), "curation final records")
    require(set(queries) == set(results) and all(
        queries[key]["source_dataset"] == row["source_dataset"] and
        queries[key]["classifier_prediction"] == row["final_class"] for key, row in results.items()),
        "Q01/Q35 populations or predictions differ")
    ranks = s.stage("stage04_retrieval/retrieval_rank_distribution.csv")
    require(ranks and all(int(row["count"]) == 4300 for row in ranks), "Q14 rank curation no longer valid")
    chunks = s.stage("stage07_trusted_contexts/trusted_context_chunk_distribution.csv")
    require(checked_counts(chunks, ("chunk_count",), total=4300) == {("5",): 4300}, "Q23 constant context size changed")
    decisions = checked_counts(s.stage("stage08_llm/test_llm_reasoning_decision_summary.csv"), ("reasoning_decision",), total=4300)
    statuses = dict(zip(STATUSES, s.final()["evidence"][0]))
    require(decisions == {("SUPPORT_CLASS",): statuses["SUPPORTED"]+statuses["CONTRADICTED"],
                          ("ABSTAIN",): statuses["INSUFFICIENT"]+statuses["CONFLICTING"]}, "Q26 decision curation changed")
    require(all(row["reasoning_decision"] == ("SUPPORT_CLASS" if row["evidence_status"] in
                ("SUPPORTED", "CONTRADICTED") else "ABSTAIN") for row in final["records"]),
            "Q26 per-query reasoning/status relationship changed")
    require(dict(load_stage8_llm_stats(s)) == {"Accepted": 4300, "Rejected": 0}, "Q27 response curation changed")


def checked_output_path(output, relative):
    path = output / relative
    require(path.resolve() == path and path.is_relative_to(output) and not path.is_symlink(), f"redirected output: {path}")
    return path


def prepare_complete_paper_figures():
    """Manual-only renderer. A failed candidate is reported, never substituted."""
    from matplotlib import pyplot as plt
    root = Path(__file__).resolve().parent
    output = root / "final_paper_figures/complete_paper_figures"
    sources = PublicationSources(root)
    candidates = complete_figure_candidates()
    registry_audit = audit_complete_registry(candidates)
    verify_curated_source_relationships(sources)
    safe_publication_directory(root, output)
    clear_generated_figure_folders(root, output)
    manifest, table_report = [], []
    for candidate in candidates:
        sources.used.clear()
        scope = figure_result_scope(candidate)
        record = {"figure_id": candidate.figure_id, "filename": candidate.name,
                  "phase": candidate.folder, "stage": candidate.stage, "source_artifact": "",
                  "result_scope": scope,
                  "main_or_supplement": candidate.recommendation, "scientific_message": candidate.message,
                  "canonical_figure_id": candidate.duplicate_of or candidate.figure_id,
                  "canonical_id": candidate.canonical_id, "source_signature": candidate.source_signature,
                  "visual_signature": candidate.visual_signature, "output_path": candidate.output_path,
                  "render": candidate.render, "suppression_reason": candidate.suppression_reason,
                  "external_scientific_notes": "",
                  "duplicate_of": candidate.duplicate_of, "status": "ASSERTION_FAILED",
                  "generated": False, "skip_reason": ""}
        if not candidate.render:
            record.update(status="REDUNDANT_SUPPRESSED", skip_reason=candidate.suppression_reason + " Use " + candidate.canonical_id)
            manifest.append(record)
            continue
        base = None
        try:
            # Source and numeric assertions run in the adapter before export.
            with rc_context(PUBLICATION_STYLE):
                if scope in ("FINAL_TEST", "FINAL_TEST_EVALUATION"):
                    sources.final()
                fig = candidate.build(sources)
                record["external_scientific_notes"] = " | ".join(prepare_visible_text(fig, candidate))
                verify_final_source_scope(sources, scope)
                base = checked_output_path(output, Path(candidate.folder) / candidate.name)
                for extension in ("pdf", "png", "svg"):
                    checked_output_path(output, base.relative_to(output).with_suffix("." + extension))
                export_figure(fig, base)
            record["generated"] = True
            record["status"] = "GENERATED"
        except (SourceNotFound, FileNotFoundError) as exc:
            record["skip_reason"] = "SOURCE_NOT_FOUND: " + str(exc)
            record["status"] = "SOURCE_NOT_FOUND"
        except Exception as exc:
            record["skip_reason"] = f"VALIDATION_OR_EXPORT_FAILED: {type(exc).__name__}: {exc}"
        finally:
            plt.close("all")
            if not record["generated"] and base is not None:
                for extension in ("pdf", "png", "svg"):
                    failed_path = checked_output_path(output, base.relative_to(output).with_suffix("." + extension))
                    if failed_path.exists():
                        failed_path.unlink()
            record["source_artifact"] = " | ".join(sorted(sources.used))
            manifest.append(record)
        print(f"{'GENERATED' if record['generated'] else 'SKIPPED'} {candidate.name} {record['skip_reason']}")
    indexed_manifest = {key: row for row in manifest for key in (row["figure_id"], row["filename"])}
    for row in manifest:
        if row["duplicate_of"]:
            canonical = indexed_manifest[row["duplicate_of"]]
            row["source_artifact"] = canonical["source_artifact"]
            row["skip_reason"] += "; canonical status: " + canonical["status"]
    for number in ACTIVE_TABLE_NUMBERS:
        name = COMPLETE_TABLE_NAMES[number-1]
        sources.used.clear()
        scope = table_result_scope(number)
        record = {"figure_id": name.split("_")[0], "filename": name, "phase": "tables",
                  "stage": f"table {number}", "result_scope": scope, "main_or_supplement": "SUPPLEMENT",
                  "scientific_message": "", "generated": False, "skip_reason": ""}
        try:
            if scope == "FINAL_TEST":
                sources.final()
            headers, rows, percentages, note = complete_table_data(sources, number)
            verify_final_source_scope(sources, scope)
            record["scientific_message"] = note
            require(bool(rows) and all(len(row) == len(headers) for row in rows), "table shape")
            # Validate all percentage display cells before either table is written.
            for row in rows:
                for index in percentages:
                    finite_number(row[index], True)
                    format_percent(row[index])
            for ext in ("csv", "tex"):
                checked_output_path(output, Path("tables") / (name + "." + ext))
            write_complete_table(output / "tables" / name, headers, rows, percentages, note)
            record["generated"] = True
        except (SourceNotFound, FileNotFoundError) as exc:
            record["skip_reason"] = "SOURCE_NOT_FOUND: " + str(exc)
        except Exception as exc:
            record["skip_reason"] = f"VALIDATION_OR_EXPORT_FAILED: {type(exc).__name__}: {exc}"
        record["source_artifact"] = " | ".join(sorted(sources.used))
        table_report.append(record)
        print(f"{'GENERATED' if record['generated'] else 'SKIPPED'} {name} {record['skip_reason']}")
    manifest_path = checked_output_path(output, "Complete_Figure_Manifest.csv")
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    report = {"mode": "complete-paper-figures", "figures": manifest, "tables": table_report,
              "canonical_registry_audit": registry_audit,
              "registry_counts": {"requested_definitions": len(candidates),
                                  "duplicate_aliases": sum(bool(c.duplicate_of) for c in candidates),
                                  "unique_definitions": sum(not c.duplicate_of for c in candidates)},
              "cleared_figure_folders": list(GENERATED_FIGURE_FOLDERS),
              "methodology_caption": METHODOLOGY_CAPTION,
              "first_recommended_methodology": "M01_Complete_Methodology_Pipeline",
              "visible_word_policy": {"Frozen": 0, "Cohort": 0, "Historical": 0,
                                      "enforcement": "all Text artists checked before every export"},
              "physical_figure_basenames": [r["phase"] + "/" + r["filename"] for r in manifest if r["generated"]],
              "V4/V5 comparison figures excluded": "YES",
              "validation-vs-TEST result figures excluded": "YES",
              "suppressed_candidates": ["R4_V4_V5_Matched_Evidence_Comparison", "S1_Validation_vs_TEST_Evidence_Status",
                                        "X03_V4_V5_Matched_Comparison", "X04_V4_V5_Transition_Matrix",
                                        "X05_Validation_TEST_Evidence_Status", "S2_TEST_Class_Evidence_Matrix",
                                        "T14_V4_V5_Matched_Comparison", "T15_Validation_TEST_Comparison"],
              "sources": [{"source": path, "sha256": digest} for path, digest in sorted(sources.hashes.items())],
              "generated_figures": sum(r["generated"] for r in manifest if r["phase"] != "table_figures"),
              "generated_table_figures": sum(r["generated"] for r in manifest if r["phase"] == "table_figures"),
              "generated_tables": sum(r["generated"] for r in table_report),
              "notes": ["No inference, training, model loading, raw label inspection, or pipeline execution.",
                        "USTC validation F1/precision/recall are binary, CIC metrics are macro.",
                        "Final TEST ROC is calculated deterministically from stage10_final/test_evaluation_records.csv and exact-ID/probability matched to saved final records; tied scores grouped; no tuning or inference.",
                        "Validation ROC excluded. Development/validation model selection and final TEST evaluation are separate figures.",
                        "Final TEST panels use the fixed historical 4,300-query cohort, not full Phase-I population.",
                        "Source-not-found and invalid candidates are not silently substituted.",
                        "generated=true requires all three exports from this run; failed exports are removed.",
                        "New numerical figures use solid fills, black text, no grid or internal heatmap borders.",
                        METHODOLOGY_CAPTION,
                        "Final-result adapters never read V4 or validation-result records. Frozen corpus metadata is permitted only as referenced shared evidence.",
                        "Only the five approved generated figure folders are cleared before rendering; source artifacts and tables are untouched by cleanup.",
                        "LaTeX requires booktabs and longtable; CSV source metric strings retain precision."]}
    checked_output_path(output, "Preparation_Report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    essential = {"M01_Complete_Methodology_Pipeline", "P09_Five_Model_Baseline_Comparison", "P12_PhaseI_Combined_ROC"}
    essential_ok = all(r["generated"] for r in manifest if r["filename"] in essential)
    failed = any(r["skip_reason"].startswith("VALIDATION_OR_EXPORT_FAILED") for r in manifest + table_report)
    print(f"Generated {report['generated_figures']} figures and {report['generated_table_figures']} table figures (three formats each), {report['generated_tables']} tables (CSV + LaTeX).")
    print(f"Manifest: {manifest_path}")
    return 0 if essential_ok and not failed else 1


# The project pack is intentionally independent of the older complete registry:
# its P identifiers refer to full held-out TEST, never to validation or 4,300 IDs.
PROJECT_PHASE1_TEST_FILES = (
    "results/final/USTC-TFC2016/ustc_test_results.csv",
    "results/final/CIC-Darknet2020/cic_test_results.csv")
PROJECT_CLOSURES = tuple(f"metadata/traffic/{p}_experiment_closure.csv" for p in ("ustc", "cic"))
PROJECT_CONFUSIONS = (
    "results/models/USTC-TFC2016/20260919T183945Z/confusion_matrix.csv",
    "results/models/CIC-Darknet2020/20260920T145831Z/confusion_matrix.csv")
PROJECT_EXCLUDED_CURVES = tuple((Path(path).parent / filename).as_posix()
                                for path in PROJECT_CONFUSIONS for filename in ("roc_curve.csv", "pr_curve.csv"))
PROJECT_AUDITS = tuple(f"results/audits/{ds}/integrity_audit/{prefix}_integrity_audit_checklist.csv"
                       for ds, prefix in zip(DATASETS, ("ustc", "cic")))
PROJECT_DUPLICATE_AUDIT = "results/audits/USTC-TFC2016/integrity_audit/ustc_cross_split_duplicate_check.csv"
PROJECT_DATASET_FILES = PROJECT_CLOSURES + (
    USTC_DATA + "dataset_summary.csv", USTC_DATA + "class_distribution.csv",
    "metadata/traffic/cic_phase1_contract.csv")
PROJECT_SPLIT_FILES = PROJECT_DATASET_FILES + (USTC_PRE + "split_distribution.csv", "metadata/traffic/cic_split_manifest.csv")
PROJECT_MODEL_NAMES = {"random_forest": "Random Forest", "1d_cnn": "1D-CNN", "xgboost": "XGBoost",
                       "lightgbm": "LightGBM", "cnn_lstm": "CNN-LSTM"}
PROJECT_SCORE_ABSENCE = (
    "No authoritative per-sample full held-out TEST probability vectors with sample IDs were found. "
    "The authoritative run directories contain saved TEST ROC/PR coordinates, but those files have "
    "no sample IDs or aligned probability vectors and cannot satisfy the requested per-sample join audit. "
    "Processed TEST data contain predictors/labels, not saved predictions. Phase-II scores cover a "
    "different 4,300-query population and are excluded. No curve reconstruction or inference is permitted.")


@dataclass
class ProjectFigure:
    name: str
    title: str
    category: str
    result_scope: str
    source_files: tuple
    source_fields: str
    calculation: str
    build: object
    stage: int = 0
    render: bool = True
    duplicate_of: str = ""
    suppression_reason: str = ""

    @property
    def figure_id(self):
        return self.name.split("_")[0]


class ProjectFigureSources(PublicationSources):
    """Per-figure allowlist and row-scope guard; no historical-result fallback."""
    def __init__(self, root, candidate):
        super().__init__(root)
        self.candidate = candidate
        self.details = {}

    def touch(self, relative):
        path = Path(relative).as_posix()
        require(path in self.candidate.source_files, "undeclared project figure source: " + path)
        if self.candidate.result_scope == "PHASE2_FINAL_4300":
            require(path.startswith(TEST_NAMESPACE.as_posix()+"/"), "Phase-II source outside final namespace")
        if self.candidate.result_scope == "PHASE1_FULL_TEST":
            require("validation" not in path.lower() and "train_results" not in path.lower(),
                    "non-TEST performance file")
            require(not path.startswith("results/rag/"), "Phase-II population in Phase-I figure")
        return super().touch(path)

    def rows(self, relative):
        rows = super().rows(relative)
        # Authoritative confusion CSVs contain multiple splits. Only TEST rows
        # leave this adapter; validation coordinates/metrics are never loaded.
        if self.candidate.result_scope == "PHASE1_FULL_TEST" and rows and "split" in rows[0]:
            rows = [row for row in rows if row["split"].lower() == "test"]
            require(bool(rows), "no TEST rows in declared performance source")
            self.details.setdefault("row_filters", {})[Path(relative).as_posix()] = "split == test"
        return rows

    def final(self):
        require(self.candidate.result_scope == "PHASE2_FINAL_4300", "final population scope")
        if not hasattr(self, "_project_final"):
            # Audit the complete, explicit read set before the read-only loader.
            for path in PROJECT_FINAL_FILES:
                self.touch(path)
            self._project_final = load_final_test_data(self.root)
            actual = {Path(item["source"]).relative_to(self.root).as_posix() for item in self._project_final["audit"]}
            require(actual == set(PROJECT_FINAL_FILES), "final loader read-set drift")
        return self._project_final


def project_phase1_metrics(s):
    result = []
    for i, (path, closure_path) in enumerate(zip(PROJECT_PHASE1_TEST_FILES, PROJECT_CLOSURES)):
        closure = s.rows(closure_path)[0]
        require(closure["test_results_path"] == path and closure["dataset"] == DATASETS[i], "registered TEST result path")
        rows = s.rows(path)
        keyed = unique_rows(rows, ("model",), "five-model TEST metrics")
        require(set(keyed) == {(key,) for key in PROJECT_MODEL_NAMES}, "five TEST model rows required")
        for row in rows:
            require(row["dataset"] == DATASETS[i] and row["authoritative_run"] == closure["authoritative_run"], "TEST run identity")
            require(int(row["total_rows" if i == 0 else "row_count"]) == int(closure["test_rows"]), "full held-out TEST support")
            keys = ("accuracy", "precision", "recall", "f1") if i == 0 else ("accuracy", "macro_precision", "macro_recall", "macro_f1")
            for key in keys:
                finite_number(row[key], True)
        result.append([keyed[key,] for key in PROJECT_MODEL_NAMES])
    s.details["sample_count"] = {ds: int(rows[0]["total_rows" if i == 0 else "row_count"])
                                  for i, (ds, rows) in enumerate(zip(DATASETS, result))}
    s.details["model_policy"] = "LightGBM is retained a priori; five models are displayed in fixed order, not TEST-ranked."
    return result


def project_table_panel(ax, rows, widths=(.62, .38)):
    import textwrap
    ax.set_axis_off()
    cells = [[textwrap.fill(str(value), 30 if j == 0 else 25, break_long_words=False)
              for j, value in enumerate(row)] for row in rows]
    table = ax.table(cellText=cells, colWidths=widths, cellLoc="left", bbox=(0, 0, 1, .91))
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    for cell in table.get_celld().values():
        cell.visible_edges = ""
        cell.set_facecolor("white")
        cell.get_text().set_color("black")
    return table


def plot_project_dataset_audit(s):
    from matplotlib import pyplot as plt
    closures, counts, summary, contract = load_phase1_dataset_stats(s)
    families = s.rows(USTC_PRE+"split_family_assignment.csv")
    unique_rows(families, ("semantic_family",), "family-disjoint assignment")
    require(len(families) == int(summary["semantic_family_count"]), "family count")
    require(contract["split_group"] == "Flow ID", "CIC group contract")
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.8), layout="constrained")
    for i, ax in enumerate(axes):
        statuses = Counter(row["status"] for row in s.rows(PROJECT_AUDITS[i]))
        require(set(statuses) <= {"PASS", "WARN", "FAIL"}, "audit status vocabulary")
        rows = [("Usable records / flows", f"{sum(counts[i].values()):,}"),
                ("Target classes", len(counts[i])), ("Retained predictors", closures[i]["predictor_count"]),
                ("Split structure", "Family-disjoint" if i == 0 else "Flow-ID group-aware"),
                ("Closure audit status", closures[i]["integrity_audit_status"]),
                ("Checklist statuses", "; ".join(f"{k}: {v}" for k, v in sorted(statuses.items())))]
        if i == 0:
            rows.insert(2, ("Original families", summary["semantic_family_count"]))
        project_table_panel(ax, rows)
        ax.set_title(DATASETS[i], fontsize=9)
    return fig


def plot_project_dataset_size(s):
    _, counts, _, _ = load_phase1_dataset_stats(s)
    return count_figure({ds: sum(values.values()) for ds, values in zip(DATASETS, counts)},
                        "Dataset Size", "Records / flows", ["#0072B2", "#D55E00"], horizontal=False)


def plot_project_performance(s, five_models=False):
    from matplotlib import pyplot as plt
    all_rows = project_phase1_metrics(s)
    if five_models:
        fig = plt.figure(figsize=(7.16, 4.9))
        fig._publication_layout_locked = True
        grid = fig.add_gridspec(2, 2, left=.14, right=.97, top=.84, bottom=.05,
                               height_ratios=(3.5, .5), hspace=.55, wspace=.75)
        for i, rows in enumerate(all_rows):
            ax = fig.add_subplot(grid[0, i])
            for j, key in enumerate(("accuracy", "f1" if i == 0 else "macro_f1")):
                positions = [index+(j-.5)*.36 for index in range(5)]
                bars = ax.barh(positions, [float(row[key])*100 for row in rows], height=.32,
                               color=("#0072B2", "#009E73")[j], edgecolor="none",
                               label=baseline_metric_label(DATASETS[i], key))
                annotate_bars(ax, bars, [format_percent(row[key]) for row in rows], horizontal=True)
            ax.set_yticks(range(5), [PROJECT_MODEL_NAMES[row["model"]] for row in rows])
            ax.invert_yaxis()
            ax.set_xlim(0, 121)
            ax.set_xticks((0, 25, 50, 75, 100))
            percentage_axis(ax)
            ax.set_xlabel("Performance (%)", labelpad=8)
            ax.set_title(f"({chr(97+i)}) {DATASETS[i]}", fontsize=9)
            clean_axes(ax)
            _place_bottom_legend(fig.add_subplot(grid[1, i]), ax, ncol=1)
        return fig
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.7), layout="constrained")
    for i, ax in enumerate(axes):
        row = next(row for row in all_rows[i] if row["model"] == "lightgbm")
        keys = ("accuracy", "precision", "recall", "f1") if i == 0 else ("accuracy", "macro_precision", "macro_recall", "macro_f1")
        bars = ax.barh(range(4), [float(row[key])*100 for key in keys], color="#0072B2", edgecolor="none")
        annotate_bars(ax, bars, [format_percent(row[key]) for key in keys], horizontal=True)
        ax.set_yticks(range(4), [baseline_metric_label(DATASETS[i], key) for key in keys])
        ax.invert_yaxis()
        ax.set_xlim(0, 120)
        ax.set_xticks((0, 25, 50, 75, 100))
        percentage_axis(ax)
        ax.set_xlabel("Performance (%)")
        ax.set_title(f"({chr(97+i)}) {DATASETS[i]}", fontsize=9)
        clean_axes(ax)
    return fig


def project_phase1_confusions(s):
    metrics = project_phase1_metrics(s)
    mapping = s.rows("metadata/traffic/cic_label_mapping.csv")
    require([(int(r["class_id"]), r["canonical_label"]) for r in mapping] ==
            list(enumerate(("Non-Tor", "NonVPN", "Tor", "VPN"))), "CIC class order")
    matrices = []
    for i, path in enumerate(PROJECT_CONFUSIONS):
        labels = [label for ds, label in CLASS_KEYS if ds == DATASETS[i]]
        rows = [row for row in s.rows(path) if row["model"] == "lightgbm"]
        actual, predicted = (("actual_class", "predicted_class") if i == 0 else
                             ("actual_class_label", "predicted_class_label"))
        keyed = unique_rows(rows, (actual, predicted), "full TEST confusion cells")
        require(set(keyed) == {(a, p) for a in labels for p in labels}, "full TEST confusion class coverage")
        metric = next(r for r in metrics[i] if r["model"] == "lightgbm")
        require(all(r["dataset"] == DATASETS[i] and r["run_id"] == metric["authoritative_run"] and
                    r["split"] == "test" for r in rows), "TEST confusion run/split")
        if i == 1:
            require(all(labels[int(r["actual_class_id"])] == r[actual] and
                        labels[int(r["predicted_class_id"])] == r[predicted] for r in rows), "confusion class IDs")
        matrix = [[int(keyed[a, p]["count"]) for p in labels] for a in labels]
        n = sum(map(sum, matrix))
        require(all(v >= 0 for row in matrix for v in row) and n == s.details["sample_count"][DATASETS[i]], "TEST confusion support")
        require(abs(fraction(sum(matrix[j][j] for j in range(len(labels))), n)-Decimal(metric["accuracy"])) < Decimal("1e-12"), "TEST confusion accuracy")
        if i == 0:
            require([v for row in matrix for v in row] == [int(metric[k]) for k in
                    ("true_negative", "false_positive", "false_negative", "true_positive")], "USTC saved confusion agreement")
        matrices.append(matrix)
    s.details.update(label_source=list(PROJECT_CONFUSIONS), score_source=[],
                     class_order={ds: [c for d, c in CLASS_KEYS if d == ds] for ds in DATASETS},
                     id_join_rule="No per-sample join: unique saved dataset/run/model/test/actual/predicted cells; support and accuracy cross-checked.",
                     calculation_method="Saved LightGBM TEST cell counts; no inference or reconstructed predictions.")
    return plot_test_confusions({"matrices": matrices})


def project_missing_test_scores(s):
    # Deliberate SOURCE_NOT_FOUND: stored curves have no sample identities.
    # No guessed schema, validation fallback, threshold inversion or hard-label scores.
    closures = [s.rows(path)[0] for path in PROJECT_CLOSURES]
    s.details.update(label_source=[], score_source=[],
                     class_order={ds: [c for d, c in CLASS_KEYS if d == ds] for ds in DATASETS},
                     sample_count={ds: int(row["test_rows"]) for ds, row in zip(DATASETS, closures)},
                     id_join_rule="Unavailable: no saved per-sample full TEST score records with IDs.",
                     calculation_method="Not calculated; stored coordinate files do not meet the requested per-sample source contract.",
                     source_issue=PROJECT_SCORE_ABSENCE,
                     examined_but_not_eligible=[path for path in PROJECT_EXCLUDED_CURVES if
                         path.endswith("roc_curve.csv" if s.candidate.figure_id == "P07" else "pr_curve.csv")])
    raise SourceNotFound("TEST_SCORE_SOURCE_NOT_FOUND")


def plot_project_leakage(s):
    from matplotlib import pyplot as plt
    raw = [s.rows(path) for path in PROJECT_AUDITS]
    duplicates = s.rows(PROJECT_DUPLICATE_AUDIT)
    groups = (
        [("Family separation", [r for r in raw[0] if r["category"] == "B"]),
         ("Capture / provenance separation", [r for r in raw[0] if r["category"] == "C"]),
         ("Numeric duplicate checks", [r for r in raw[0] if r["category"] == "D"]),
         ("Preprocessing fit policy", [r for r in raw[0] if r["category"] == "F"]),
         ("TEST isolation evidence", [r for r in raw[0] if r["category"] == "G"])],
        [("Flow-ID grouping", [r for r in raw[1] if r["check"] == "flow_id_grouping_contract"]),
         ("Split integrity", [r for r in raw[1] if r["check"] == "split_row_counts"]),
         ("Preprocessing fit policy", [r for r in raw[1] if r["check"] == "preprocessing_leakage_status"]),
         ("TEST isolation / selection", [r for r in raw[1] if r["section"] == "configuration"])])
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 4.5), layout="constrained")
    for i, ax in enumerate(axes):
        cells = []
        for label, rows in groups[i]:
            require(rows, "missing audit group: " + label)
            counts = Counter(r["status"] for r in rows)
            require(set(counts) <= {"PASS", "WARN", "FAIL"}, "audit status vocabulary")
            cells.append((label, "; ".join(f"{key}: {value}" for key, value in sorted(counts.items()))))
        if i == 0:
            for row in duplicates:
                if row["comparison"] == "predictors":
                    cells.append((f"Shared numeric rows: {row['left_split']} / {row['right_split']}",
                                  f"{int(row['shared_distinct_rows']):,} ({row['status']})"))
        project_table_panel(ax, cells)
        ax.set_title(DATASETS[i], fontsize=9)
    s.details["interpretation"] = "Authentic selected audit status counts. Numeric duplicates and audit WARN rows do not by themselves establish leakage; no zero-leakage claim."
    return fig


def project_stage_paths(*paths):
    return tuple((TEST_NAMESPACE / path).as_posix() for path in paths)


def plot_project_corpus(s):
    rows = load_stage2_corpus_stats(s)
    counts = checked_counts(rows, ("source_type",), None)
    return count_figure({SOURCE_DISPLAY_NAMES.get(k[0], k[0]): n for k, n in counts.items()},
                        "Evidence Corpus", "Documents", SOURCE_PALETTE[:len(counts)])


def plot_project_index(s):
    rows, ref = load_stage3_index_stats(s)
    fig = plot_index_stats(s, "tokens")
    fig.axes[0].texts[0].set_visible(False)
    fig.axes[0].set_xlabel(f"Tokens per chunk\n{int(ref['document_count']):,} documents; {len(rows):,} chunks; token limit {ref['token_limit']}")
    return fig


def plot_project_retrieval(s):
    counts, _ = load_stage4_retrieval_stats(s)
    names = {"dense_bge": "Dense BGE", "hybrid_rrf": "Hybrid RRF", "tfidf": "TF-IDF"}
    require({k[0] for k in counts} == set(names), "retrieval methods")
    return count_figure({names[key[0]]: value for key, value in counts.items()}, "RAG Retrieval", "Candidate occurrences", horizontal=False)


def plot_project_provenance(s):
    load_stage5_provenance_stats(s)
    rows = s.stage("stage05_provenance/provenance_source_type_summary.csv")
    require(all(r["verification_status"] == "VERIFIED" and r["authorization_status"] == "YES" for r in rows), "authorized source statuses")
    counts = checked_counts(rows, ("source_type",), total=129000)
    return count_figure({SOURCE_DISPLAY_NAMES.get(k[0], k[0]): n for k, n in counts.items()},
                        "Provenance Verification", "Verified / authorized candidate occurrences")


def plot_project_context(s):
    from matplotlib import pyplot as plt
    chunks = load_stage7_context_stats(s)
    ready = checked_counts(s.stage("stage07_trusted_contexts/trusted_context_summary.csv"), ("context_status",), total=4300)
    sizes = checked_counts(s.stage("stage07_trusted_contexts/trusted_context_chunk_distribution.csv"), ("chunk_count",), total=4300)
    require(ready == {("READY",): 4300} and sizes == {("5",): 4300}, "context readiness / size")
    rows = s.stage("stage07_trusted_contexts/trusted_context_document_diversity.csv")
    diversity = checked_counts(rows, ("distinct_document_count",), total=4300)
    documents = {}
    for row in chunks:
        documents.setdefault(row["query_id"], set()).add(row["document_id"])
    require(Counter({int(k[0]): v for k, v in diversity.items()}) == Counter(map(len, documents.values())), "document diversity linkage")
    require(diversity == {("5",): 4100, ("4",): 151, ("3",): 46, ("2",): 3}, "document diversity counts")
    sources = checked_counts(chunks, ("source_type",), None, 21500)
    require(sources == {("FROZEN_TRAIN_EMPIRICAL_EVIDENCE",): 20851, ("peer_reviewed_research",): 637,
                        ("dataset_documentation",): 12}, "context source counts")
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.6), layout="constrained")
    draw_counts(axes[0], [(k[0]+" documents", n) for k, n in sorted(diversity.items())], "Contexts")
    draw_counts(axes[1], [(SOURCE_DISPLAY_NAMES.get(k[0], k[0]), n) for k, n in sources.items()], "Chunk occurrences")
    axes[0].set_title("(a) Document diversity", fontsize=9)
    axes[1].set_title("(b) Context sources", fontsize=9)
    s.details["context_summary"] = {"ready_contexts": sum(ready.values()), "chunks_per_context": 5, "chunk_occurrences": len(chunks)}
    return fig


def plot_project_reasoning(s):
    validity = dict(load_stage8_llm_stats(s))
    counts = checked_counts(s.stage("stage08_llm/test_llm_evidence_status_summary.csv"), ("evidence_status",), total=4300)
    final = checked_counts(s.stage("stage10_final/test_final_evidence_status_distribution.csv"), ("evidence_status",), total=4300)
    require(counts == final and [counts[status,] for status in STATUSES] == [2216, 561, 875, 648], "published evidence status counts")
    s.details.update(response_validity=validity, interpretation="Evidence Status is not Classifier Correctness.")
    return count_figure({status: counts[status,] for status in STATUSES}, "Constrained LLM Reasoning", "Queries", STATUS_COLORS)


def plot_project_fusion(s):
    counts = checked_counts(load_stage9_fusion_stats(s), ("decision_source",), total=4300)
    fusion = unique_rows(s.stage("stage09_fusion/test_fusion_records.csv"), ("query_id",), "fusion IDs")
    final = unique_rows(s.stage("stage10_final/test_final_classification_records.csv"), ("query_id",), "final IDs")
    require(set(fusion) == set(final) and len(final) == 4300, "fusion/final population")
    require(all(r["decision_source"] == "retain_classifier" and r["final_class"] == r["classifier_prediction"] ==
                final[key]["final_class"] for key, r in fusion.items()), "zero overrides")
    status = checked_counts(s.stage("stage09_fusion/test_fusion_by_evidence_status.csv"), ("evidence_status", "decision_source"), total=4300)
    require(status == checked_counts(list(fusion.values()), ("evidence_status", "decision_source"), None, 4300), "fusion status record consistency")
    fig, ax = new_axes("Decision Fusion", (3.5, 1.8))
    total = sum(counts.values())
    ax.barh([0], [total], height=.28, color="#0072B2", edgecolor="none")
    ax.set_yticks([0], ["Retained\nclassifier"])
    ax.set_xlim(0, total*1.05)
    ax.set_ylim(-.6, .6)
    ax.set_xlabel(f"{total:,} / {total:,} retained ({format_percent(fraction(total, total))})\nOverrides: 0")
    ax.set_xticks([])
    return fig


def plot_project_final(s):
    final = unique_rows(s.stage("stage10_final/test_final_classification_records.csv"), ("query_id",), "final IDs")
    evaluated = unique_rows(s.stage("stage10_final/test_evaluation_records.csv"), ("query_id",), "evaluation IDs")
    require(set(final) == set(evaluated) and len(final) == 4300, "final evaluation exact IDs")
    require(all(r["final_class"] == final[key]["final_class"] and r["source_dataset"] == final[key]["source_dataset"]
                for key, r in evaluated.items()), "final evaluation identity")
    counts = checked_counts(s.stage("stage10_final/test_evaluation_confusion_matrix.csv"),
                            ("source_dataset", "actual_class", "predicted_class"), total=4300)
    reconstructed = Counter((r["source_dataset"], r["true_class_label"], r["final_class"]) for r in evaluated.values())
    require(counts == reconstructed, "saved confusion vs evaluation labels")
    matrices = []
    for ds, support in zip(DATASETS, (1700, 2600)):
        labels = [c for d, c in CLASS_KEYS if d == ds]
        matrix = [[counts[ds, a, p] for p in labels] for a in labels]
        require(sum(map(sum, matrix)) == support, "Phase-II dataset support")
        matrices.append(matrix)
    require(matrices == [[[850, 1], [0, 849]], [[848, 2, 0, 2], [0, 800, 2, 36], [0, 4, 48, 1], [2, 44, 0, 811]]], "final confusion counts")
    s.details.update(label_source=project_stage_paths("stage10_final/test_evaluation_records.csv"),
                     score_source=[], sample_count={DATASETS[0]: 1700, DATASETS[1]: 2600},
                     class_order={ds: [c for d, c in CLASS_KEYS if d == ds] for ds in DATASETS},
                     id_join_rule="Exact unique query_id sets; dataset and final class equality; 4,300 rows.",
                     calculation_method="Saved raw confusion counts cross-checked against completed evaluation labels and saved final classes.")
    return plot_test_confusions({"matrices": matrices})


PROJECT_FINAL_FILES = project_stage_paths(
    "stage10_final/test_evaluation_metrics.csv", "stage10_final/test_evaluation_per_class_metrics.csv",
    "stage10_final/test_evaluation_confusion_matrix.csv", "stage10_final/test_final_evidence_status_distribution.csv",
    "stage08_llm/test_llm_by_dataset.csv", "stage10_final/test_evaluation_correctness_by_evidence_status.csv",
    "stage10_final/test_final_classification_records.csv", "stage09_fusion/test_fusion_records.csv",
    "stage08_llm/test_llm_rejected_responses.jsonl", "stage10_final/test_final_class_evidence_matrix.csv",
    "stage10_final/test_final_provenance_summary.csv", "stage07_trusted_contexts/trusted_context_chunks.csv",
    "stage07_trusted_contexts/trusted_context_document_diversity.csv", "stage07_trusted_contexts/trusted_context_summary.csv",
    "stage08_llm/test_llm_responses.jsonl", "stage04_retrieval/retrieval_method_summary.csv",
    "plotting/stage05_provenance_status_counts.csv")
PROJECT_STAGE_FILES = {
    1: project_stage_paths("stage01_structured_queries/test_query_records.csv", "stage01_structured_queries/test_query_class_distribution.csv",
                          "stage01_structured_queries/test_query_confidence_histogram.csv", "stage01_structured_queries/test_query_probability_summary.csv"),
    2: project_stage_paths("stage02_corpus/corpus_document_summary.csv", "stage02_corpus/corpus_reference_summary.csv",
                          "stage02_corpus/corpus_curation_summary.csv"),
    3: project_stage_paths("stage03_index/index_chunk_summary.csv", "stage03_index/index_reference_summary.csv",
                          "stage03_index/index_document_chunk_counts.csv", "stage03_index/index_token_distribution.csv"),
    4: project_stage_paths("stage04_retrieval/retrieval_method_summary.csv", "stage04_retrieval/retrieval_source_distribution.csv",
                          "stage04_retrieval/retrieval_rank_distribution.csv", "stage04_retrieval/retrieval_query_summary.csv"),
    5: project_stage_paths("stage05_provenance/provenance_status_summary.csv", "stage05_provenance/provenance_source_type_summary.csv"),
    6: project_stage_paths("stage06_reranking/reranked_candidates.csv"),
    7: project_stage_paths("stage07_trusted_contexts/trusted_context_chunks.csv", "stage07_trusted_contexts/trusted_context_summary.csv",
                          "stage07_trusted_contexts/trusted_context_document_diversity.csv", "stage07_trusted_contexts/trusted_context_chunk_distribution.csv"),
    8: project_stage_paths("stage08_llm/test_llm_execution_summary.csv", "stage08_llm/test_llm_rejected_responses.jsonl",
                          "stage08_llm/test_llm_evidence_status_summary.csv", "stage08_llm/test_llm_reasoning_decision_summary.csv") + PROJECT_FINAL_FILES,
    9: project_stage_paths("stage09_fusion/test_fusion_summary.csv", "stage09_fusion/test_fusion_by_evidence_status.csv",
                          "stage09_fusion/test_fusion_records.csv", "stage10_final/test_final_classification_records.csv"),
    10: PROJECT_FINAL_FILES + project_stage_paths("stage10_final/test_evaluation_records.csv")}


def project_missing_source(s):
    s.details["source_issue"] = (PROJECT_SCORE_ABSENCE if s.candidate.figure_id in ("P11", "P12", "P13") else
                                 "No Stage-2 claim-level counts in the final namespace; external references are not followed.")
    raise SourceNotFound(s.details["source_issue"])


def project_matrix_values(s):
    """Validate saved full-split matrices without rendering or inference."""
    metrics = project_phase1_metrics(s)
    matrices = []
    for i, path in enumerate(PROJECT_CONFUSIONS):
        labels = [c for ds, c in CLASS_KEYS if ds == DATASETS[i]]
        rows = [r for r in s.rows(path) if r["model"] == "lightgbm"]
        keys = ("actual_class", "predicted_class") if i == 0 else ("actual_class_label", "predicted_class_label")
        cells = unique_rows(rows, keys, "full held-out cells")
        require(set(cells) == {(a, p) for a in labels for p in labels}, "confusion class coverage")
        metric = next(r for r in metrics[i] if r["model"] == "lightgbm")
        require(all(r["split"] == "test" and r["run_id"] == metric["authoritative_run"] for r in rows), "TEST run")
        matrix = [[int(cells[a, p]["count"]) for p in labels] for a in labels]
        require(sum(map(sum, matrix)) == s.details["sample_count"][DATASETS[i]], "matrix support")
        require(abs(fraction(sum(matrix[j][j] for j in range(len(labels))), sum(map(sum, matrix))) - Decimal(metric["accuracy"])) < Decimal("1e-12"), "matrix accuracy")
        matrices.append(matrix)
    return matrices


def project_phase1_classes(s):
    rows = []
    for ds, matrix in zip(DATASETS, project_matrix_values(s)):
        labels = [c for d, c in CLASS_KEYS if d == ds]
        for j, label in enumerate(labels):
            tp, support, predicted = matrix[j][j], sum(matrix[j]), sum(row[j] for row in matrix)
            rows.append(dict(source_dataset=ds, class_label=label, support=support, precision=fraction(tp, predicted),
                             recall=fraction(tp, support), f1=fraction(2*tp, support+predicted)))
    s.details["calculation_method"] = "Per-class precision, recall and F1 from saved full held-out LightGBM confusion counts."
    return rows


def project_library_data(s, key):
    """Read-only descriptive recipes returning explicit headers and values."""
    def counts(rows, fields, value="count"):
        result = checked_counts(rows, tuple(fields), value)
        return [" / ".join(fields).replace("_", " ").title(), "Count"], [(" / ".join(k), n) for k, n in result.items()]
    if key in ("D01", "D02", "D03", "D05", "D12"):
        closures, values, summary, contract = load_phase1_dataset_stats(s)
        if key in ("D01", "D02"):
            return ["Dataset", "Usable records", "Classes", "Predictors"], [(ds, sum(v.values()), len(v), int(c["predictor_count"])) for ds, v, c in zip(DATASETS, values, closures)]
        if key == "D03":
            return ["Dataset / class", "Records"], [(ds+" / "+k[0], v) for ds, cs in zip(DATASETS, values) for k, v in cs.items()]
        if key == "D05":
            return ["Dataset", "TRAIN", "VALIDATION", "TEST"], [(ds, *v) for ds, v in zip(DATASETS, load_split_stats(s))]
        pre = s.rows(USTC_PRE+"processing_summary.csv")[0]
        cic_audit = s.rows(PROJECT_AUDITS[1])
        require(any(r["check"] == "62_predictors" and r["status"] == "PASS" and "15 TRAIN-zero-variance" in r["details"] for r in cic_audit), "CIC predictor removals audit")
        initial = [int(pre["input_predictor_count"]), int(contract["initial_predictor_candidates"])]
        final = [int(c["predictor_count"]) for c in closures]
        removed = [a-b for a, b in zip(initial, final)]
        require(final == [50, 62] and removed == [10, 15] and removed[0] == int(pre["zero_variance_removed_count"]), "predictor accounting")
        return ["Dataset", "Candidate predictors", "Retained", "TRAIN zero-variance removals"], [(ds, a, b, c) for ds, a, b, c in zip(DATASETS, initial, final, removed)]
    if key == "D09":
        extraction = s.rows(USTC_DATA+"processing_summary.csv")[0]
        closures, _, _, contract = load_phase1_dataset_stats(s)
        return ["Processing measure", "Count"], [("USTC source PCAP files", int(extraction["source_pcap_count"])), ("USTC extracted flows", int(extraction["output_flow_count"])), ("USTC usable flows", sum(int(closures[0][x+"_rows"]) for x in SPLITS)), ("CIC source records", int(contract["raw_rows"])), ("CIC usable records", sum(int(closures[1][x+"_rows"]) for x in SPLITS))]
    if key in ("P01", "P02"):
        rows = []
        for i, models in enumerate(project_phase1_metrics(s)):
            for r in models:
                if key == "P01" and r["model"] != "lightgbm":
                    continue
                for metric in (("accuracy", "precision", "recall", "f1") if i == 0 else ("accuracy", "macro_precision", "macro_recall", "macro_f1")):
                    rows.append((DATASETS[i], PROJECT_MODEL_NAMES[r["model"]], baseline_metric_label(DATASETS[i], metric), format_percent(r[metric])))
        return ["Dataset", "Model", "Metric definition", "Value"], rows
    if key == "P09":
        return ["Dataset", "Actual", "Predicted", "Count"], [(ds, labels[a], labels[p], n) for ds, m in zip(DATASETS, project_matrix_values(s)) for labels in [[c for d, c in CLASS_KEYS if d == ds]] for a, row in enumerate(m) for p, n in enumerate(row)]
    if key in ("A03", "A04", "A05"):
        result = []
        for i, path in enumerate(PROJECT_AUDITS):
            rows = s.rows(path)
            if key == "A03":
                rows = [r for r in rows if r.get("category") == "B" or r.get("check") in ("flow_id_grouping_contract", "split_row_counts")]
            if key == "A04":
                rows = [r for r in rows if r.get("check_name") in ("median_fit_train_only", "variance_filter_train_only", "scaler_fit_train_only", "test_not_used_for_preprocessing") or r.get("check") == "preprocessing_leakage_status"]
            require(rows, "audit selection empty")
            if key == "A05":
                result.extend((DATASETS[i], status, n) for status, n in sorted(Counter(r["status"] for r in rows).items()))
            else:
                result.extend((DATASETS[i], r.get("check_name", r.get("check")), r["status"]) for r in rows)
        s.details["audit_interpretation"] = "Saved checklist evidence and its documented limitations; WARN is not a leakage finding."
        return (["Dataset", "Status", "Checks"] if key == "A05" else ["Dataset", "Saved check", "Status"]), result
    if key == "Q01":
        return counts(load_stage1_query_stats(s)["records"], ("source_dataset", "classifier_prediction"), None)
    if key in ("Q06", "Q07"):
        return counts(load_stage2_corpus_stats(s), ("source_type",), None)
    if key == "Q08":
        _, rows = counts(s.stage("stage02_corpus/corpus_curation_summary.csv"), ("quality_decision",))
        s.details["unit"] = "Saved curation entries, not retained corpus documents"
        return ["Curation decision", "Curation entries"], rows
    if key == "Q10":
        chunks, ref = load_stage3_index_stats(s)
        docs = load_stage2_corpus_stats(s)
        return ["Inventory measure", "Count"], [("Corpus documents", len(docs)), ("Indexed chunks", len(chunks)), ("Embedding dimensions", int(ref["embedding_dimension"])), ("Token limit per chunk", int(ref["token_limit"]))]
    if key == "Q14":
        values, _ = load_stage4_retrieval_stats(s)
        return ["Retrieval method", "Candidate occurrences"], [(k[0], n) for k, n in values.items()]
    if key == "Q17":
        rows = s.stage("stage04_retrieval/retrieval_query_summary.csv")
        unique_rows(rows, ("query_id", "retrieval_method"), "query retrieval methods")
        per_query = Counter()
        for r in rows:
            per_query[r["query_id"]] += int(r["count"])
        require(len(per_query) == 4300 and sum(per_query.values()) == 129000, "retrieval query accounting")
        return ["Candidates per query (all methods)", "Queries"], sorted(Counter(per_query.values()).items())
    if key == "Q19":
        return counts(load_stage5_provenance_stats(s), ("verification_status", "authorization_status"))
    if key == "Q20":
        load_stage5_provenance_stats(s)
        return counts(s.stage("stage05_provenance/provenance_source_type_summary.csv"), ("source_type",))
    if key == "Q22":
        return counts(load_stage6_reranking_stats(s), ("source_type",), None)
    if key in ("Q26", "Q29"):
        chunks = load_stage7_context_stats(s)
        ready = checked_counts(s.stage("stage07_trusted_contexts/trusted_context_summary.csv"), ("context_status",), total=4300)
        sizes = checked_counts(s.stage("stage07_trusted_contexts/trusted_context_chunk_distribution.csv"), ("chunk_count",), total=4300)
        require(ready == {("READY",): 4300} and sizes == {("5",): 4300}, "context readiness/size")
        return ["Context measure", "Count"], [("Ready contexts", 4300), ("Chunks per context", 5), ("Chunk occurrences", len(chunks))]
    if key in ("Q27", "Q30"):
        return counts(load_stage7_context_stats(s), ("source_type",), None)
    if key == "Q31":
        cs = checked_counts(s.stage("stage08_llm/test_llm_evidence_status_summary.csv"), ("evidence_status",), total=4300)
        require([cs[x,] for x in STATUSES] == [2216, 561, 875, 648], "evidence counts")
        return ["Evidence status", "Queries"], [(x, cs[x,]) for x in STATUSES]
    if key == "Q34":
        return ["Response validity", "Responses"], load_stage8_llm_stats(s)
    if key == "Q35":
        return counts(s.stage("stage08_llm/test_llm_reasoning_decision_summary.csv"), ("reasoning_decision",))
    if key == "Q37":
        rows = load_stage9_fusion_stats(s)
        fi = unique_rows(s.stage("stage09_fusion/test_fusion_records.csv"), ("query_id",), "fusion")
        ff = unique_rows(s.stage("stage10_final/test_final_classification_records.csv"), ("query_id",), "final")
        require(len(fi) == 4300 and set(fi) == set(ff), "fusion/final IDs")
        require(all(r["decision_source"] == "retain_classifier" and r["final_class"] == r["classifier_prediction"] == ff[k]["final_class"] for k, r in fi.items()), "zero overrides")
        return ["Fusion action", "Queries"], [("Retained classifier", sum(int(r["count"]) for r in rows)), ("Overrides", 0)]
    if key in ("Q41", "Q42", "Q36", "Q46", "Q47"):
        data = s.final()
        if key == "Q41":
            return ["Population", "Accuracy", "Macro F1", "Weighted F1"], [(ds, *(format_percent(data["metrics"][ds, k]["value"]) for k in ("accuracy", "macro_f1", "weighted_f1"))) for ds in ("OVERALL",)+DATASETS]
        if key == "Q42":
            return ["Class", "Precision", "Recall", "F1", "Support"], [(r["source_dataset"]+" / "+r["class_label"], *(format_percent(r[k]) for k in ("precision", "recall", "f1")), int(r["support"])) for r in data["classes"]]
        if key == "Q36":
            return ["Evidence status", "Correct", "Incorrect", "Accuracy within status"], [(status, a, b, format_percent(fraction(a, a+b))) for status, (a, b) in zip(STATUSES, data["correctness"])]
        if key == "Q46":
            return ["Coverage measure", "Available", "Total", "Coverage"], [(r[0], r[1], r[2], format_percent(fraction(r[1], r[2]))) for i in (2, 4, 6, 5) for r in [data["provenance"][i]]]
        mapping = {r["chunk_id"]: r["document_id"] for r in s.stage("stage07_trusted_contexts/trusted_context_chunks.csv")}
        citations = [c for r in data["records"] for c in json.loads(r["cited_chunk_ids"])]
        require(set(citations) <= set(mapping), "selected citation linkage")
        docs = len({mapping[c] for c in citations})
        require(docs == 51, "distinct selected documents")
        return ["Selected evidence measure", "Count"], [("Citation occurrences", len(citations)), ("Distinct chunks", len(set(citations))), ("Distinct documents", docs), ("Distinct claims", data["provenance"][9][1])]
    if key == "Q44":
        return ["True class to predicted class", "Errors"], [(ds+" / "+labels[a]+" to "+labels[p], n) for ds, m in zip(DATASETS, s.final()["matrices"]) for labels in [[c for d, c in CLASS_KEYS if d == ds]] for a, row in enumerate(m) for p, n in enumerate(row) if a != p and n]
    raise ValueError("Unknown library data recipe: "+key)


def project_display(value):
    text = display_text(value)
    for raw, friendly in {"dense_bge": "Dense BGE", "hybrid_rrf": "Hybrid RRF", "tfidf": "TF-IDF",
                          "retain_classifier": "Retained classifier"}.items():
        text = text.replace(raw, friendly)
    return text.replace("_", " ")


def plot_project_table(s, key):
    import textwrap
    from matplotlib import pyplot as plt
    headers, rows = project_library_data(s, key)
    blocks = [rows[i:i+22] for i in range(0, len(rows), 22)]
    compact = len(headers) == 2 and len(rows) <= 4
    fig, axes = plt.subplots(1, len(blocks), squeeze=False,
                             figsize=((3.5 if compact else 7.16)*len(blocks),
                                      max(1.5 if compact else 2.2, .48*(min(22, len(rows))+2))), layout="constrained")
    for ax, block in zip(axes[0], blocks):
        ax.set_axis_off()
        cells = [[textwrap.fill(project_display(v), 32 if len(headers) == 2 else 22, break_long_words=False) for v in row] for row in block]
        table = ax.table(cellText=cells, colLabels=[textwrap.fill(h, 22) for h in headers], cellLoc="left", bbox=(0, 0, 1, .91))
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        for (i, j), cell in table.get_celld().items():
            cell.visible_edges = "B" if i == 0 else ""
            cell.set_linewidth(.5)
            cell.set_facecolor("white")
            if i == 0:
                cell.get_text().set_weight("bold")
    return fig


def plot_project_data(s, key, pie=False, share=False):
    from matplotlib import pyplot as plt
    headers, rows = project_library_data(s, key)
    if len(headers) != 2 or key in ("Q17", "Q19", "Q26", "Q34", "Q37", "Q47", "D09", "Q10"):
        return plot_project_table(s, key)
    pairs = [(project_display(k), n) for k, n in rows]
    if pie:
        fig = plt.figure(figsize=(7.16, 3.5), layout="constrained")
        grid = fig.add_gridspec(1, 2, width_ratios=(1, 1.3))
        safe_pie(fig.add_subplot(grid[0, 0]), dict(pairs), SOURCE_PALETTE, fig.add_subplot(grid[0, 1]))
        return fig
    if share:
        total = sum(n for _, n in pairs)
        fig, ax = new_axes(size=(7.16, 3.4))
        values = [fraction(n, total) for _, n in pairs]
        bars = ax.barh(range(len(pairs)), [float(v*100) for v in values], color="#0072B2", edgecolor="none")
        annotate_bars(ax, bars, [format_percent(v) for v in values], horizontal=True)
        ax.set_yticks(range(len(pairs)), [k for k, _ in pairs])
        ax.set_xlim(0, 120)
        ax.set_xlabel("Share of context chunk occurrences (%)")
        return fig
    return count_figure(dict(pairs), "", headers[1], horizontal=True)


def plot_project_model_metric(s, metric):
    from matplotlib import pyplot as plt
    models = project_phase1_metrics(s)
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 4), layout="constrained")
    for i, ax in enumerate(axes):
        key = metric if i == 0 or metric == "accuracy" else "macro_"+metric
        rows = models[i]
        bars = ax.barh(range(5), [float(r[key])*100 for r in rows], color=("#0072B2", "#D55E00")[i], edgecolor="none")
        annotate_bars(ax, bars, [format_percent(r[key]) for r in rows], horizontal=True)
        ax.set_yticks(range(5), [PROJECT_MODEL_NAMES[r["model"]] for r in rows])
        ax.invert_yaxis()
        ax.set_xlim(0, 120)
        ax.set_xticks((0, 25, 50, 75, 100))
        ax.set_xlabel(baseline_metric_label(DATASETS[i], key)+" (%)")
        ax.set_title(DATASETS[i], fontsize=9)
        clean_axes(ax)
    return fig


def plot_project_per_class(s, phase1=False):
    rows = project_phase1_classes(s) if phase1 else s.final()["classes"]
    fig = plot_per_class_performance({"classes": rows})
    # The original Phase-II renderer's 90% lower bound is not safe for other
    # populations. Choose a disclosed bound from all displayed saved values.
    lower = max(0, 5*math.floor(min(float(r[k])*100 for r in rows for k in ("precision", "recall", "f1"))/5)-5)
    ax = fig.axes[0]
    ax.set_xlim(lower, 100.2)
    ax.set_xticks(list(range(lower, 101, 5)))
    ax.set_xlabel(f"Per-class performance (%) — axis begins at {lower}%")
    return fig


def plot_project_source_mix(s, kind):
    """Separate stage denominators and actual reranker-input methods."""
    _, retrieval = load_stage4_retrieval_stats(s)
    if kind in ("counts", "shares"):
        groups = [(method, Counter({r["source_type"]: int(r["count"]) for r in retrieval if r["retrieval_method"] == method})) for method in sorted({r["retrieval_method"] for r in retrieval})]
    else:
        reranked = load_stage6_reranking_stats(s)
        methods = {r["original_method"] for r in reranked}
        before = Counter()
        for r in retrieval:
            if r["retrieval_method"] in methods:
                before[r["source_type"]] += int(r["count"])
        after = Counter(r["source_type"] for r in reranked)
        s.details["reranking_input_methods"] = sorted(methods)
        if kind == "retention":
            require(all(after[k] <= before[k] for k in after), "source occurrence retention denominator")
            fig, ax = new_axes(size=(7.16, 3.5))
            keys = sorted(before)
            values = [fraction(after[k], before[k]) for k in keys]
            bars = ax.barh(range(len(keys)), [float(v*100) for v in values], color="#0072B2", edgecolor="none")
            annotate_bars(ax, bars, [format_percent(v) for v in values], horizontal=True)
            ax.set_yticks(range(len(keys)), [project_display(k) for k in keys])
            ax.set_xlim(0, 120)
            ax.set_xlabel("Output occurrences / input-method occurrences (%)")
            s.details["denominator"] = dict(before)
            return fig
        groups = [("Retrieval inputs", before), ("Reranked", after)]
        if kind == "progression":
            corpus = Counter(r["source_type"] for r in load_stage2_corpus_stats(s))
            all_retrieval = Counter()
            for r in retrieval:
                all_retrieval[r["source_type"]] += int(r["count"])
            groups = [("Corpus documents", corpus), ("Retrieved occurrences", all_retrieval), ("Reranked occurrences", after), ("Context chunk occurrences", Counter(r["source_type"] for r in load_stage7_context_stats(s)))]
    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=(7.16, 4.5))
    fig._publication_layout_locked = True
    grid = fig.add_gridspec(2, 1, height_ratios=(3, 1), left=.30, right=.97, bottom=.03, top=.86, hspace=.3)
    ax = fig.add_subplot(grid[0])
    categories = sorted(set().union(*(set(c) for _, c in groups)))
    left = [0.0]*len(groups)
    for j, category in enumerate(categories):
        values = [float(c.get(category, 0) if kind == "counts" else fraction(c.get(category, 0), sum(c.values()))*100) for _, c in groups]
        ax.barh(range(len(groups)), values, left=left, color=SOURCE_PALETTE[j], edgecolor="none", linewidth=0, label=project_display(category))
        left = [a+b for a, b in zip(left, values)]
    ax.set_yticks(range(len(groups)), [project_display(label)+f" (n={sum(c.values()):,})" for label, c in groups])
    ax.set_xlabel("Candidate occurrences" if kind == "counts" else "Source share within each row (%)")
    clean_axes(ax)
    _place_bottom_legend(fig.add_subplot(grid[1]), ax, 2)
    s.details["denominators"] = {label: sum(c.values()) for label, c in groups}
    return fig


def project_final_candidates():
    registry = []
    def add(fid, title, category, scope, paths, build, fields, calculation, stage=0, alias="", suppression=""):
        import re
        name = fid+"_"+re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")
        registry.append(ProjectFigure(name, title, category, scope, tuple(dict.fromkeys(paths)), fields, calculation,
                                     build, stage, not (alias or suppression), alias, suppression))
    structure, phase1, phase2 = "DATASET_STRUCTURE", "PHASE1_FULL_TEST", "PHASE2_FINAL_4300"
    metrics = PROJECT_CLOSURES+PROJECT_PHASE1_TEST_FILES
    cm = metrics+PROJECT_CONFUSIONS+("metadata/traffic/cic_label_mapping.csv",)
    predictor = PROJECT_DATASET_FILES+(USTC_PRE+"processing_summary.csv", PROJECT_AUDITS[1])
    prep = PROJECT_DATASET_FILES+(USTC_DATA+"processing_summary.csv", USTC_PRE+"processing_summary.csv")
    dataset_specs = [
        ("D01", "Dataset Audit Summary", PROJECT_DATASET_FILES+PROJECT_AUDITS+(USTC_PRE+"split_family_assignment.csv",), plot_project_dataset_audit),
        ("D02", "Dataset Size", PROJECT_DATASET_FILES, plot_project_dataset_size),
        ("D03", "Dataset Class Distribution", PROJECT_DATASET_FILES, lambda s: plot_dataset_overview(s)),
        ("D04", "Dataset Class Composition", PROJECT_DATASET_FILES, lambda s: plot_dataset_overview(s, pies=True)),
        ("D05", "Dataset Split Counts", PROJECT_SPLIT_FILES, lambda s: plot_dataset_overview(s, splits=True)),
        ("D06", "Dataset Split Composition", PROJECT_SPLIT_FILES, lambda s: plot_dataset_overview(s, pies=True, splits=True)),
        ("D07", "USTC Family Distribution", PROJECT_DATASET_FILES+(USTC_DATA+"family_distribution.csv",), plot_family_distribution),
        ("D08", "USTC Family-Disjoint Split", PROJECT_DATASET_FILES+(USTC_PRE+"split_family_assignment.csv",), lambda s: plot_family_distribution(s, True)),
        ("D09", "Flow Extraction and Preparation", prep, lambda s: plot_project_data(s, "D09")),
        ("D10", "Preprocessing Retention", prep, lambda s: plot_preparation(s, True)),
        ("D11", "Traffic Representation", (USTC_DATA+"feature_summary.csv", "metadata/traffic/cic_feature_contract.csv"), plot_feature_roles),
        ("D12", "Predictor Summary", predictor, lambda s: plot_project_data(s, "D12"))]
    for fid, title, paths, build in dataset_specs:
        add(fid, title, "dataset", "AUDIT" if fid == "D01" else structure, paths, build,
            "dataset; class/flow_count; partition counts; schema roles; predictor counts; saved audit status as applicable", title+": saved counts and structural accounting, no performance estimates.")
    p_specs = [("P01", "Classification Performance", metrics, plot_project_performance),
               ("P02", "Five-Model Comparison", metrics, lambda s: plot_project_performance(s, True)),
               ("P03", "Model Accuracy Comparison", metrics, lambda s: plot_project_model_metric(s, "accuracy")),
               ("P04", "Model F1 Comparison", metrics, lambda s: plot_project_model_metric(s, "f1")),
               ("P05", "Model Precision Comparison", metrics, lambda s: plot_project_model_metric(s, "precision")),
               ("P06", "Model Recall Comparison", metrics, lambda s: plot_project_model_metric(s, "recall")),
               ("P07", "ROC Curves", PROJECT_CLOSURES, project_missing_test_scores),
               ("P08", "Precision-Recall Curves", PROJECT_CLOSURES, project_missing_test_scores),
               ("P09", "Confusion Matrices", cm, project_phase1_confusions),
               ("P10", "Per-Class Performance", cm, lambda s: plot_project_per_class(s, True)),
               ("P11", "Prediction Confidence", PROJECT_CLOSURES, project_missing_source),
               ("P12", "Confidence by Class", PROJECT_CLOSURES, project_missing_source),
               ("P13", "Classification Errors", PROJECT_CLOSURES, project_missing_source)]
    for fid, title, paths, build in p_specs:
        add(fid, title, "phase1", phase1, paths, build, "model,split,run_id,accuracy,precision,recall,f1,macro_precision,macro_recall,macro_f1,count; sample IDs/scores required for P07/P08/P11/P12/P13",
            title+": full held-out results; fixed model order; USTC Binary F1 and CIC Macro F1 kept separate. P10 is calculated from saved confusion counts; no model selection.")
    for fid, title, paths, build in [
        ("A01", "Leakage Check Summary", PROJECT_AUDITS+(PROJECT_DUPLICATE_AUDIT,), plot_project_leakage),
        ("A02", "Cross-Split Duplicate Check", (PROJECT_DUPLICATE_AUDIT,), plot_integrity_audit),
        ("A03", "Split Integrity", PROJECT_AUDITS, lambda s: plot_project_table(s, "A03")),
        ("A04", "Preprocessing Isolation", PROJECT_AUDITS, lambda s: plot_project_table(s, "A04")),
        ("A05", "Integrity Audit", PROJECT_AUDITS, lambda s: plot_project_table(s, "A05"))]:
        add(fid, title, "audit", "AUDIT", paths, build, "category,section,check_name,check,status,shared_distinct_rows", title+": authentic saved checklist statuses; preserve WARN and documented limits.")
    q_specs = [
        (1, "Query Class Distribution", 1, lambda s: plot_project_data(s, "Q01")),
        (2, "Query Dataset Composition", 1, plot_query_dataset_composition),
        (3, "Query Confidence Distribution", 1, lambda s: plot_query_confidence(s, saved_histogram=True)),
        (4, "Query Probability Summary", 1, plot_query_probabilities),
        (5, "Confidence by Predicted Class", 1, lambda s: plot_query_confidence(s, by_class=True)),
        (6, "Evidence Corpus", 2, plot_project_corpus),
        (7, "Corpus Source Distribution", 2, lambda s: plot_project_data(s, "Q07", pie=True)),
        (8, "Corpus Document Summary", 2, lambda s: plot_project_data(s, "Q08")),
        (9, "Corpus Claim Counts", 2, project_missing_source),
        (10, "Evidence Indexing", 3, lambda s: plot_project_data(s, "Q10")),
        (11, "Chunks per Document", 3, lambda s: plot_index_stats(s, "documents")),
        (12, "Index Token Distribution", 3, lambda s: plot_index_stats(s, "tokens")),
        (13, "Index Source Composition", 3, lambda s: plot_index_stats(s, "source")),
        (14, "Retrieval Methods", 4, plot_project_retrieval),
        (15, "Retrieval Source Distribution", 4, lambda s: plot_project_source_mix(s, "counts")),
        (16, "Retrieval Rank Distribution", 4, lambda s: plot_retrieval(s, "ranks")),
        (17, "Retrieved Candidates per Query", 4, lambda s: plot_project_data(s, "Q17")),
        (18, "Retrieval Source Mix", 4, lambda s: plot_project_source_mix(s, "shares")),
        (19, "Provenance Status", 5, lambda s: plot_project_data(s, "Q19")),
        (20, "Verified Source Types", 5, plot_project_provenance),
        (21, "Verification Rate", 5, lambda s: plot_project_data(s, "Q19")),
        (22, "Evidence after Reranking", 6, lambda s: plot_project_data(s, "Q22")),
        (23, "Reranking Score Distribution", 6, lambda s: plot_reranking_comparison(s, "scores")),
        (24, "Retrieval and Reranking", 6, lambda s: plot_project_source_mix(s, "comparison")),
        (25, "Source Retention after Reranking", 6, lambda s: plot_project_source_mix(s, "retention")),
        (26, "Trusted Context Coverage", 7, lambda s: plot_project_data(s, "Q26")),
        (27, "Trusted Context Sources", 7, lambda s: plot_project_data(s, "Q27")),
        (28, "Document Diversity", 7, lambda s: plot_context_stats(s, "diversity")),
        (29, "Context Chunk Distribution", 7, lambda s: plot_project_data(s, "Q29")),
        (30, "Trusted Context Source Share", 7, lambda s: plot_project_data(s, "Q30", share=True)),
        (31, "Evidence Status", 8, lambda s: plot_project_data(s, "Q31")),
        (32, "Evidence Status by Dataset", 8, lambda s: plot_evidence_status(s, "datasets")),
        (33, "Evidence Status by Class", 8, lambda s: plot_evidence_status(s, "classes")),
        (34, "Response Validity", 8, lambda s: plot_project_data(s, "Q34")),
        (35, "Reasoning Decision", 8, lambda s: plot_project_data(s, "Q35")),
        (36, "Evidence Status and Correctness", 8, lambda s: plot_evidence_and_correctness(s.final())),
        (37, "Decision Fusion", 9, lambda s: plot_project_data(s, "Q37")),
        (38, "Evidence Status under Decision Fusion", 9, lambda s: plot_project_data(s, "Q31")),
        (39, "Fusion Action by Evidence Status", 9, lambda s: plot_project_data(s, "Q37")),
        (40, "Final Classification", 10, plot_project_final),
        (41, "Classification Performance", 10, plot_final_metrics),
        (42, "Per-Class Performance", 10, plot_project_per_class),
        (43, "Final Class Distribution", 10, plot_final_class_distribution),
        (44, "Final Classification Errors", 10, lambda s: plot_project_data(s, "Q44")),
        (45, "Correctness by Evidence Status", 10, lambda s: plot_project_data(s, "Q36")),
        (46, "Provenance Coverage", 10, lambda s: plot_provenance_coverage(s.final())),
        (47, "Selected Evidence Summary", 10, lambda s: plot_project_data(s, "Q47")),
        (48, "Evidence and Provenance Summary", 10, lambda s: plot_project_data(s, "Q46"))]
    aliases = {29: "Q26", 38: "Q31", 43: "Q01", 45: "Q36", 48: "Q46"}
    low = {21: "Uniform verified/authorized status is already in Q19 and Q20.", 39: "REDUNDANT_LOW_INFORMATION: all statuses retain classifier; Q37 validates every query."}
    stage_fields = {
        1: "query_id,source_dataset,source_split,classifier_prediction,classifier_confidence,confidence_bin,class_label,mean_probability,min_probability,max_probability,count",
        2: "document_id,source_type,document_count,quality_decision,count",
        3: "chunk_id,document_id,source_type,chunk_token_count,chunk_count,document_count,embedding_dimension,token_limit,count",
        4: "query_id,retrieval_method,source_type,rank,count",
        5: "source_dataset,source_type,verification_status,authorization_status,count",
        6: "query_id,original_method,final_rank,reranking_score,source_type; retrieval_method,count for comparison/retention",
        7: "query_id,chunk_id,document_id,source_type,context_status,chunk_count,distinct_document_count,count",
        8: "query_id,source_dataset,evidence_status,response_valid,reasoning_decision,final_class,correct,count",
        9: "query_id,decision_source,classifier_prediction,final_class,evidence_status,count",
        10: "query_id,source_dataset,true_class_label,final_class,actual_class,predicted_class,metric,value,sample_size,precision,recall,f1,support,evidence_status,correct,provenance_available,explanation_available,cited_chunk_ids,claim_ids_json,count"}
    for number, title, stage, build in q_specs:
        paths = PROJECT_STAGE_FILES[stage]
        if number == 10:
            paths += PROJECT_STAGE_FILES[2]
        if number in (24, 25):
            paths += PROJECT_STAGE_FILES[4]
        add(f"Q{number:02d}", title, f"phase2/stage{stage:02d}", phase2, paths, build,
            stage_fields[stage],
            title+": deterministic saved-record aggregation; occurrence units differ from query units. Evidence status is not classifier correctness. Confidence boxplots omit outlier marks only.", stage,
            aliases.get(number, ""), low.get(number, ""))
    all_stage = tuple(dict.fromkeys(p for files in PROJECT_STAGE_FILES.values() for p in files))
    for fid, title, build, alias in [
        ("X01", "Query Record Progression", lambda s: plot_progression(s, True), ""),
        ("X02", "Evidence Processing Progression", lambda s: plot_progression(s), ""),
        ("X03", "Evidence Source Progression", lambda s: plot_project_source_mix(s, "progression"), ""),
        ("X04", "Evidence Processing Funnel", lambda s: plot_progression(s), "X02"),
        ("X05", "Final Analysis Summary", lambda s: plot_final_metrics(s), "Q41")]:
        add(fid, title, "cross_stage", phase2, all_stage, build, "query_id,source_type,retrieval_method,count,final_rank,context_status,response_valid,decision_source",
            title+": comparable query counts or explicitly stage-specific occurrences; source shares use separate denominators.", alias=alias)
    add("M01", "Complete Methodology Pipeline", "methodology", "METHODOLOGY", (), plot_methodology_pipeline,
        "METHODOLOGY_BRANCHES,METHODOLOGY_STAGES,METHODOLOGY_CAPTION", METHODOLOGY_CAPTION)
    table_specs = [("Dataset Summary", "D01"), ("Dataset Class Summary", "D03"), ("Dataset Split Summary", "D05"),
                   ("Predictor Summary", "D12"), ("Five-Model Final Performance", "P02"), ("Selected Classifier Performance", "P01"),
                   ("Phase-I Confusion Matrix Counts", "P09"), ("Integrity Audit Summary", "A05"), ("Evidence Corpus and Index Summary", "Q10"),
                   ("Retrieval Summary", "Q14"), ("Provenance Verification Summary", "Q20"), ("Reranking Summary", "Q22"),
                   ("Trusted Context Summary", "Q26"), ("LLM Reasoning Summary", "Q34"), ("Evidence Status Summary", "Q31"),
                   ("Fusion Summary", "Q37"), ("Final Classification Performance", "Q41"), ("Per-Class Final Performance", "Q42"),
                   ("Correctness by Evidence Status", "Q36"), ("Final Provenance Summary", "Q46")]
    by_id = {c.figure_id: c for c in registry}
    table_aliases = {4: "D12", 8: "A05", 9: "Q10", 13: "Q26", 14: "Q34", 16: "Q37"}
    for number, (title, recipe) in enumerate(table_specs, 1):
        parent = by_id[recipe]
        add(f"TF{number:02d}", title, "table_figures", parent.result_scope, parent.source_files,
            lambda s, key=recipe: plot_project_table(s, key), parent.source_fields,
            "Exact-value table: "+parent.calculation, alias=table_aliases.get(number, ""))
    return registry


def audit_project_registry(candidates):
    expected = {f"{prefix}{i:02d}" for prefix, n in (("D", 12), ("P", 13), ("A", 5), ("Q", 48), ("X", 5), ("M", 1), ("TF", 20)) for i in range(1, n+1)}
    indexed = {c.figure_id: c for c in candidates}
    require(set(indexed) == expected and len(indexed) == len(candidates), "comprehensive project registry")
    require({c.stage for c in candidates if c.render and c.stage} == set(range(1, 11)), "canonical coverage of ten stages")
    for c in candidates:
        require(c.result_scope in {"DATASET_STRUCTURE", "PHASE1_FULL_TEST", "PHASE2_FINAL_4300", "AUDIT", "METHODOLOGY"}, "scope")
        require(callable(c.build) and c.source_fields and (c.source_files or c.result_scope == "METHODOLOGY"), "incomplete definition")
        require(display_text(c.title) == c.title and "TEST" not in c.title, "visible title policy")
        if c.duplicate_of:
            require(c.duplicate_of in indexed and indexed[c.duplicate_of].render and not c.render, "alias must target physical canonical")
            require(indexed[c.duplicate_of].result_scope == c.result_scope, "alias scope")
        if c.result_scope == "PHASE1_FULL_TEST":
            require(all("validation" not in p.lower() and "train_results" not in p.lower() and not p.startswith("results/rag/") for p in c.source_files), "non-TEST Phase-I source")
        if c.result_scope == "PHASE2_FINAL_4300":
            require(all(p.startswith(TEST_NAMESPACE.as_posix()+"/") for p in c.source_files), "Phase-II namespace")
    physical = [c for c in candidates if c.render]
    require(len({c.name for c in physical}) == len(physical), "duplicate output basename")
    require(len({(c.category, c.name) for c in physical}) == len(physical), "output path collision")


def verify_project_alias_sources(s):
    """Confirm cross-stage aliases from saved IDs/counts before any cleanup."""
    queries = unique_rows(load_stage1_query_stats(s)["records"], ("query_id",), "query alias IDs")
    final = unique_rows(s.stage("stage10_final/test_final_classification_records.csv"), ("query_id",), "final alias IDs")
    fusion = unique_rows(s.stage("stage09_fusion/test_fusion_records.csv"), ("query_id",), "fusion alias IDs")
    require(len(queries) == 4300 and set(queries) == set(final) == set(fusion), "alias populations")
    require(all(queries[k]["source_dataset"] == final[k]["source_dataset"] and
                queries[k]["classifier_prediction"] == final[k]["final_class"] == fusion[k]["final_class"] and
                fusion[k]["decision_source"] == "retain_classifier" for k in queries), "final-class alias equivalence")
    status = checked_counts(s.stage("stage08_llm/test_llm_evidence_status_summary.csv"), ("evidence_status",), total=4300)
    require(status == checked_counts(list(final.values()), ("evidence_status",), None, 4300) ==
            checked_counts(list(fusion.values()), ("evidence_status",), None, 4300), "status alias equivalence")
    require(all(final[k]["evidence_status"] == fusion[k]["evidence_status"] for k in final), "status alias ID match")
    load_stage5_provenance_stats(s)
    project_library_data(s, "Q26")
    s.details["alias_verification"] = "Q43/Q01 matched query IDs and predictions; Q38/Q31 status totals and fusion/final IDs; Q29/Q26 saved context size. Other aliases reuse identical recipes."
    return s.details["alias_verification"]


def prepare_project_output(root):
    """Clear only this explicit new output root; reject all link redirection."""
    import os
    import shutil
    import stat
    root = root.resolve()
    output = root / "final_paper_figures" / "project_final_test_figures"
    def check(path):
        require(path.resolve() == path and not path.is_symlink(), "redirected project output")
        if path.exists():
            require(not (getattr(path.lstat(), "st_file_attributes", 0) &
                         getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)), "project output reparse point")
    check(output.parent)
    check(output)
    require(output.parent == root/"final_paper_figures" and output.name == "project_final_test_figures", "project cleanup boundary")
    if output.exists():
        for current, directories, files in os.walk(output, followlinks=False):
            for name in directories+files:
                child = Path(current)/name
                require(child.resolve().is_relative_to(output), "project cleanup escape")
                check(child)
        shutil.rmtree(output)
    output.mkdir(parents=True)
    for folder in ("dataset", "phase1", "audit", "cross_stage", "methodology", "table_figures") + tuple(f"phase2/stage{i:02d}" for i in range(1, 11)):
        (output/folder).mkdir(parents=True)
    return output


def prepare_project_final_test_figures():
    """Explicit manual-only, TEST-only project pack; old modes are untouched."""
    from matplotlib import pyplot as plt
    from matplotlib.text import Text
    import re
    root = Path(__file__).resolve().parent
    candidates = project_final_candidates()
    audit_project_registry(candidates)
    alias_sources = ProjectFigureSources(root, next(c for c in candidates if c.figure_id == "X01"))
    alias_verification = verify_project_alias_sources(alias_sources)
    output = prepare_project_output(root)
    manifest, source_hashes = [], dict(alias_sources.hashes)
    for c in candidates:
        s = ProjectFigureSources(root, c)
        base = checked_output_path(output, Path(c.category)/c.name)
        record = dict(figure_id=c.figure_id, title=c.title, category=c.category, section=c.category,
                      stage=c.stage or "", result_scope=c.result_scope, dpi=600, visible_scope_label="",
                      canonical_id=c.duplicate_of or c.figure_id, duplicate_of=c.duplicate_of, render=c.render,
                      source_files=" | ".join(c.source_files), source_fields=c.source_fields, calculation=c.calculation,
                      generated=False, status="ASSERTION_FAILED", skip_reason="", source_issue="",
                      png_path=str(base.with_suffix(".png")), pdf_path=str(base.with_suffix(".pdf")),
                      svg_path=str(base.with_suffix(".svg")), source_trace="", source_files_read="",
                      proposed_external_caption=c.title+". "+c.result_scope+". "+c.calculation)
        if not c.render:
            record.update(status="REDUNDANT_SUPPRESSED" if c.duplicate_of else "LOW_VALUE_SUPPRESSED",
                          skip_reason="REDUNDANT_EQUIVALENT: use canonical " + c.duplicate_of if c.duplicate_of else c.suppression_reason,
                          png_path="", pdf_path="", svg_path="")
            manifest.append(record)
            continue
        try:
            with rc_context(PUBLICATION_STYLE):
                fig = c.build(s)
                axes = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]
                if len(axes) == 1:
                    for loc in ("left", "center", "right"):
                        axes[0].set_title("", loc=loc)
                if c.figure_id != "M01":
                    safe_suptitle(fig, c.title)
                if not fig.get_constrained_layout() and not getattr(fig, "_publication_layout_locked", False):
                    fig.subplots_adjust(top=min(fig.subplotpars.top, .84))
                if c.figure_id in ("D05", "D06", "D08"):
                    for text in fig.findobj(Text):
                        text.set_text(re.sub(r"\b(train|validation|test)\b", lambda m: m[0].upper(), text.get_text(), flags=re.I))
                prepare_visible_text(fig)
                for artist in fig.findobj(Text):
                    text = project_display(artist.get_text())
                    text = re.sub(r"\b(?:held-out\s+|final\s+)?test\s+(?:set\s+)?only\b", "", text, flags=re.I)
                    text = text.replace("Number of TEST queries", "Queries")
                    if text == "Evidence status ≠ classifier correctness":
                        text = ""  # This interpretation belongs in the caption.
                    artist.set_text(text)
                    require(not re.search(r"frozen|cohort|\bV[45]\b|test\s+(?:set\s+)?only", text, re.I), "project visible wording")
                if c.figure_id == "M01":
                    assert_methodology_text(fig)
                require(s.used.issubset(set(c.source_files)), "undeclared source trace")
                for ext in ("png", "pdf", "svg"):
                    checked_output_path(output, base.relative_to(output).with_suffix("."+ext))
                export_figure(fig, base)
                require(all(base.with_suffix("."+ext).is_file() and base.with_suffix("."+ext).stat().st_size > 0
                            for ext in ("png", "pdf", "svg")), "three nonempty output formats")
            record.update(generated=True, status="GENERATED")
        except (SourceNotFound, FileNotFoundError) as exc:
            record.update(status="SOURCE_NOT_FOUND", skip_reason=("TEST_SCORE_SOURCE_NOT_FOUND" if str(exc) == "TEST_SCORE_SOURCE_NOT_FOUND"
                          else "SOURCE_NOT_FOUND"), source_issue=s.details.get("source_issue", str(exc)))
        except Exception as exc:
            record["skip_reason"] = f"ASSERTION_FAILED: {type(exc).__name__}: {exc}"
        finally:
            plt.close("all")
            if not record["generated"]:
                for ext in ("png", "pdf", "svg"):
                    path = checked_output_path(output, base.relative_to(output).with_suffix("."+ext))
                    if path.exists():
                        path.unlink()
            record["source_trace"] = json.dumps(s.details, sort_keys=True)
            record["source_files_read"] = " | ".join(sorted(s.used))
            source_hashes.update(s.hashes)
            manifest.append(record)
        print(f"{record['status']} {c.name}: {record['skip_reason']}")
    with checked_output_path(output, "Project_Figure_Manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    report = dict(mode="project-final-test-figures", figures=manifest, requested=len(candidates),
                  canonical_definitions=sum(c.render for c in candidates),
                  alias_source_verification=alias_verification,
                  generated=sum(row["generated"] for row in manifest), png_dpi=600,
                  source_not_found=sum(row["status"] == "SOURCE_NOT_FOUND" for row in manifest),
                  suppressed_duplicate=sum(bool(c.duplicate_of) for c in candidates),
                  suppressed_low_value=sum(bool(c.suppression_reason) for c in candidates),
                  phase1_sources=sorted({p for c in candidates if c.result_scope == "PHASE1_FULL_TEST" for p in c.source_files}),
                  phase2_sources=sorted({p for c in candidates if c.result_scope == "PHASE2_FINAL_4300" for p in c.source_files}),
                  audit_sources=PROJECT_AUDITS+(PROJECT_DUPLICATE_AUDIT,),
                  roc_source="TEST_SCORE_SOURCE_NOT_FOUND", pr_source="TEST_SCORE_SOURCE_NOT_FOUND",
                  confusion_sources={"PHASE1_FULL_TEST": PROJECT_CONFUSIONS,
                                     "PHASE2_FINAL_4300": project_stage_paths("stage10_final/test_evaluation_confusion_matrix.csv")},
                  stage_coverage={str(i): [c.figure_id for c in candidates if c.stage == i and c.render] for i in range(1, 11)},
                  output_format_verification={"formats": ["png", "pdf", "svg"], "png_dpi": 600,
                                              "nonempty_files_checked_after_each_export": True},
                  source_hashes=source_hashes, validation_performance_sources_used=False,
                  visible_word_policy={"Frozen": 0, "Cohort": 0, "V4": 0, "V5": 0},
                  scope_notes={"PHASE1_FULL_TEST": "Full held-out splits: USTC 98,255; CIC 31,723; retained LightGBM chosen upstream.",
                               "PHASE2_FINAL_4300": "Final executed 4,300 queries: USTC 1,700; CIC 2,600; not the full Phase-I population."},
                  test_score_source_issue=PROJECT_SCORE_ABSENCE,
                  examined_but_not_eligible_curve_files=PROJECT_EXCLUDED_CURVES,
                  notes=["No training, inference, model loading, threshold tuning or TEST-based model selection.",
                         "Stored full-TEST ROC/PR coordinate files were found but deliberately not substituted for the required per-sample ID/score sources.",
                         "No validation ROC/PR is read. Existing execution modes and complete-paper outputs are independent."])
    checked_output_path(output, "Preparation_Report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 1 if any(row["status"] == "ASSERTION_FAILED" for row in manifest) else 0


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--final-test-results", action="store_true",
                        help="Generate publication figures/tables from frozen saved artifacts only; preserve legacy figures.")
    modes.add_argument("--complete-paper-figures", action="store_true",
                       help="Generate the complete source-backed publication library; never run research pipelines.")
    modes.add_argument("--project-final-test-figures", action="store_true",
                       help="Generate the comprehensive library from full Phase-I TEST and final Phase-II saved sources.")
    args = parser.parse_args()
    if args.project_final_test_figures:
        return prepare_project_final_test_figures()
    if args.complete_paper_figures:
        return prepare_complete_paper_figures()
    return prepare_final_test_results() if args.final_test_results else main()


if __name__ == "__main__":
    raise SystemExit(cli())
