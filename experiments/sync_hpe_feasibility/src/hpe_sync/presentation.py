"""Local report formatting and conventional scientific plots."""
import html
import os
import re


def render_html(markdown, image_names=()):
    def inline(text):
        escaped = html.escape(text)
        return re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)

    body, paragraph, table, listing = [], [], False, False

    def flush():
        if paragraph:
            body.append("<p>" + inline(" ".join(paragraph)) + "</p>")
            paragraph.clear()

    for line in markdown.splitlines():
        if line.startswith("|"):
            flush()
            if listing:
                body.append("</ul>")
                listing = False
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(re.fullmatch(r"[-: ]+", c) for c in cells):
                continue
            if not table:
                body.append("<div class='table'><table><thead><tr>" +
                            "".join(f"<th>{inline(c)}</th>" for c in cells) + "</tr></thead><tbody>")
                table = True
            else:
                body.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells) + "</tr>")
            continue
        if table:
            body.append("</tbody></table></div>")
            table = False
        if line.startswith("- "):
            flush()
            if not listing:
                body.append("<ul>")
                listing = True
            body.append("<li>" + inline(line[2:]) + "</li>")
            continue
        if listing:
            body.append("</ul>")
            listing = False
        if line.startswith("#"):
            flush()
            level = min(len(line) - len(line.lstrip("#")), 3)
            body.append(f"<h{level}>{inline(line.lstrip('# '))}</h{level}>")
        elif not line:
            flush()
        else:
            paragraph.append(line)
    flush()
    if table:
        body.append("</tbody></table></div>")
    if listing:
        body.append("</ul>")
    if image_names:
        body.append("<h2>Figures</h2>")
        for name in image_names:
            body.append(f"<figure><img src='{html.escape(name, quote=True)}' "
                        f"alt='{html.escape(name, quote=True)}'></figure>")
    css = """
    body{font:16px/1.6 system-ui,sans-serif;max-width:1150px;margin:36px auto;
         padding:0 28px 50px;color:#172635;background:#fcfcfb}
    h1{font-size:30px;line-height:1.25;color:#143d50}h2{font-size:22px;margin-top:38px}
    p,li{max-width:1000px}code{font-size:.88em;background:#eaf0f3;padding:2px 5px;overflow-wrap:anywhere}
    .table{overflow-x:auto;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:14px}
    th{background:#143d50;color:white;text-align:left}td,th{padding:9px 12px;border-bottom:1px solid #dce4e7}
    tr:nth-child(even){background:#f0f4f5}img{width:100%;height:auto}figure{margin:20px 0}
    @media print{body{font-size:11px;margin:0}h2{break-after:avoid}tr,figure{break-inside:avoid}}
    """
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>19_MM synchronization feasibility</title><style>" + css +
            "</style></head><body>" + "\n".join(body) + "</body></html>")


def plot_results(folder, displacement, comparisons):
    os.environ["MPLCONFIGDIR"] = str(folder / "matplotlib_cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = []
    if displacement:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
        for ax, task in zip(axes, ("image", "shape")):
            rr = [r for r in displacement[task] if r["condition"] not in ("reference", "unrelated")]
            ax.bar(np.arange(len(rr)), [r["raw_displacement_mae"] for r in rr], color="#2c7891")
            ax.set_xticks(np.arange(len(rr)), [r["condition"].replace("shift_", "") for r in rr], rotation=20)
            ax.set_title(f"{task.capitalize()} target changes")
            ax.set_ylabel("Mean absolute change (" + ("pixels" if task == "image" else "torso lengths") + ")")
            ax.grid(axis="y", alpha=.2)
            ax.set_axisbelow(True)
        fig.suptitle("Label changes relative to the saved mapping; common held-out samples", fontsize=12)
        names.append("label_displacement.png")
        fig.savefig(folder / names[-1], dpi=180)
        plt.close(fig)
    if comparisons:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
        for ax, task in zip(axes, ("image", "shape")):
            rr = [r for r in comparisons if r["task"] == task and r["model"] == "pointnet" and r["context"] == 1]
            if rr:
                ax.barh(np.arange(len(rr)), [r["mse"] for r in rr], color="#2c7891")
                ax.set_yticks(np.arange(len(rr)), [r["condition"] for r in rr])
                ax.axvline(rr[0]["mean_baseline_mse"], color="#b85139", linestyle="--", label="Mean-pose baseline")
                ax.legend(fontsize=8)
            ax.set_title(task.capitalize())
            ax.set_xlabel("Test standardized MSE (lower is better)")
        fig.suptitle("Single-frame radar models; uncertainty is reported in the table", fontsize=12)
        names.append("radar_results.png")
        fig.savefig(folder / names[-1], dpi=180)
        plt.close(fig)
    return names
