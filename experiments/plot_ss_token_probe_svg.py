import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(description="Render a paper-style SVG figure for SS token probe results.")
    parser.add_argument("--summary", type=str, required=True, help="Path to summary.json from ss_token_probe.")
    parser.add_argument("--token-csv", type=str, required=True, help="Path to token_metrics.csv from ss_token_probe.")
    parser.add_argument("--output", type=str, default="output/ss_token_probe_figure.svg", help="Output SVG path.")
    return parser.parse_args()


def load_summary(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_token_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "grid_x": float(row["grid_x"]),
                "grid_y": float(row["grid_y"]),
                "grid_z": float(row["grid_z"]),
                "token_norm": float(row["token_norm"]),
                "occ_ratio": float(row["occ_ratio"]),
                "surf_ratio": float(row["surf_ratio"]),
                "dist_to_shape_center": float(row["dist_to_shape_center"]),
                "knn_occ_gap": float(row["knn_occ_gap"]),
                "rand_occ_gap": float(row["rand_occ_gap"]),
                "knn_surf_gap": float(row["knn_surf_gap"]),
                "rand_surf_gap": float(row["rand_surf_gap"]),
                "knn_spatial_gap": float(row["knn_spatial_gap"]),
                "rand_spatial_gap": float(row["rand_spatial_gap"]),
            })
    return rows


def setup_style():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 14,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "figure.titlesize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def write_chinese_log(path: Path, summary: dict):
    lines = [
        "SS token 几何一致性分析报告",
        "=" * 32,
        f"缓存位置: {summary['cache']}",
        f"SS latent 张量形状: {tuple(summary['z_s_shape'])}",
        f"token 张量形状: {tuple(summary['token_shape'])}",
        f"decode 后体素分辨率: {summary['decoded_resolution']}",
        f"非空 token block 数量: {summary['num_nonempty_token_blocks']}",
        "",
        "一、总体结论",
        "1. SS token 之间的表征相似性和 decode 后局部几何确实存在稳定联系。",
        "2. token cosine 越高，对应局部 occupancy 差异和 surface 差异越小。",
        "3. KNN 聚在一起的 token，不仅几何统计更接近，空间位置也明显更接近。",
        "",
        "二、关键指标",
        f"- cosine 与 occupancy gap 的相关系数: {summary['pairwise']['corr_cos_vs_occ_gap']:.4f}",
        f"- cosine 与 surface gap 的相关系数: {summary['pairwise']['corr_cos_vs_surf_gap']:.4f}",
        f"- cosine 与空间距离的相关系数: {summary['pairwise']['corr_cos_vs_spatial_dist']:.4f}",
        f"- L2 与 occupancy gap 的相关系数: {summary['pairwise']['corr_l2_vs_occ_gap']:.4f}",
        f"- L2 与 surface gap 的相关系数: {summary['pairwise']['corr_l2_vs_surf_gap']:.4f}",
        "",
        "三、KNN 对比",
        f"- occupancy gap: KNN = {summary['knn']['mean_knn_occ_gap']:.4f}, Random = {summary['knn']['mean_rand_occ_gap']:.4f}",
        f"- surface gap: KNN = {summary['knn']['mean_knn_surf_gap']:.4f}, Random = {summary['knn']['mean_rand_surf_gap']:.4f}",
        f"- spatial gap: KNN = {summary['knn']['mean_knn_spatial_gap']:.4f}, Random = {summary['knn']['mean_rand_spatial_gap']:.4f}",
        "",
        "四、解释",
        "1. 这说明 SS 阶段的 token 不是纯粹抽象特征，它已经编码了局部几何占有率、边界复杂度以及空间邻近性。",
        "2. 从结果看，token 相似度越高，通常会落到更相似的局部结构区域，而不是随机分布到完全不同的几何位置。",
        "3. 因此后续如果想做 token 级插值、匹配、聚类或可控编辑，SS token 是可以直接作为几何语义单元来处理的。",
        "",
        "五、图中各面板含义",
        "1. 左上: decode 后 occupancy 在 token 网格上的切片热图。",
        "2. 中上: decode 后 surface density 在 token 网格上的切片热图。",
        "3. 右上: KNN 与随机邻居在几何差异上的对比柱状图。",
        "4. 左下: 每个 token 的 occupancy 与其 KNN occupancy gap 的关系散点图。",
        "5. 中下: 表征距离/相似度与几何差异之间的相关性汇总。",
        "6. 右下: 非空 token 的 3D 占据云图，用颜色表示 occupancy ratio。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =========================
# 单独微调坐标轴位置的工具函数
# =========================
def nudge(ax, dx: float = 0.0, dy: float = 0.0, dw: float = 0.0, dh: float = 0.0):
    """
    微调某一个子图的位置和尺寸。

    参数解释:
    dx > 0  向右移动
    dx < 0  向左移动
    dy > 0  向上移动
    dy < 0  向下移动
    dw > 0  变宽
    dw < 0  变窄
    dh > 0  变高
    dh < 0  变矮

    重要说明:
    用 GridSpec / subplot 创建出来的 ax，默认会被 matplotlib 的 locator 接管位置。
    如果不先解除 locator，set_position 看起来就像“没效果”。
    所以这里会先 ax.set_axes_locator(None)，再手动改位置。

    常用例子:
    nudge(ax_bar, dx=-0.01)     # 向左一点
    nudge(ax_corr, dy=-0.01)    # 向下一点
    nudge(ax_cloud, dw=0.01)    # 宽一点
    nudge(ax_cloud, dh=0.01)    # 高一点
    """
    ax.set_axes_locator(None)
    pos = ax.get_position()
    ax.set_position([pos.x0 + dx, pos.y0 + dy, pos.width + dw, pos.height + dh], which="both")


# =========================
# 固定 colorbar 位置的函数
# 不再用 inset_axes，避免保存时乱跑
# =========================
def add_fixed_cbar(fig, ax, mappable, label: str,
                   pad: float = 0.006,
                   width: float = 0.012,
                   shrink: float = 0.88):
    """
    给某个子图创建一个“固定坐标”的 colorbar。

    参数解释:
    pad     控制 colorbar 离主图左右距离
            pad > 0  更靠右
            pad < 0  更靠左（一般不建议）

    width   控制 colorbar 宽度
            width 变大，色条更粗
            width 变小，色条更细

    shrink  控制 colorbar 高度占主图的比例
            shrink = 0.88 表示 colorbar 高度 = 主图高度的 88%
            shrink 越大，色条越长
            shrink 越小，色条越短
    """
    pos = ax.get_position()
    cax = fig.add_axes([
        pos.x1 + pad,
        pos.y0 + (1.0 - shrink) * 0.5 * pos.height,
        width,
        pos.height * shrink,
    ])
    cb = fig.colorbar(mappable, cax=cax)
    cb.set_label(label)
    return cb, cax


def build_slice_grids(rows, z_mid: int, grid_size: int = 16):
    occ_grid = np.full((grid_size, grid_size), np.nan, dtype=np.float32)
    surf_grid = np.full((grid_size, grid_size), np.nan, dtype=np.float32)
    for r in rows:
        if int(r["grid_z"]) == z_mid:
            occ_grid[int(r["grid_y"]), int(r["grid_x"])] = r["occ_ratio"]
            surf_grid[int(r["grid_y"]), int(r["grid_x"])] = r["surf_ratio"]
    return occ_grid, surf_grid


def main():
    args = parse_args()
    summary = load_summary(Path(args.summary))
    rows = load_token_rows(Path(args.token_csv))

    setup_style()

    gx = np.array([r["grid_x"] for r in rows])
    gy = np.array([r["grid_y"] for r in rows])
    gz = np.array([r["grid_z"] for r in rows])
    occ = np.array([r["occ_ratio"] for r in rows])
    surf = np.array([r["surf_ratio"] for r in rows])
    knn_occ_gap = np.array([r["knn_occ_gap"] for r in rows])

    occupied = occ > 0
    sel = occupied if occupied.any() else np.ones_like(occ, dtype=bool)
    z_mid = int(np.median(gz[sel]))
    occ_grid, surf_grid = build_slice_grids(rows, z_mid=z_mid, grid_size=16)

    # =========================
    # 先创建 2x3 的六个标准子图
    # =========================
    fig = plt.figure(figsize=(14.2, 7.6))
    # 把整张图的边距直接写进 GridSpec，后面就不要再调用 fig.subplots_adjust() 了
    # left/right/bottom/top 的作用和 subplots_adjust 一样，但更稳定
    gs = fig.add_gridspec(
        2, 3,
        left=0.06,
        right=0.965,
        bottom=0.08,
        top=0.95,
        wspace=0.30,
        hspace=0.34,
    )

    ax_occ = fig.add_subplot(gs[0, 0])
    ax_surf = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[0, 2])
    ax_scatter = fig.add_subplot(gs[1, 0])
    ax_corr = fig.add_subplot(gs[1, 1])
    ax_cloud = fig.add_subplot(gs[1, 2], projection="3d")

    # =========================
    # (a) occupancy 热图
    # =========================
    occ_vmax = max(0.2, float(np.nanmax(occ_grid))) if np.isfinite(np.nanmax(occ_grid)) else 0.2
    im0 = ax_occ.imshow(occ_grid, cmap="YlOrRd", origin="lower", vmin=0.0, vmax=occ_vmax)
    ax_occ.set_title(f"(a) Occupancy Slice on Token Grid  |  z = {z_mid}")
    ax_occ.set_xlabel("grid x")
    ax_occ.set_ylabel("grid y")
    ax_occ.set_xticks([])
    ax_occ.set_yticks([])

    # =========================
    # (b) surface 热图
    # =========================
    surf_vmax = max(0.12, float(np.nanmax(surf_grid))) if np.isfinite(np.nanmax(surf_grid)) else 0.12
    im1 = ax_surf.imshow(surf_grid, cmap="GnBu", origin="lower", vmin=0.0, vmax=surf_vmax)
    ax_surf.set_title(f"(b) Surface Density Slice  |  z = {z_mid}")
    ax_surf.set_xlabel("grid x")
    ax_surf.set_ylabel("grid y")
    ax_surf.set_xticks([])
    ax_surf.set_yticks([])

    # =========================
    # (c) KNN vs Random 柱状图
    # =========================
    metrics = ["Occ. gap", "Surf. gap", "Spatial gap"]
    knn_vals = [
        summary["knn"]["mean_knn_occ_gap"],
        summary["knn"]["mean_knn_surf_gap"],
        summary["knn"]["mean_knn_spatial_gap"],
    ]
    rand_vals = [
        summary["knn"]["mean_rand_occ_gap"],
        summary["knn"]["mean_rand_surf_gap"],
        summary["knn"]["mean_rand_spatial_gap"],
    ]
    x = np.arange(len(metrics), dtype=float) * 1.22
    w = 0.32
    ax_bar.bar(x - w / 2, knn_vals, width=w, color="#D1495B", label="KNN")
    ax_bar.bar(x + w / 2, rand_vals, width=w, color="#8D99AE", label="Random")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metrics)
    ax_bar.set_xlim(x[0] - 0.62, x[-1] + 0.62)
    ax_bar.set_title("(c) KNN Tokens Decode to Closer Geometry")
    ax_bar.set_ylabel("mean absolute difference")
    ax_bar.legend(frameon=False, loc="upper left")

    # =========================
    # (d) 散点图
    # =========================
    scatter_idx = np.where(sel)[0]
    if scatter_idx.size > 1400:
        rng = np.random.default_rng(0)
        scatter_idx = rng.choice(scatter_idx, size=1400, replace=False)
    ax_scatter.scatter(
        occ[scatter_idx],
        knn_occ_gap[scatter_idx],
        s=14,
        alpha=0.7,
        color="#2A9D8F",
        edgecolors="none",
    )
    ax_scatter.set_title("(d) Per-Token Occupancy vs KNN Gap")
    ax_scatter.set_xlabel("decoded occupancy ratio")
    ax_scatter.set_ylabel("mean KNN occupancy gap")

    # =========================
    # (e) 相关性柱状图
    # =========================
    corr_labels = [
        "cos / occ gap",
        "cos / surf gap",
        "cos / spatial dist",
        "l2 / occ gap",
        "l2 / surf gap",
        "norm / occ gap",
    ]
    corr_vals = [
        summary["pairwise"]["corr_cos_vs_occ_gap"],
        summary["pairwise"]["corr_cos_vs_surf_gap"],
        summary["pairwise"]["corr_cos_vs_spatial_dist"],
        summary["pairwise"]["corr_l2_vs_occ_gap"],
        summary["pairwise"]["corr_l2_vs_surf_gap"],
        summary["pairwise"]["corr_norm_gap_vs_occ_gap"],
    ]
    y = np.arange(len(corr_labels))
    colors = ["#264653" if v < 0 else "#E76F51" for v in corr_vals]
    ax_corr.barh(y, corr_vals, color=colors)
    ax_corr.axvline(0, color="black", linewidth=0.8)
    ax_corr.set_yticks(y)
    ax_corr.set_yticklabels(corr_labels)
    ax_corr.tick_params(axis="y", pad=2)
    ax_corr.invert_yaxis()
    ax_corr.set_xlim(-0.65, 0.65)
    ax_corr.set_title("(e) Pairwise Correlation Summary")
    ax_corr.set_xlabel("correlation")

    # =========================
    # (f) 3D occupancy 点云
    # =========================
    cloud_mask = occ > 0
    cloud_x = gx[cloud_mask]
    cloud_y = gy[cloud_mask]
    cloud_z = gz[cloud_mask]
    cloud_c = occ[cloud_mask]
    if cloud_x.size > 0:
        sc = ax_cloud.scatter(
            cloud_x,
            cloud_y,
            cloud_z,
            c=cloud_c,
            cmap="YlOrRd",
            s=20 + 70 * cloud_c,
            alpha=0.9,
            linewidths=0.2,
            edgecolors="black",
        )
    else:
        sc = None

    ax_cloud.set_title("(f) Occupied Token Cells in 3D")
    ax_cloud.set_xlabel("x")
    ax_cloud.set_ylabel("y")
    ax_cloud.set_zlabel("z")
    ax_cloud.view_init(elev=24, azim=-58)
    ax_cloud.set_box_aspect((1.0, 1.0, 0.8))
    ax_cloud.set_xticks([])
    ax_cloud.set_yticks([])
    ax_cloud.set_zticks([])
    ax_cloud.xaxis.pane.set_alpha(0.0)
    ax_cloud.yaxis.pane.set_alpha(0.0)
    ax_cloud.zaxis.pane.set_alpha(0.0)
    ax_cloud.grid(False)

    # =========================
    # 不再使用 fig.subplots_adjust()
    # 因为它会重新分配所有 subplot 的位置，容易让你误以为 colorbar 在“乱跑”
    # 现在整张图的边距已经在上面的 add_gridspec(left/right/bottom/top=...) 里固定好了
    # =========================

    # 先 draw 一次，让 GridSpec 把六个标准子图的位置真正算出来
    # 否则你在这里 nudge，拿到的可能还不是最终位置

    # =========================
    # 再做单图微调
    # 这里按需取消注释
    # =========================

    # 例1: 把 (b) 往左一点
    nudge(ax_surf, dx=-0.02)

    # 例2: 把 (c) 往左一点
    nudge(ax_bar, dx=0.012)

    # 例3: 把 (e) 往下一点
    nudge(ax_corr, dx=0.04)

    # 例4: 把 (f) 稍微往右、往上一点，同时放大一点
    nudge(ax_cloud, dx=-0.006,dy=-0.01)

    # 例5: 把 (a) 变宽一点
    # nudge(ax_occ, dw=0.01)

    # =========================
    # 注意：colorbar 一定放在最后创建
    # 因为前面你可能还会 nudge 主图
    # 主图位置完全确定后，再创建固定 colorbar
    # =========================
    fig.canvas.draw()
    _, cax_occ = add_fixed_cbar(
        fig, ax_occ, im0, "occupancy ratio",
        pad=0.006,   # 改大 -> 色条更靠右
        width=0.012, # 改大 -> 色条更粗
        shrink=0.88  # 改大 -> 色条更长
    )

    _, cax_surf = add_fixed_cbar(
        fig, ax_surf, im1, "surface ratio",
        pad=0.006,
        width=0.012,
        shrink=0.88
    )

    if sc is not None:
        _, cax_cloud = add_fixed_cbar(
            fig, ax_cloud, sc, "occupancy ratio",
            pad=0.010,
            width=0.012,
            shrink=0.82
        )

        # 如果 3D 那个 colorbar 还想继续微调，就在这里动它
        # 例如：往右一点
        # nudge(cax_cloud, dx=0.004)
        # 例如：往上 一点
        # nudge(cax_cloud, dy=0.004)
    else:
        cax_cloud = None

    # 这两个 colorbar 也都可以单独再调
    # 例如把 (a) 的 colorbar 往左一点:
    # nudge(cax_occ, dx=-0.002)

    # 例如把 (b) 的 colorbar 变细一点:
    # nudge(cax_surf, dw=-0.002)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 不用 bbox_inches="tight"，避免 colorbar 位置在保存时再次被挤来挤去
    fig.savefig(output_path, format="svg")

    log_path = output_path.with_suffix(".log")
    write_chinese_log(log_path, summary)
    print(output_path)


if __name__ == "__main__":
    main()
