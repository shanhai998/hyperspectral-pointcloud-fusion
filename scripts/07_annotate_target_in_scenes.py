

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from hyperspectral_pointcloud_fusion.common import load_config, ensure_dir, load_json
from hyperspectral_pointcloud_fusion.geometry import project_world_points_pinhole


ANN_COLUMNS = [
    'scene_id', 'annotation_status',
    'u_top', 'v_top', 'u_base', 'v_base',
    'top_anchor_x', 'top_anchor_y', 'top_anchor_z',
    'base_anchor_x', 'base_anchor_y', 'base_anchor_z',
    'note',
]
ALLOWED_STATUS = {'pending', 'full', 'top_only', 'base_only', 'none'}


def select_target_points(xyz: np.ndarray, target_cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:


    sel = target_cfg.get('target_selection', {})
    mode = str(sel.get('mode', 'bbox')).lower()

    if mode == 'bbox':
        bbox = sel.get('bbox')
        if not bbox or len(bbox) != 4:
            raise ValueError("target_selection.bbox 必须是 [xmin, ymin, xmax, ymax]")
        mask_xy = (
            (xyz[:, 0] >= float(bbox[0])) & (xyz[:, 0] <= float(bbox[2])) &
            (xyz[:, 1] >= float(bbox[1])) & (xyz[:, 1] <= float(bbox[3]))
        )
    elif mode == 'center_radius':
        cx, cy = [float(v) for v in sel['center_xy']]
        r = float(sel['radius_m'])
        dxy2 = (xyz[:, 0] - cx) ** 2 + (xyz[:, 1] - cy) ** 2
        mask_xy = dxy2 <= r * r
    else:
        raise ValueError(f"不支持的 target_selection.mode: {mode}")

    z_min = float(sel.get('z_min', -np.inf))
    z_max = float(sel.get('z_max', np.inf))
    mask = mask_xy & (xyz[:, 2] >= z_min) & (xyz[:, 2] <= z_max)
    target_pts = xyz[mask]
    if target_pts.shape[0] < 20:
        raise ValueError(
            f"目标点云选出的点太少 ({int(target_pts.shape[0])} 个)，请检查 target_selection 参数"
        )

    z = target_pts[:, 2]
    z_sorted = np.sort(z)
    n = len(z_sorted)
    top_thresh = float(z_sorted[max(0, int(0.95 * n) - 1)])
    base_thresh = float(z_sorted[min(n - 1, int(0.05 * n))])
    top_pts = target_pts[z >= top_thresh]
    base_pts = target_pts[z <= base_thresh]
    top_anchor = top_pts.mean(axis=0)
    base_anchor = base_pts.mean(axis=0)

    if target_cfg.get('top_xyz') is not None:
        top_anchor = np.asarray(target_cfg['top_xyz'], dtype=np.float64)
    if target_cfg.get('base_xyz') is not None:
        base_anchor = np.asarray(target_cfg['base_xyz'], dtype=np.float64)

    return target_pts.astype(np.float64), top_anchor.astype(np.float64), base_anchor.astype(np.float64)


def project_points_to_scene(points_xyz: np.ndarray, scene_meta: dict):
    cam_xyz = np.asarray(scene_meta['camera_xyz'], dtype=np.float64)
    R_wc = np.asarray(scene_meta['world_to_camera_R'], dtype=np.float64)
    cm = scene_meta['camera_model']
    W = int(cm['image_width'])
    H = int(cm['image_height'])
    fov_h = float(cm['fov_h_deg'])
    fov_v = float(cm['fov_v_deg'])
    flip_u = bool(cm.get('flip_u', False))
    flip_v = bool(cm.get('flip_v', False))
    u, v, rng, offaxis, border, inside = project_world_points_pinhole(
        points_xyz, cam_xyz, R_wc, W, H, fov_h, fov_v, flip_u=flip_u, flip_v=flip_v
    )
    return u, v, rng, inside, W, H


def load_annotations(csv_path: Path) -> dict:
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    out = {}
    for _, row in df.iterrows():
        sid = str(row['scene_id'])
        status = str(row.get('annotation_status', 'pending')).strip()
        if status not in ALLOWED_STATUS:
            status = 'pending'
        out[sid] = {
            'annotation_status': status,
            'u_top': float(row.get('u_top', np.nan)) if pd.notna(row.get('u_top', np.nan)) else np.nan,
            'v_top': float(row.get('v_top', np.nan)) if pd.notna(row.get('v_top', np.nan)) else np.nan,
            'u_base': float(row.get('u_base', np.nan)) if pd.notna(row.get('u_base', np.nan)) else np.nan,
            'v_base': float(row.get('v_base', np.nan)) if pd.notna(row.get('v_base', np.nan)) else np.nan,
            'note': str(row.get('note', '')) if pd.notna(row.get('note', '')) else '',
        }
    return out


def save_annotations(annotations: dict, csv_path: Path, top_anchor: np.ndarray, base_anchor: np.ndarray) -> None:
    ensure_dir(csv_path.parent)
    rows = []
    for sid, ann in annotations.items():
        status = ann.get('annotation_status', 'pending')
        if status == 'pending':
            continue
        rows.append({
            'scene_id': sid,
            'annotation_status': status,
            'u_top': ann.get('u_top', np.nan),
            'v_top': ann.get('v_top', np.nan),
            'u_base': ann.get('u_base', np.nan),
            'v_base': ann.get('v_base', np.nan),
            'top_anchor_x': float(top_anchor[0]),
            'top_anchor_y': float(top_anchor[1]),
            'top_anchor_z': float(top_anchor[2]),
            'base_anchor_x': float(base_anchor[0]),
            'base_anchor_y': float(base_anchor[1]),
            'base_anchor_z': float(base_anchor[2]),
            'note': ann.get('note', ''),
        })
    df = pd.DataFrame(rows, columns=ANN_COLUMNS)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')


def infer_status(ann: dict) -> str:
    has_top = np.isfinite(ann.get('u_top', np.nan)) and np.isfinite(ann.get('v_top', np.nan))
    has_base = np.isfinite(ann.get('u_base', np.nan)) and np.isfinite(ann.get('v_base', np.nan))
    if ann.get('annotation_status') == 'none':
        return 'none'
    if has_top and has_base:
        return 'full'
    if has_top and not has_base:
        return 'top_only'
    if has_base and not has_top:
        return 'base_only'
    return 'pending'


class TargetAnnotator:
    def __init__(self, scenes, target_pts_xyz, top_anchor_xyz, base_anchor_xyz, out_csv,
                 margin_px_for_auto_skip=None, load_existing=True):


        self.scenes = scenes
        self.target_pts = target_pts_xyz
        self.top_anchor = top_anchor_xyz
        self.base_anchor = base_anchor_xyz
        self.out_csv = Path(out_csv)
        self.margin_px = margin_px_for_auto_skip

        self.annotations = load_annotations(self.out_csv) if load_existing else {}

        for sc in self.scenes:
            sid = sc['scene_id']
            if sid not in self.annotations:
                self.annotations[sid] = {
                    'annotation_status': 'pending',
                    'u_top': np.nan, 'v_top': np.nan,
                    'u_base': np.nan, 'v_base': np.nan,
                    'note': '',
                }


        self.mode = None
        self.current_idx = self._first_pending_idx()

        self.fig = None
        self.ax = None

    def _first_pending_idx(self) -> int:
        for i, sc in enumerate(self.scenes):
            if self.annotations[sc['scene_id']]['annotation_status'] == 'pending':
                return i
        return 0


    def run(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 11))
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self._render()
        plt.show()

        self._save()


    def _render(self):
        self.ax.clear()
        sc = self.scenes[self.current_idx]
        sid = sc['scene_id']
        meta = sc['scene_meta']
        preview_path = sc['preview_path']

        if preview_path and Path(preview_path).exists():
            try:
                img = np.asarray(Image.open(preview_path).convert('RGB'))
                self.ax.imshow(img)
            except Exception as e:
                self.ax.text(0.5, 0.5, f'预览图打不开：{e}', transform=self.ax.transAxes, ha='center')
        else:
            self.ax.text(0.5, 0.5, f'预览图不存在:\n{preview_path}', transform=self.ax.transAxes, ha='center')


        u_all, v_all, _, inside_all, W, H = project_points_to_scene(self.target_pts, meta)
        finite = np.isfinite(u_all) & np.isfinite(v_all)
        u_in = u_all[finite & inside_all]
        v_in = v_all[finite & inside_all]
        z_in = self.target_pts[finite & inside_all, 2]
        if u_in.size > 0:
            sc_plot = self.ax.scatter(u_in, v_in, c=z_in, s=6, cmap='viridis',
                                      alpha=0.75, edgecolors='none',
                                      label=f'projected target ({u_in.size} in / {int(np.count_nonzero(finite))} total)')

            if not hasattr(self, '_cbar') or self._cbar is None:
                self._cbar = self.fig.colorbar(sc_plot, ax=self.ax, shrink=0.6, label='point elevation (m)')
            else:
                try:
                    self._cbar.update_normal(sc_plot)
                except Exception:
                    pass


        u_top_pred, v_top_pred, _, inside_top, _, _ = project_points_to_scene(self.top_anchor[None, :], meta)
        u_bot_pred, v_bot_pred, _, inside_bot, _, _ = project_points_to_scene(self.base_anchor[None, :], meta)
        if np.isfinite(u_top_pred[0]) and np.isfinite(v_top_pred[0]):
            self.ax.plot(u_top_pred[0], v_top_pred[0], marker='^', markersize=18,
                         markerfacecolor='none', markeredgecolor='yellow', markeredgewidth=2,
                         label='predicted top anchor')
        if np.isfinite(u_bot_pred[0]) and np.isfinite(v_bot_pred[0]):
            self.ax.plot(u_bot_pred[0], v_bot_pred[0], marker='v', markersize=18,
                         markerfacecolor='none', markeredgecolor='orange', markeredgewidth=2,
                         label='predicted base anchor')


        ann = self.annotations[sid]
        if np.isfinite(ann.get('u_top', np.nan)) and np.isfinite(ann.get('v_top', np.nan)):
            self.ax.plot(ann['u_top'], ann['v_top'], marker='+', color='red',
                         markersize=22, markeredgewidth=3, label='marked TOP')
        if np.isfinite(ann.get('u_base', np.nan)) and np.isfinite(ann.get('v_base', np.nan)):
            self.ax.plot(ann['u_base'], ann['v_base'], marker='x', color='magenta',
                         markersize=20, markeredgewidth=3, label='marked BASE')


        self.ax.set_xlim(-50, W + 50)
        self.ax.set_ylim(H + 50, -50)
        self.ax.set_xlabel('u / pixel')
        self.ax.set_ylabel('v / pixel')
        self.ax.legend(loc='upper right', fontsize=8)

        n_annotated = sum(1 for a in self.annotations.values()
                          if a['annotation_status'] not in ('pending',))
        title = (
            f"[{self.current_idx + 1}/{len(self.scenes)}] {sid}  |  "
            f"status={ann['annotation_status']}  |  mode={self.mode or '—'}  |  "
            f"annotated {n_annotated}/{len(self.scenes)}"
        )
        self.ax.set_title(title)


        help_text = "keys: [t]=markTop  [b]=markBase  [n]=noTarget  [c]=clear  [space/→]=next  [←]=prev  [s]=saveExit  [q]=quit  [r]=redraw"
        self.fig.suptitle(help_text, fontsize=9, y=0.995)
        self.fig.tight_layout(rect=[0, 0, 1, 0.975])
        self.fig.canvas.draw_idle()


    def _on_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.button != 1:
            return
        if self.mode not in ('top', 'base'):
            print("先按 't' 或 'b' 选择标注模式（top / base）")
            return
        u, v = float(event.xdata), float(event.ydata)
        sid = self.scenes[self.current_idx]['scene_id']
        ann = self.annotations[sid]
        if self.mode == 'top':
            ann['u_top'] = u
            ann['v_top'] = v
        else:
            ann['u_base'] = u
            ann['v_base'] = v
        ann['annotation_status'] = infer_status(ann)
        self.mode = None
        self._save()
        self._render()

    def _on_key(self, event):
        key = event.key
        if key == 't':
            self.mode = 'top'
            self._render()
        elif key == 'b':
            self.mode = 'base'
            self._render()
        elif key == 'n':
            sid = self.scenes[self.current_idx]['scene_id']
            self.annotations[sid] = {
                'annotation_status': 'none',
                'u_top': np.nan, 'v_top': np.nan,
                'u_base': np.nan, 'v_base': np.nan,
                'note': self.annotations[sid].get('note', ''),
            }
            self._save()
            self._advance(1)
        elif key == 'c':
            sid = self.scenes[self.current_idx]['scene_id']
            self.annotations[sid] = {
                'annotation_status': 'pending',
                'u_top': np.nan, 'v_top': np.nan,
                'u_base': np.nan, 'v_base': np.nan,
                'note': '',
            }
            self._save()
            self._render()
        elif key in (' ', 'right'):
            self._advance(1)
        elif key == 'left':
            self._advance(-1)
        elif key == 'r':
            self._render()
        elif key == 's':
            self._save()
            plt.close(self.fig)
        elif key == 'q':
            self._save()
            plt.close(self.fig)

    def _advance(self, delta: int):
        new_idx = self.current_idx + delta
        if 0 <= new_idx < len(self.scenes):
            self.current_idx = new_idx
            self.mode = None
            self._render()
        else:
            print("已经到边界了")

    def _render_scene_to_png(self, sc: dict, out_png: Path) -> None:

        ensure_dir(out_png.parent)
        sid = sc['scene_id']
        meta = sc['scene_meta']
        preview_path = sc['preview_path']
        fig, ax = plt.subplots(figsize=(12, 11), dpi=160)
        if preview_path and Path(preview_path).exists():
            try:
                img = np.asarray(Image.open(preview_path).convert('RGB'))
                ax.imshow(img)
            except Exception as e:
                ax.text(0.5, 0.5, f'preview open failed: {e}', transform=ax.transAxes, ha='center')
        else:
            ax.text(0.5, 0.5, f'missing preview:\n{preview_path}', transform=ax.transAxes, ha='center')

        u_all, v_all, _, inside_all, W, H = project_points_to_scene(self.target_pts, meta)
        finite = np.isfinite(u_all) & np.isfinite(v_all)
        sel = finite & inside_all
        if np.any(sel):
            sc_plot = ax.scatter(
                u_all[sel], v_all[sel], c=self.target_pts[sel, 2], s=5,
                cmap='viridis', alpha=0.75, edgecolors='none',
                label=f'projected target ({int(np.count_nonzero(sel))})'
            )
            fig.colorbar(sc_plot, ax=ax, shrink=0.6, label='point elevation (m)')

        u_top_pred, v_top_pred, _, _, _, _ = project_points_to_scene(self.top_anchor[None, :], meta)
        u_bot_pred, v_bot_pred, _, _, _, _ = project_points_to_scene(self.base_anchor[None, :], meta)
        if np.isfinite(u_top_pred[0]) and np.isfinite(v_top_pred[0]):
            ax.plot(u_top_pred[0], v_top_pred[0], marker='^', markersize=18,
                    markerfacecolor='none', markeredgecolor='yellow', markeredgewidth=2,
                    label='predicted top anchor')
        if np.isfinite(u_bot_pred[0]) and np.isfinite(v_bot_pred[0]):
            ax.plot(u_bot_pred[0], v_bot_pred[0], marker='v', markersize=18,
                    markerfacecolor='none', markeredgecolor='orange', markeredgewidth=2,
                    label='predicted base anchor')

        ann = self.annotations.get(sid, {})
        if np.isfinite(ann.get('u_top', np.nan)) and np.isfinite(ann.get('v_top', np.nan)):
            ax.plot(ann['u_top'], ann['v_top'], marker='+', color='red',
                    markersize=22, markeredgewidth=3, label='marked TOP')
        if np.isfinite(ann.get('u_base', np.nan)) and np.isfinite(ann.get('v_base', np.nan)):
            ax.plot(ann['u_base'], ann['v_base'], marker='x', color='magenta',
                    markersize=20, markeredgewidth=3, label='marked BASE')

        ax.set_xlim(-50, W + 50)
        ax.set_ylim(H + 50, -50)
        ax.set_xlabel('u / pixel')
        ax.set_ylabel('v / pixel')
        ax.set_title(f"{sid} | status={ann.get('annotation_status', 'pending')}")
        ax.legend(loc='upper right', fontsize=8)
        fig.tight_layout()
        fig.savefig(out_png, bbox_inches='tight')
        plt.close(fig)

    def export_all_pngs(self, out_dir: Path) -> None:

        ensure_dir(out_dir)
        for sc in self.scenes:
            self._render_scene_to_png(sc, Path(out_dir) / f"{sc['scene_id']}.png")

    def _save(self):
        save_annotations(self.annotations, self.out_csv, self.top_anchor, self.base_anchor)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='主工程配置文件')
    ap.add_argument('--start-from', default=None, help='从指定 scene_id 开始，默认从第一个 pending 的开始')
    ap.add_argument('--only-in-image', action='store_true',
                    help='只显示目标顶点/目标基点预测投影至少一个落在图内的场景（加速过一遍）')
    ap.add_argument(
        '--fresh', action='store_true',
        help='Ignore an existing annotation CSV and start every scene as pending.',
    )
    ap.add_argument(
        '--export-previews', action='store_true',
        help='Also export the optional before/after annotation PNG sets.',
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    target_cfg = cfg.get('target_calibration', {})
    if not target_cfg:
        raise SystemExit(
            '主配置文件里没有 target_calibration 段，请参考 docs/target_calibration.md 或'
            ' config/project_config.yaml 文末示例补上。'
        )


    point_xyz_path = Path(cfg['outputs']['pointcloud_dir']) / 'point_xyz.npy'
    if not point_xyz_path.exists():
        raise SystemExit(f'找不到点云 npy：{point_xyz_path}，请先跑完步骤 04。')
    xyz = np.load(point_xyz_path, mmap_mode='r')
    target_pts, top_anchor, base_anchor = select_target_points(np.asarray(xyz), target_cfg)
    print(f'目标点云筛出 {target_pts.shape[0]} 个点')
    print(f'  目标顶点锚点 = {top_anchor}')
    print(f'  目标基点锚点 = {base_anchor}')


    scene_db_csv = Path(cfg['outputs']['scene_db_dir']) / 'scene_database.csv'
    if not scene_db_csv.exists():
        raise SystemExit(f'找不到 scene_database.csv：{scene_db_csv}，请先跑完步骤 03。')
    scene_db = pd.read_csv(scene_db_csv, encoding='utf-8-sig')

    scene_root = Path(cfg['outputs']['scene_db_dir']) / 'scenes'
    scenes = []
    for _, row in scene_db.iterrows():
        sid = str(row['scene_id'])
        meta_path = scene_root / sid / 'scene_meta.json'
        if not meta_path.exists():
            print(f'[skip] {sid}: 找不到 scene_meta.json')
            continue
        meta = load_json(meta_path)
        preview = meta.get('preview_rgb_path', '')
        entry = {'scene_id': sid, 'scene_meta': meta, 'preview_path': preview}

        if args.only_in_image:
            u_t, v_t, _, in_t, W, H = project_points_to_scene(top_anchor[None, :], meta)
            u_b, v_b, _, in_b, _, _ = project_points_to_scene(base_anchor[None, :], meta)
            margin = 100
            def _near_image(u_arr, v_arr):
                if not (np.isfinite(u_arr[0]) and np.isfinite(v_arr[0])):
                    return False
                return (-margin <= u_arr[0] <= W + margin) and (-margin <= v_arr[0] <= H + margin)
            if not (_near_image(u_t, v_t) or _near_image(u_b, v_b)):
                continue
        scenes.append(entry)

    if not scenes:
        raise SystemExit('没有找到任何可标注的场景。')


    start_idx = 0
    if args.start_from:
        for i, sc in enumerate(scenes):
            if sc['scene_id'] == args.start_from:
                start_idx = i
                break

    out_csv = Path(target_cfg.get('annotation_csv'))
    print(f'\n输出标注 CSV: {out_csv}')
    print(f'共 {len(scenes)} 个场景。\n')

    ann = TargetAnnotator(
        scenes, target_pts, top_anchor, base_anchor, out_csv,
        load_existing=not args.fresh,
    )
    pending_scene_ids = [
        sc['scene_id']
        for sc in scenes
        if infer_status(ann.annotations[sc['scene_id']]) == 'pending'
    ]
    if not pending_scene_ids:
        print(
            f'现有标注缓存已覆盖全部 {len(scenes)} 景，直接复用，不打开人工标注窗口。'
        )
        if args.export_previews:
            ann.export_all_pngs(out_csv.parent / "mark")
    else:
        print(f'仍有 {len(pending_scene_ids)} 景待标注，进入人工交互。')
        print('交互快捷键：')
        print('  t=标目标顶点  b=标目标基点  n=无目标  c=清空本景  空格/→=下一景  ←=上一景  s=保存退出  q=退出')
        print('黄色三角 = 目标顶点锚点投影，橙色三角 = 目标基点锚点投影（修正前）')
        print('红 + = 你手动标的目标顶点；粉 × = 你手动标的目标基点\n')
        if args.export_previews:
            ann.export_all_pngs(out_csv.parent / "un_mark")
        if args.start_from:
            ann.current_idx = start_idx
        ann.run()
        if args.export_previews:
            ann.export_all_pngs(out_csv.parent / "mark")


    final = load_annotations(out_csv)
    by_status = {}
    for a in final.values():
        by_status[a['annotation_status']] = by_status.get(a['annotation_status'], 0) + 1
    print('\n标注完成。按状态统计：')
    for k, v in sorted(by_status.items()):
        print(f'  {k:10s} : {v}')
    print(f'\n下一步：')
    print(f'  python scripts/06_solve_per_scene_correction.py --config {args.config}')


if __name__ == '__main__':
    main()
