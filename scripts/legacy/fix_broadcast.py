filepath = r"E:\HyperCAD_BEV_2026\scripts\run_experiments.py"
with open(filepath, "r", encoding="utf-8") as f:
    lines = f.readlines()

replacement = [
    '            # Riemannian: flatten metric field (200x200->40000) to match coords\n',
    '            xx = coords[..., 0]\n',
    '            zz = coords[..., 1]\n',
    '            g12_flat = metric["g_inv_12"].ravel()\n',
    '            h_flat   = height.ravel()\n',
    '            xx_adj = xx + 0.1 * g12_flat * h_flat.mean()\n',
    '            zz_adj = zz + 0.1 * g12_flat * h_flat.mean()\n',
    '            h = np.stack([xx_adj, zz_adj], axis=-1).astype(np.float32)\n',
]

new_lines = lines[:273] + replacement + lines[283:]
with open(filepath, "w", encoding="utf-8") as f:
    f.writelines(new_lines)
print("Fixed! Removed duplicate xx, added zz definition.")
