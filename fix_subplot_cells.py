"""
Fix subplot cells in both notebooks:
1. Replace hist_64 (averaged LBPH values) with real LBP pixel histogram via np.histogram
2. Add cv2.normalize for Cr/Cb channels in 6ch notebook for proper JET colormap display
3. Fix histogram bar rendering to use 0-255 range counts
"""
import json

# ============================================================
# NEW SUBPLOT CELL SOURCE - 6 KANAL (utuh)
# ============================================================
NEW_SUBPLOT_6CH = """\
# Sel 5: Tampilkan hasil capture step-by-step dalam subplot (otomatis pakai sample jika belum capture)
frame_uji = captured_frame if ("captured_frame" in globals() and captured_frame is not None) else cv2.imread(str(uji_real[0]))

crop, box = deteksi_wajah(frame_uji)
if crop is None:
    raise RuntimeError("Wajah tidak terdeteksi pada frame yang di-capture. Ulangi capture.")

x, y, w, h = box
img_haar = frame_uji.copy()
cv2.rectangle(img_haar, (x, y), (x + w, y + h), (0, 220, 0), 3)

face_128 = cv2.resize(crop, (128, 128))
ycrcb   = cv2.cvtColor(face_128, cv2.COLOR_BGR2YCrCb)
hsv     = cv2.cvtColor(face_128, cv2.COLOR_BGR2HSV)

# Normalisasi Cr & Cb agar rentang 0-255 kelihatan di colormap
cr_vis = cv2.normalize(ycrcb[:,:,1], None, 0, 255, cv2.NORM_MINMAX)
cb_vis = cv2.normalize(ycrcb[:,:,2], None, 0, 255, cv2.NORM_MINMAX)

gray_100 = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (100, 100))
eq_100   = cv2.equalizeHist(gray_100)

# Hitung peta LBP 98x98 untuk visualisasi
c_px    = eq_100[1:99, 1:99]
lbp_map = np.zeros_like(c_px, dtype=np.uint8)
for dy, dx, bit in [(-1,-1,0),(-1,0,1),(-1,1,2),(0,1,3),(1,1,4),(1,0,5),(1,-1,6),(0,-1,7)]:
    lbp_map |= ((eq_100[1+dy:99+dy, 1+dx:99+dx] >= c_px).astype(np.uint8) << bit)

# Histogram NYATA dari peta LBP (distribusi 0-255)
hist_vals, _ = np.histogram(lbp_map, bins=64, range=(0, 256))

res = uji_gambar(frame_uji)

# Buat gambar grafik histogram pakai OpenCV
hist_canvas = np.full((220, 270, 3), 245, dtype=np.uint8)
cv2.putText(hist_canvas, "LBP Histogram", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60,60,60), 1)
mx = float(hist_vals.max()) if hist_vals.max() > 0 else 1.0
bar_color = (50, 170, 50) if res["is_real"] else (30, 30, 200)
for i, val in enumerate(hist_vals):
    bh = int((float(val) / mx) * 155)
    cv2.rectangle(hist_canvas, (4 + i*4, 195 - bh), (6 + i*4, 196), bar_color, -1)
cv2.line(hist_canvas, (4, 196), (262, 196), (100,100,100), 1)
lbl = f"{'REAL' if res['is_real'] else 'SPOOF'} | {res['nama']}"
lbl2 = f"Chi2={res['chi2']:.2f} | {'AKSES OK' if res['granted'] else 'DITOLAK'}"
clr = (20, 130, 20) if res["granted"] else (0, 0, 200)
cv2.putText(hist_canvas, lbl,  (5, 212), cv2.FONT_HERSHEY_SIMPLEX, 0.38, clr, 1)
cv2.putText(hist_canvas, lbl2, (5, 219), cv2.FONT_HERSHEY_SIMPLEX, 0.33, clr, 1)

langkah = [
    ("1. Input + Haar",        img_haar,       None),
    ("2. ROI Crop (BGR)",      face_128,        None),
    ("3a. Kanal Y (YCrCb)",   ycrcb[:,:,0],   "gray"),
    ("3b. Kanal Cr (YCrCb)",  cr_vis,          cv2.COLORMAP_JET),
    ("3c. Kanal Cb (YCrCb)",  cb_vis,          cv2.COLORMAP_OCEAN),
    ("3d. Kanal H (HSV)",     hsv[:,:,0],      cv2.COLORMAP_HSV),
    ("3e. Kanal S (HSV)",     hsv[:,:,1],      cv2.COLORMAP_MAGMA),
    ("3f. Kanal V (HSV)",     hsv[:,:,2],      "gray"),
    ("4. Equalize 100x100",   eq_100,          "gray"),
    ("5. Peta Tekstur LBP",   lbp_map,         "gray"),
    ("6. LBP Histogram",      hist_canvas,     None),
]

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    fig.suptitle("Pipeline 6-Kanal: YCrCb + HSV (Y,Cr,Cb,H,S,V)", fontsize=13, fontweight="bold")
    for ax, (judul, gbr, cmap) in zip(axes.flat, langkah):
        if isinstance(cmap, int):        # OpenCV colormap int -> apply
            gbr_c = cv2.applyColorMap(gbr, cmap)
            ax.imshow(cv2.cvtColor(gbr_c, cv2.COLOR_BGR2RGB))
        elif gbr.ndim == 2:
            ax.imshow(gbr, cmap="gray")
        elif gbr.ndim == 3 and gbr.shape[2] == 3:
            ax.imshow(cv2.cvtColor(gbr, cv2.COLOR_BGR2RGB))
        ax.set_title(judul, fontsize=9, fontweight="bold")
        ax.axis("off")
    for ax in axes.flat[len(langkah):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("hasil_subplot_utuh.png", dpi=110, bbox_inches="tight")
    plt.show()
    print("Subplot tersimpan: hasil_subplot_utuh.png")
except ImportError:
    # Fallback: susun dengan OpenCV tile
    TILE = 220
    rows, cols = 3, 4
    canvas = np.full((rows * TILE, cols * TILE, 3), 230, dtype=np.uint8)
    for idx, (judul, gbr, cmap) in enumerate(langkah):
        r, c = divmod(idx, cols)
        # Konversi ke BGR 3ch
        if isinstance(cmap, int):
            tile = cv2.applyColorMap(gbr, cmap)
        elif gbr.ndim == 2:
            tile = cv2.cvtColor(gbr, cv2.COLOR_GRAY2BGR)
        else:
            tile = gbr.copy()
        tile = cv2.resize(tile, (TILE - 10, TILE - 36))
        y0 = r * TILE + 30; x0 = c * TILE + 5
        canvas[y0:y0 + TILE - 36, x0:x0 + TILE - 10] = tile
        cv2.putText(canvas, judul, (x0, r * TILE + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20,20,20), 1)
    cv2.imwrite("hasil_subplot_utuh.png", canvas)
    from IPython.display import Image, display
    display(Image("hasil_subplot_utuh.png"))
    print("Subplot tersimpan: hasil_subplot_utuh.png (OpenCV fallback)")
"""

# ============================================================
# NEW SUBPLOT CELL SOURCE - 4 KANAL (tanpa Y dan V)
# ============================================================
NEW_SUBPLOT_4CH = """\
# Sel 5: Tampilkan hasil capture step-by-step dalam subplot (otomatis pakai sample jika belum capture)
frame_uji = captured_frame if ("captured_frame" in globals() and captured_frame is not None) else cv2.imread(str(uji_real[0]))

crop, box = deteksi_wajah(frame_uji)
if crop is None:
    raise RuntimeError("Wajah tidak terdeteksi pada frame yang di-capture. Ulangi capture.")

x, y, w, h = box
img_haar = frame_uji.copy()
cv2.rectangle(img_haar, (x, y), (x + w, y + h), (0, 220, 0), 3)

face_128 = cv2.resize(crop, (128, 128))
ycrcb   = cv2.cvtColor(face_128, cv2.COLOR_BGR2YCrCb)
hsv     = cv2.cvtColor(face_128, cv2.COLOR_BGR2HSV)

# Normalisasi Cr & Cb agar rentang 0-255 kelihatan di colormap
cr_vis = cv2.normalize(ycrcb[:,:,1], None, 0, 255, cv2.NORM_MINMAX)
cb_vis = cv2.normalize(ycrcb[:,:,2], None, 0, 255, cv2.NORM_MINMAX)

gray_100 = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (100, 100))
eq_100   = cv2.equalizeHist(gray_100)

# Hitung peta LBP 98x98 untuk visualisasi
c_px    = eq_100[1:99, 1:99]
lbp_map = np.zeros_like(c_px, dtype=np.uint8)
for dy, dx, bit in [(-1,-1,0),(-1,0,1),(-1,1,2),(0,1,3),(1,1,4),(1,0,5),(1,-1,6),(0,-1,7)]:
    lbp_map |= ((eq_100[1+dy:99+dy, 1+dx:99+dx] >= c_px).astype(np.uint8) << bit)

# Histogram NYATA dari peta LBP (distribusi 0-255)
hist_vals, _ = np.histogram(lbp_map, bins=64, range=(0, 256))

res = uji_gambar(frame_uji)

# Buat gambar grafik histogram pakai OpenCV
hist_canvas = np.full((220, 270, 3), 245, dtype=np.uint8)
cv2.putText(hist_canvas, "LBP Histogram", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60,60,60), 1)
mx = float(hist_vals.max()) if hist_vals.max() > 0 else 1.0
bar_color = (50, 170, 50) if res["is_real"] else (30, 30, 200)
for i, val in enumerate(hist_vals):
    bh = int((float(val) / mx) * 155)
    cv2.rectangle(hist_canvas, (4 + i*4, 195 - bh), (6 + i*4, 196), bar_color, -1)
cv2.line(hist_canvas, (4, 196), (262, 196), (100,100,100), 1)
lbl = f"{'REAL' if res['is_real'] else 'SPOOF'} | {res['nama']}"
lbl2 = f"Chi2={res['chi2']:.2f} | {'AKSES OK' if res['granted'] else 'DITOLAK'}"
clr = (20, 130, 20) if res["granted"] else (0, 0, 200)
cv2.putText(hist_canvas, lbl,  (5, 212), cv2.FONT_HERSHEY_SIMPLEX, 0.38, clr, 1)
cv2.putText(hist_canvas, lbl2, (5, 219), cv2.FONT_HERSHEY_SIMPLEX, 0.33, clr, 1)

langkah = [
    ("1. Input + Haar",       img_haar,      None),
    ("2. ROI Crop (BGR)",     face_128,       None),
    ("3a. Kanal Cr (YCrCb)", cr_vis,         cv2.COLORMAP_JET),
    ("3b. Kanal Cb (YCrCb)", cb_vis,         cv2.COLORMAP_OCEAN),
    ("3c. Kanal H (HSV)",    hsv[:,:,0],     cv2.COLORMAP_HSV),
    ("3d. Kanal S (HSV)",    hsv[:,:,1],     cv2.COLORMAP_MAGMA),
    ("4. Equalize 100x100",  eq_100,         "gray"),
    ("5. Peta Tekstur LBP",  lbp_map,        "gray"),
    ("6. LBP Histogram",     hist_canvas,    None),
]

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.5))
    fig.suptitle("Pipeline 4-Kanal: Krominans Saja (Cr,Cb,H,S)", fontsize=13, fontweight="bold")
    for ax, (judul, gbr, cmap) in zip(axes.flat, langkah):
        if isinstance(cmap, int):
            gbr_c = cv2.applyColorMap(gbr, cmap)
            ax.imshow(cv2.cvtColor(gbr_c, cv2.COLOR_BGR2RGB))
        elif gbr.ndim == 2:
            ax.imshow(gbr, cmap="gray")
        elif gbr.ndim == 3 and gbr.shape[2] == 3:
            ax.imshow(cv2.cvtColor(gbr, cv2.COLOR_BGR2RGB))
        ax.set_title(judul, fontsize=9, fontweight="bold")
        ax.axis("off")
    for ax in axes.flat[len(langkah):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("hasil_subplot_tanpa_yv.png", dpi=110, bbox_inches="tight")
    plt.show()
    print("Subplot tersimpan: hasil_subplot_tanpa_yv.png")
except ImportError:
    # Fallback: susun dengan OpenCV tile
    TILE = 220
    rows, cols = 2, 5
    canvas = np.full((rows * TILE, cols * TILE, 3), 230, dtype=np.uint8)
    for idx, (judul, gbr, cmap) in enumerate(langkah):
        r, c = divmod(idx, cols)
        if isinstance(cmap, int):
            tile = cv2.applyColorMap(gbr, cmap)
        elif gbr.ndim == 2:
            tile = cv2.cvtColor(gbr, cv2.COLOR_GRAY2BGR)
        else:
            tile = gbr.copy()
        tile = cv2.resize(tile, (TILE - 10, TILE - 36))
        y0 = r * TILE + 30; x0 = c * TILE + 5
        canvas[y0:y0 + TILE - 36, x0:x0 + TILE - 10] = tile
        cv2.putText(canvas, judul, (x0, r * TILE + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20,20,20), 1)
    cv2.imwrite("hasil_subplot_tanpa_yv.png", canvas)
    from IPython.display import Image, display
    display(Image("hasil_subplot_tanpa_yv.png"))
    print("Subplot tersimpan: hasil_subplot_tanpa_yv.png (OpenCV fallback)")
"""

def fix_notebook(nb_path, cell_id, new_source):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    found = False
    for cell in nb['cells']:
        if cell.get('id', '') == cell_id:
            # Store as list of lines (each ending with \n except last)
            lines = new_source.splitlines(keepends=True)
            cell['source'] = lines
            cell['outputs'] = []
            cell['execution_count'] = None
            found = True
            print(f"  Fixed cell: {cell_id}")
            break
    if not found:
        print(f"  ERROR: cell {cell_id} not found!")
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"  Saved: {nb_path}")

print("Fixing 6-channel notebook...")
fix_notebook(
    "cv_liveness_ycrcb_hsv_utuh.ipynb",
    "cell-subplot-6ch",
    NEW_SUBPLOT_6CH
)

print("Fixing 4-channel notebook...")
fix_notebook(
    "cv_liveness_tanpa_yv.ipynb",
    "cell-subplot-4ch",
    NEW_SUBPLOT_4CH
)

print("\nDone! Verifying cell source lines...")
for nb_file, cid in [("cv_liveness_ycrcb_hsv_utuh.ipynb","cell-subplot-6ch"),
                      ("cv_liveness_tanpa_yv.ipynb","cell-subplot-4ch")]:
    with open(nb_file, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    for cell in nb['cells']:
        if cell.get('id','') == cid:
            src = ''.join(cell['source'])
            print(f"\n{nb_file} - {cid}: {len(cell['source'])} lines")
            # Check key fixes
            print("  np.histogram:", "np.histogram(lbp_map" in src)
            print("  cv2.normalize Cr:", "cr_vis = cv2.normalize" in src)
            print("  bar_color:", "bar_color" in src)
            print("  COLORMAP handled:", "isinstance(cmap, int)" in src)
            break
