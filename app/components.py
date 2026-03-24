"""Reusable UI components: molecule rendering, modals, search, layout panels."""

from __future__ import annotations

import base64
import html
import io
import json
import urllib.request
import urllib.parse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as st_components
from scipy.stats import percentileofscore

from app.charts import shap_bar, distribution, plotly_layout


# ---------------------------------------------------------------------------
# Molecule rendering
# ---------------------------------------------------------------------------

def molecule_image(smiles: str, size: tuple[int, int] = (360, 220)) -> bytes | None:
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        draw_opts = drawer.drawOptions()
        draw_opts.clearBackground = False
        draw_opts.padding = 0.18
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
        drawer.FinishDrawing()
        img_data = drawer.GetDrawingText()
        if isinstance(img_data, str):
            img_bytes = img_data.encode('latin-1')
        else:
            img_bytes = img_data
        return img_bytes
    except Exception:
        return None


def molecule_props(smiles: str) -> tuple[float | None, float | None]:
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None
        return float(Descriptors.MolWt(mol)), float(Crippen.MolLogP(mol))
    except Exception:
        return None, None


@st.cache_data(show_spinner=False)
def pubchem_name(smiles: str) -> str | None:
    """Look up a compound's preferred IUPAC name from PubChem via SMILES.

    Returns the name string on success, or None on any failure.
    Cached by Streamlit so each SMILES is only looked up once per session.
    """
    try:
        encoded = urllib.parse.quote(smiles, safe="")
        url = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
            f"{encoded}/property/IUPACName/JSON"
        )
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read())
        return data["PropertyTable"]["Properties"][0]["IUPACName"]
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def molecule_3d_block(smiles: str) -> str | None:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xF00D
        if AllChem.EmbedMolecule(mol, params) != 0:
            return None

        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=250)
        except Exception:
            AllChem.UFFOptimizeMolecule(mol, maxIters=250)

        return Chem.MolToMolBlock(Chem.RemoveHs(mol))
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def molecule_3d_image(smiles: str, size: tuple[int, int] = (980, 560)) -> bytes | None:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        import matplotlib.pyplot as plt

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xF00D
        if AllChem.EmbedMolecule(mol, params) != 0:
            return None

        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=250)
        except Exception:
            AllChem.UFFOptimizeMolecule(mol, maxIters=250)

        mol = Chem.RemoveHs(mol)
        conf = mol.GetConformer()
        coords = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())], dtype=float)
        if coords.size == 0:
            return None

        coords -= coords.mean(axis=0)
        max_span = float(np.max(np.ptp(coords, axis=0)))
        if max_span > 0:
            coords /= max_span

        z = coords[:, 2]
        z_min, z_max = float(z.min()), float(z.max())
        z_norm = (z - z_min) / (z_max - z_min + 1e-8)

        atom_colors = {
            1: "#dce6ef",
            6: "#2f3846",
            7: "#2d69b2",
            8: "#c94242",
            9: "#2e9f67",
            15: "#d08a2a",
            16: "#c2a32f",
            17: "#2e9f67",
        }

        w_in, h_in = size[0] / 100.0, size[1] / 100.0
        fig, ax = plt.subplots(figsize=(w_in, h_in), dpi=100)
        fig.patch.set_alpha(0.0)
        ax.set_facecolor((1, 1, 1, 0))
        ax.axis("off")
        ax.set_aspect("equal")

        bonds = []
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            depth = float((z_norm[i] + z_norm[j]) * 0.5)
            bonds.append((depth, i, j))
        bonds.sort(key=lambda x: x[0])

        for depth, i, j in bonds:
            alpha = 0.42 + 0.52 * depth
            lw = 1.8 + 1.1 * depth
            ax.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]], color=(0.43, 0.48, 0.55, alpha), linewidth=lw, solid_capstyle="round")

        atom_order = np.argsort(z_norm)
        for i in atom_order:
            atom = mol.GetAtomWithIdx(int(i))
            anum = int(atom.GetAtomicNum())
            c = atom_colors.get(anum, "#778190")
            size_pts = 150 + 170 * z_norm[i]
            ax.scatter(coords[i, 0], coords[i, 1], s=size_pts, c=c, edgecolors="#f7fbff", linewidths=0.75, alpha=0.96)

        pad = 0.35
        ax.set_xlim(float(coords[:, 0].min()) - pad, float(coords[:, 0].max()) + pad)
        ax.set_ylim(float(coords[:, 1].min()) - pad, float(coords[:, 1].max()) + pad)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", transparent=True, dpi=100, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
        return buf.getvalue()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def status(ad) -> tuple[str, str, int]:
    n_pass = int(ad.structural.pass_) + int(ad.y_range.pass_) + int(ad.local_error.pass_)
    if n_pass == 3:
        return "WITHIN TRAINING DOMAIN", "ad-ok", n_pass
    if n_pass == 2:
        return "BORDERLINE", "ad-mid", n_pass
    return "OUTSIDE TRAINING DOMAIN", "ad-bad", n_pass


def with_info(label: str, help_text: str | None = None) -> str:
    if not help_text:
        return label
    tip = json.dumps(help_text)
    return f"{label}<span class='info-dot' data-tip={tip} aria-label={tip}>i</span>"


def truncate_smiles(text: str, head: int = 38, tail: int = 17) -> str:
    if len(text) <= head + tail + 3:
        return text
    return f"{text[:head]}...{text[-tail:]}"


def render_chart_card(title: str, fig, note: str | None = None, info: str | None = None) -> None:
    with st.container(border=True):
        st.markdown(f"<div class='card-title card-title-row'>{with_info(title, info)}</div>", unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        if note:
            st.markdown(f"<div class='card-note'>{note}</div>", unsafe_allow_html=True)


def descriptor_table(result, predictor) -> pd.DataFrame:
    vals = pd.Series(result.feature_values, index=result.feature_names, dtype=float)
    stats = predictor.get_feature_stats().set_index("feature")

    rows = []
    for i, feat in enumerate(result.feature_names):
        train = predictor._X_cv_pre[:, i]
        pctl = float(percentileofscore(train, vals[feat], kind="mean"))
        mean = float(stats.loc[feat, "train_mean"]) if feat in stats.index else float(np.mean(train))
        std = float(stats.loc[feat, "train_std"]) if feat in stats.index else float(np.std(train))
        dev = abs(vals[feat] - mean) / std if std > 0 else abs(vals[feat] - mean)
        status_label = "OK" if 10 <= pctl <= 90 else "Outlying"

        rows.append(
            {
                "Descriptor": feat,
                "Query Value": float(vals[feat]),
                "Training Mean": mean,
                "Percentile": pctl,
                "Status": status_label,
                "Deviation": float(dev),
            }
        )

    df = pd.DataFrame(rows).sort_values("Deviation", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Molecule panels and modals
# ---------------------------------------------------------------------------

def render_example_molecule_panel(smiles: str, title: str, key_prefix: str) -> None:
    thumb = molecule_image(smiles, (260, 150))
    detail = molecule_image(smiles, (980, 560))

    if thumb is None or detail is None:
        st.markdown(
            f"<div style='border: 2px dashed #ff6b6b; padding: 12px; border-radius: 8px; text-align: center; color: #ff6b6b;'>"
            f"<strong>Rendering Failed:</strong> {title}<br><code>{truncate_smiles(smiles, 28, 12)}</code>"
            f"</div>",
            unsafe_allow_html=True
        )
        return

    thumb_b64 = base64.b64encode(thumb).decode("ascii")
    detail_b64 = base64.b64encode(detail).decode("ascii")
    st.markdown(
        f"""
<div style="display:flex; justify-content:center;">
    <img class="example-thumb-link mol-zoomable"
         src="data:image/png;base64,{thumb_b64}"
         data-detail="data:image/png;base64,{detail_b64}"
         data-title={json.dumps(title)}
         alt={json.dumps(title)} />
</div>
        """,
        unsafe_allow_html=True,
    )


def inject_mol_modal() -> None:
    """Inject a full-page molecule modal into the parent Streamlit document body."""
    st_components.html(
        """
<script>
(function () {
  var pd = window.parent.document;
  if (pd.getElementById('mol-global-modal')) return;

  var wrapper = pd.createElement('div');
  wrapper.id = 'mol-global-modal';
  wrapper.innerHTML = [
    '<div id="mol-backdrop" style="',
      'position:fixed;inset:0;z-index:99998;',
      'background:rgba(20,29,42,0.45);backdrop-filter:blur(3px);',
      'display:none;align-items:center;justify-content:center;">',
      '<div id="mol-card" style="',
        'position:relative;z-index:99999;',
        'background:#fff;border-radius:14px;',
        'box-shadow:0 24px 64px rgba(20,29,42,0.28);',
        'width:min(720px,92vw);max-height:88vh;overflow:auto;',
        'padding:0;">',
        '<div style="',
          'display:flex;align-items:center;justify-content:space-between;',
          'padding:12px 16px;border-bottom:1px solid #e3e9f1;">',
          '<span id="mol-modal-title" style="',
            'font-family:Space Grotesk,sans-serif;font-size:1.05rem;',
            'font-weight:700;color:#28354a;"></span>',
          '<button id="mol-close-btn" style="',
            'border:1px solid #d0d7e3;border-radius:9px;padding:4px 12px;',
            'background:#f7fafc;color:#5f6678;font-size:0.85rem;cursor:pointer;">',
            'Close</button>',
        '</div>',
        '<div style="padding:12px;">',
          '<img id="mol-modal-img" style="width:100%;height:auto;display:block;border-radius:8px;" />',
        '</div>',
      '</div>',
    '</div>'
  ].join('');
  pd.body.appendChild(wrapper);

  var backdrop = pd.getElementById('mol-backdrop');
  var img      = pd.getElementById('mol-modal-img');
  var titleEl  = pd.getElementById('mol-modal-title');
  var closeBtn = pd.getElementById('mol-close-btn');

  function open(src, title) {
    img.src = src;
    titleEl.textContent = title;
    backdrop.style.display = 'flex';
  }
  function close() {
    backdrop.style.display = 'none';
    img.src = '';
  }

  backdrop.addEventListener('click', function (e) {
    if (e.target === backdrop) close();
  });
  closeBtn.addEventListener('click', close);
  pd.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') close();
  });

  function wireThumbs() {
    pd.querySelectorAll('.mol-zoomable').forEach(function (el) {
      if (el._molWired) return;
      el._molWired = true;
      el.style.cursor = 'zoom-in';
      el.addEventListener('click', function () {
        open(el.dataset.detail, el.dataset.title || '');
      });
    });
  }
  wireThumbs();
  new window.parent.MutationObserver(wireThumbs)
    .observe(pd.documentElement, { childList: true, subtree: true });
})();
</script>
        """,
        height=0,
    )


# ---------------------------------------------------------------------------
# Search and datalists
# ---------------------------------------------------------------------------

def render_compound_search(label: str, search_key: str) -> None:
    st.markdown(f"<div class='label-overline'>{label}</div>", unsafe_allow_html=True)
    st.text_input(
        label,
        key=search_key,
        placeholder="Name or SMILES\u2026",
        label_visibility="collapsed",
    )


def inject_compound_datalists(cont_lookup: dict, surf_lookup: dict) -> None:
    cont_names = json.dumps(list(cont_lookup.keys()))
    surf_names = json.dumps(list(surf_lookup.keys()))
    st_components.html(
        f"""
<script>
(function () {{
  var pd = window.parent.document;

  function ensureDatalist(id, names) {{
    if (pd.getElementById(id)) return;
    var dl = pd.createElement('datalist');
    dl.id = id;
    names.forEach(function (n) {{
      var o = pd.createElement('option'); o.value = n; dl.appendChild(o);
    }});
    pd.body.appendChild(dl);
  }}

  function attachInput(el, listId) {{
    if (el._appListAttached) return;
    el._appListAttached = true;
    el.addEventListener('input', function () {{
      if (el.value.length > 0) {{
        el.setAttribute('list', listId);
      }} else {{
        el.removeAttribute('list');
      }}
    }});
    el.removeAttribute('list');
  }}

  function linkInputs() {{
    var c = pd.querySelector('.st-key-_cont_search input');
    var s = pd.querySelector('.st-key-_surf_search input');
    if (c) attachInput(c, 'app-cont-list');
    if (s) attachInput(s, 'app-surf-list');
  }}

  ensureDatalist('app-cont-list', {cont_names});
  ensureDatalist('app-surf-list', {surf_names});
  linkInputs();
  new window.parent.MutationObserver(linkInputs)
    .observe(pd.documentElement, {{ childList: true, subtree: true }});
}})();
</script>
        """,
        height=0,
    )


# ---------------------------------------------------------------------------
# Topbar
# ---------------------------------------------------------------------------

def render_topbar() -> None:
    st.markdown(
        """
<div class="topbar">
  <div style="display:flex; justify-content:space-between; align-items:center;">
        <div style="display:flex; align-items:center; gap:10px;">
            <svg width="30" height="30" viewBox="0 0 30 30" fill="none" xmlns="http://www.w3.org/2000/svg" style="flex-shrink:0;">
              <rect width="30" height="30" rx="8" fill="#e4f5f2"/>
              <circle cx="15" cy="9"  r="3.2" fill="#0c8a7f"/>
              <circle cx="22" cy="19" r="3.2" fill="#0c8a7f"/>
              <circle cx="8"  cy="19" r="3.2" fill="#0c8a7f"/>
              <line x1="15" y1="9"  x2="22" y2="19" stroke="#0c8a7f" stroke-width="1.8" stroke-linecap="round"/>
              <line x1="15" y1="9"  x2="8"  y2="19" stroke="#0c8a7f" stroke-width="1.8" stroke-linecap="round"/>
              <line x1="8"  y1="19" x2="22" y2="19" stroke="#0c8a7f" stroke-width="1.8" stroke-linecap="round"/>
            </svg>
            <div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:1.20rem; font-weight:700; color:#24445f;">MSR Estimator</div>
                <div class="small">Estimate molar solubilization ratios for contaminant–surfactant pairs</div>
            </div>
    </div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )
