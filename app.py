import json
import random
import re
import time
from pathlib import Path
from typing import List, Dict, Union

import pandas as pd
from collections import defaultdict
import streamlit as st

# ---------------------------
# Text helpers
# ---------------------------

def bold_keywords(text: str, keywords: list[str]) -> str:
    """Return text with keywords wrapped in ** ** (whole-word, case-insensitive)."""
    if not keywords:
        return text
    out = text
    for kw in keywords:
        kw = kw.strip()
        if not kw:
            continue
        out = re.sub(rf"\b({re.escape(kw)})\b", r"**\1**", out, flags=re.IGNORECASE)
    return out


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

# ---------------------------
# Deck loading
# ---------------------------

def rows_to_deck(df: pd.DataFrame) -> dict:
    required = {"topic", "title", "acronym", "letter", "text"}
    missing = required - set(map(str.lower, df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    # Normalize column names to lowercase
    cols = {c.lower(): c for c in df.columns}
    df2 = df.rename(columns=cols)

    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    imgs: dict[tuple[str, str, str], str] = {}

    for _, r in df2.iterrows():
        topic = str(r["topic"]).strip()
        title = str(r["title"]).strip()
        acronym = str(r["acronym"]).strip()
        letter = str(r["letter"]).strip()
        text = str(r["text"]).strip()

        bold_raw = str(r.get("bold_words", "") or "").strip()
        bold_list = [w.strip() for w in bold_raw.split(",") if w.strip()]

        raw_img = r.get("img", "")
        if pd.isna(raw_img):
            raw_img = ""
        img = str(raw_img).strip()

        # ------------------ IMAGE NORMALISATION ------------------
        # Accept values like "4", "1.jpg", "img/1.jpg", "./img/1.jpg", or URL
        if img:
            img = img.replace("\\", "/")  # Windows → POSIX
            # If it's a pure number like "4", treat as stem and let resolver find extension
            if img.isdigit():
                img = f"img/{img}"  # extension resolved later
            elif not img.lower().startswith(("http://", "https://")):
                # Just a bare filename => assume under img/
                if "/" not in img:
                    img = f"img/{img}"
                if img.startswith("./"):
                    img = img[2:]

        key = (topic, title, acronym)

        # Keep the FIRST non-empty image we encounter for this (topic,title,acronym)
        if img and key not in imgs:
            imgs[key] = img

        groups[key].append(
            {
                "letter": letter,
                "text": bold_keywords(text, bold_list),
                "plain_text": text,
            }
        )

    cards = []
    for (topic, title, acronym), items in groups.items():
        order = {ch: i for i, ch in enumerate(acronym)}
        items.sort(key=lambda it: order.get(it["letter"], 999))
        cards.append(
            {
                "topic": topic,
                "title": title,
                "acronym": acronym,
                # populate img correctly (previously always empty)
                "img": imgs.get((topic, title, acronym), ""),
                "items": items,
            }
        )

    return {"name": "Audit Exam Rote Learning", "source": "uploaded", "cards": cards}


def load_any_deck(upload: Union[Path, str, object]) -> dict:
    """Accept Path/str (local file) or Streamlit UploadedFile, return deck dict."""
    if isinstance(upload, (str, Path)):
        path = Path(upload)
        suffix = path.suffix.lower()
        if suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        elif suffix == ".csv":
            df = pd.read_csv(path)
            return rows_to_deck(df)
        elif suffix in (".xlsx", ".xls"):
            try:
                df = pd.read_excel(path, engine="openpyxl")
            except ImportError as e:
                raise RuntimeError(
                    "Excel support requires openpyxl. Add `openpyxl>=3.1` to requirements.txt."
                ) from e
            return rows_to_deck(df)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    # UploadedFile branch
    name = getattr(upload, "name", "").lower()
    if name.endswith(".json"):
        return json.load(upload)
    elif name.endswith(".csv"):
        df = pd.read_csv(upload)
        return rows_to_deck(df)
    elif name.endswith((".xlsx", ".xls")):
        try:
            df = pd.read_excel(upload, engine="openpyxl")
        except ImportError as e:
            raise RuntimeError(
                "Excel support requires openpyxl. Add `openpyxl>=3.1` to requirements.txt."
            ) from e
        return rows_to_deck(df)
    else:
        raise ValueError("Unsupported file type. Upload .json, .csv, or .xlsx.")

# ---------------------------
# Image helpers
# ---------------------------

def resolve_local_image_path(img: str) -> str | None:
    """Best-effort resolver for local images.

    Handles: bare numbers ("4"), filenames with or without folder, and cases
    where the provided extension doesn't actually exist on disk (e.g. sheet has
    "4.jpg" but file is "4.png"). Tries sibling extensions with same stem.
    """
    if not img:
        return None

    def _exists(p: Path) -> str | None:
        return str(p) if p.exists() else None

    # Normalise slashes
    img = img.replace("\", "/").lstrip("./")

    repo_root = Path.cwd()
    img_dir = repo_root / "img"
    candidates: list[Path] = []

    # If it's just a bare name like "4" or "4.jpg"
    name = Path(img).name
    base_stem = Path(name).stem
    has_ext = Path(name).suffix != ""

    # Primary candidates: honour the given path as-is and common fallbacks
    p = Path(img)
    candidates.extend([
        repo_root / p,
        img_dir / name,
        repo_root / name,
    ])

    # If not found yet, try sibling extensions for same stem in /img and repo root
    exts = [".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"]

    # If no extension OR the exact file doesn't exist, try other extensions
    for folder in [img_dir, repo_root]:
        for ext in exts:
            cand = folder / f"{base_stem}{ext}"
            candidates.append(cand)

    # Deduplicate while preserving order
    seen = set()
    ordered = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            ordered.append(c)

    for c in ordered:
        res = _exists(c)
        if res:
            return res

    return None

    def _exists(p: Path) -> str | None:
        return str(p) if p.exists() else None

    p = Path(img)
    if p.exists():
        return str(p)

    repo_root = Path.cwd()
    candidates = [
        repo_root / img,
        repo_root / "img" / Path(img).name,
        repo_root / Path(img.lstrip("./")),
    ]
    for c in candidates:
        res = _exists(c)
        if res:
            return res

    # If no suffix given, try common extensions
    stem = Path(img).stem
    if "." not in Path(img).name:  # rough check for "no extension"
        for folder in [repo_root / "img", repo_root]:
            for ext in [".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"]:
                candidate = folder / f"{stem}{ext}"
                res = _exists(candidate)
                if res:
                    return res

    return None


def show_card_image(card: dict, debug: bool = False):
    raw = (card.get("img") or "").strip()
    if debug:
        st.caption(f"Raw img from card: '{raw or '(empty)'}'")

    if not raw:
        if debug:
            st.caption("No `img` specified for this card.")
        return

    try:
        # Remote URL
        if raw.lower().startswith(("http://", "https://")):
            if debug:
                st.caption(f"Loading remote image: {raw}")
            st.image(raw, caption=None, use_container_width=True)
            return

        # Local path resolution
        local = resolve_local_image_path(raw)
        if local:
            if debug:
                st.caption(f"Loading local image: {local}")
            st.image(local, caption=None, use_container_width=True)
        else:
            if debug:
                st.error(f"Image not found after resolution attempts: {raw}")

    except Exception as e:
        if debug:
            st.error(f"Image load error for '{raw}': {e}")

# ---------------------------
# Exercise renderers
# ---------------------------

def verdict_icon(ok: bool) -> str:
    return "✅" if ok else "❌"


def exercise_heading_only(card: Dict):
    st.subheader(card["title"])
    st.caption(f"Topic: {card['topic']}")

    acronym_guess = st.text_input("Acronym", key="h_only_acronym")
    ac_ok = normalize(acronym_guess) == normalize(card["acronym"])
    st.write(f"Acronym: {verdict_icon(ac_ok)}")

    answers_ok = []
    for i, it in enumerate(card["items"]):
        ans = st.text_input(f"{it['letter']} →", key=f"h_only_item_{i}")
        answers_ok.append(normalize(ans) == normalize(it["text"]))

    if st.button("Check answers", type="primary"):
        st.write("---")
        st.write(f"Acronym: {verdict_icon(ac_ok)}")
        for i, it in enumerate(card["items"]):
            st.write(
                f"{it['letter']} → {verdict_icon(answers_ok[i])}  "
                f"**Your:** {st.session_state.get(f'h_only_item_{i}','')}  "
                f"**Correct:** {it['text']}"
            )
        st.success("All correct! 🎉" if ac_ok and all(answers_ok) else "Keep going!")


def exercise_heading_plus_acronym(card: Dict):
    st.subheader(card["title"])
    st.caption(f"Topic: {card['topic']}")
    st.markdown(f"**Acronym:** `{card['acronym']}`")

    answers_ok = []
    for i, it in enumerate(card["items"]):
        ans = st.text_input(f"{it['letter']} →", key=f"h_ac_item_{i}")
        answers_ok.append(normalize(ans) == normalize(it["text"]))

    if st.button("Check answers", type="primary"):
        st.write("---")
        for i, it in enumerate(card["items"]):
            st.write(
                f"{it['letter']} → {verdict_icon(answers_ok[i])}  "
                f"**Your:** {st.session_state.get(f'h_ac_item_{i}','')}  "
                f"**Correct:** {it['text']}"
            )
        st.success("All correct! 🎉" if all(answers_ok) else "Keep going!")


def exercise_acronym_only(card: Dict):
    st.markdown(f"### Acronym: `{card['acronym']}`")
    st.caption(f"Topic: {card['topic']}")

    title_guess = st.text_input("Card heading/title", key="a_only_title")
    title_ok = normalize(title_guess) == normalize(card["title"])

    answers_ok = []
    for i, it in enumerate(card["items"]):
        ans = st.text_input(f"{it['letter']} →", key=f"a_only_item_{i}")
        answers_ok.append(normalize(ans) == normalize(it["text"]))

    if st.button("Check answers", type="primary"):
        st.write("---")
        st.write(
            f"Title: {verdict_icon(title_ok)}  **Your:** {title_guess}  **Correct:** {card['title']}"
        )
        for i, it in enumerate(card["items"]):
            st.write(
                f"{it['letter']} → {verdict_icon(answers_ok[i])}  "
                f"**Your:** {st.session_state.get(f'a_only_item_{i}','')}  "
                f"**Correct:** {it['text']}"
            )
        st.success("All correct! 🎉" if title_ok and all(answers_ok) else "Keep going!")


def exercise_letters_blank(card: Dict):
    st.markdown(
        f"#### Letters: `{''.join([it['letter'] for it in card['items']])}`"
    )
    st.caption(f"Topic: {card['topic']} — Card: {card['title']}")

    answers_ok = []
    for i, it in enumerate(card["items"]):
        ans = st.text_input(f"{it['letter']} →", key=f"letters_blank_{i}")
        answers_ok.append(normalize(ans) == normalize(it["text"]))

    if st.button("Check answers", type="primary"):
        st.write("---")
        for i, it in enumerate(card["items"]):
            st.write(
                f"{it['letter']} → {verdict_icon(answers_ok[i])}  "
                f"**Your:** {st.session_state.get(f'letters_blank_{i}','')}  "
                f"**Correct:** {it['text']}"
            )
        st.success("All correct! 🎉" if all(answers_ok) else "Keep going!")


def exercise_letters_some_prefilled(card: Dict, prefill_ratio: float = 0.4):
    st.markdown(
        f"#### Letters: `{''.join([it['letter'] for it in card['items']])}`"
    )
    st.caption(f"Topic: {card['topic']} — Card: {card['title']}")

    n = len(card["items"])
    k = max(1, int(n * prefill_ratio))
    rng = random.Random(st.session_state.get("seed", 0))
    locked_idx = set(rng.sample(range(n), k))

    answers_ok = []
    for i, it in enumerate(card["items"]):
        if i in locked_idx:
            st.text_input(
                f"{it['letter']} → (prefilled)",
                value=it["text"],
                disabled=True,
                key=f"pref_{i}",
            )
            answers_ok.append(True)
        else:
            ans = st.text_input(f"{it['letter']} →", key=f"letters_some_{i}")
            answers_ok.append(normalize(ans) == normalize(it["text"]))

    if st.button("Check answers", type="primary"):
        st.write("---")
        for i, it in enumerate(card["items"]):
            if i in locked_idx:
                st.write(f"{it['letter']} → ✅ (prefilled) **{it['text']}**")
            else:
                st.write(
                    f"{it['letter']} → {verdict_icon(answers_ok[i])}  "
                    f"**Your:** {st.session_state.get(f'letters_some_{i}','')}  "
                    f"**Correct:** {it['text']}"
                )
        st.success("All correct! 🎉" if all(answers_ok) else "Keep going!")


def exercise_missing_keywords(card: Dict, difficulty: str = "medium"):
    st.caption(f"Topic: {card['topic']} — Card: {card['title']}")
    answers_ok = []
    for i, it in enumerate(card["items"]):
        masked = mask_text(it["text"], difficulty=difficulty)
        st.write(f"{it['letter']}: `{masked}`")
        ans = st.text_input("Full text:", key=f"mask_{i}")
        answers_ok.append(normalize(ans) == normalize(it["text"]))

    if st.button("Check answers", type="primary"):
        st.write("---")
        for i, it in enumerate(card["items"]):
            st.write(
                f"{it['letter']} → {verdict_icon(answers_ok[i])}  "
                f"**Your:** {st.session_state.get(f'mask_{i}','')}  "
                f"**Correct:** {it['text']}"
            )
        st.success("All correct! 🎉" if all(answers_ok) else "Keep going!")


# ---------------------------
# UI & Layout
# ---------------------------

st.set_page_config(page_title="Rote Cards", page_icon="🗂️", layout="centered")

st.title("🗂️ Rote Learning Cards")
with st.sidebar:
    st.markdown("### Deck")
    deck_file = st.file_uploader(
        "Upload a deck (.json, .csv, .xlsx)", type=["json", "csv", "xlsx", "xls"]
    )

    if deck_file:
        try:
            deck = load_any_deck(deck_file)
            st.success(f"Loaded {len(deck['cards'])} cards from {deck_file.name}")
        except Exception as e:
            st.error(f"Failed to load deck: {e}")
            st.stop()
    else:
        default_path = Path("deck.json")
        if not default_path.exists():
            st.info("Upload a deck or add a `deck.json` next to app.py.")
            st.stop()
        deck = load_any_deck(default_path)

    topics = sorted({c["topic"] for c in deck["cards"]})
    topic = st.selectbox("Filter by topic", options=["(All)"] + topics)
    filtered = [c for c in deck["cards"] if topic == "(All)" or c["topic"] == topic]

    if "seed" not in st.session_state:
        st.session_state.seed = int(time.time())

    st.markdown("---")
    st.markdown("### Exercise Type")
    mode = st.selectbox(
        "Choose a practice mode",
        options=[
            "Card heading only",
            "Card heading + acronym",
            "Acronym only",
            "Letters down the side (all blank)",
            "Letters down the side (some prefilled)",
            "Missing key words",
        ],
    )

    if mode == "Missing key words":
        difficulty = st.select_slider(
            "Masking difficulty", options=["easy", "medium", "hard"], value="medium"
        )
    else:
        difficulty = None

    if mode == "Letters down the side (some prefilled)":
        prefill_ratio = st.slider(
            "Prefill ratio", min_value=0.2, max_value=0.8, value=0.4, step=0.1
        )
    else:
        prefill_ratio = 0.4

    show_img_debug = st.checkbox("Debug images", value=False)
    st.markdown("---")
    if st.button("🎲 New Card"):
        st.session_state.seed = int(time.time())

# Pick a card deterministically from seed
if not filtered:
    st.warning("No cards match the current filter.")
    st.stop()

idx = random.Random(st.session_state.seed).randint(0, len(filtered) - 1)
card = filtered[idx]

# Title banner
st.markdown(
    f"""
    <div style="background:#166534; color:#fff; padding:12px 16px; border-radius:8px; font-weight:600; letter-spacing:0.2px;">{card['title']}</div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([2, 1], vertical_alignment="top")

with right:
    # Only render once; show debug details if enabled
    show_card_image(card, debug=show_img_debug)

with left:
    if mode == "Card heading only":
        exercise_heading_only(card)
    elif mode == "Card heading + acronym":
        exercise_heading_plus_acronym(card)
    elif mode == "Acronym only":
        exercise_acronym_only(card)
    elif mode == "Letters down the side (all blank)":
        exercise_letters_blank(card)
    elif mode == "Letters down the side (some prefilled)":
        exercise_letters_some_prefilled(card, prefill_ratio=prefill_ratio)
    elif mode == "Missing key words":
        exercise_missing_keywords(card, difficulty=difficulty or "medium")

st.markdown("---")
with st.expander("Show full answer"):
    st.markdown(f"**Title:** {card['title']}  \n**Acronym:** `{card['acronym']}`")
    st.write("**Items:**")
    for it in card["items"]:
        st.write(f"- **{it['letter']}** → {it['text']}")

