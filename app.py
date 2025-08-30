import random
import re
import time
from pathlib import Path
from typing import Dict, Union

import pandas as pd
from collections import defaultdict
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
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

        raw_img = r.get("img", "")
        img = "" if pd.isna(raw_img) else str(raw_img).strip().replace("\\", "/")

        key = (topic, title, acronym)
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
    """Excel-only deck loader.

    Accepts either a filesystem path to an Excel file or a Streamlit UploadedFile.
    We intentionally do NOT support JSON/CSV or deck.json.
    Expected sheet has the canonical columns used by rows_to_deck().
    """
    if isinstance(upload, (str, Path)):
        path = Path(upload)
        if path.suffix.lower() not in (".xlsx", ".xls"):
            raise ValueError("Only Excel files are supported (.xlsx, .xls).")
        try:
            df = pd.read_excel(path, engine="openpyxl")
        except ImportError as e:
            raise RuntimeError(
                "Excel support requires openpyxl. Add `openpyxl>=3.1` to requirements.txt."
            ) from e
        return rows_to_deck(df)

    # UploadedFile branch (Streamlit)
    name = getattr(upload, "name", "").lower()
    if name.endswith((".xlsx", ".xls")):
        try:
            df = pd.read_excel(upload, engine="openpyxl")
        except ImportError as e:
            raise RuntimeError(
                "Excel support requires openpyxl. Add `openpyxl>=3.1` to requirements.txt."
            ) from e
        return rows_to_deck(df)

    raise ValueError("Only Excel uploads are supported (.xlsx, .xls).")

# ---------------------------
# Image helpers
# ---------------------------

def resolve_local_image_path(img: str) -> str | None:
    """
    Simple + reliable:
    - Trust the spreadsheet: expects "img/{integer}.{ext}".
    - Resolve relative to the folder containing this file (BASE_DIR),
      not the current working directory.
    - No prefixing or extension guessing: load exactly what the sheet says.
    """
    if not img:
        return None

    p = Path(img)

    # 1) Absolute / already-correct path
    if p.exists():
        return str(p)

    # 2) Relative to app code location
    p2 = (BASE_DIR / img).resolve()
    if p2.exists():
        return str(p2)

    # 3) If the sheet gave only a filename (no folder), try BASE_DIR/img/<filename>
    if p.name == img:  # no folder component present
        p3 = (BASE_DIR / "img" / img).resolve()
        if p3.exists():
            return str(p3)

    return None


def show_card_image(card: dict, debug: bool = False):
    raw = (card.get("img") or "").strip()
    if debug:
        st.caption(f"Raw img from card: '{raw or '(empty)'}'")
        st.caption(f"CWD: {Path.cwd()}")

    if not raw:
        if debug:
            st.caption("No `img` specified for this card.")
        return

    try:
        # Remote URL
        if raw.lower().startswith(("http://", "https://")):
            if debug:
                st.caption(f"Loading remote image: {raw}")
            st.markdown("<div style='margin-top:1em'></div>", unsafe_allow_html=True)
            st.image(raw, caption=None, width="stretch")
            return

        # Local path resolution (minimal, non-interfering)
        local = resolve_local_image_path(raw)
        if local:
            if debug:
                st.caption(f"Loading local image: {local}")
            st.markdown("<div style='margin-top:1em'></div>", unsafe_allow_html=True)
            st.image(local, caption=None, width="stretch")
        else:
            if debug:
                # Helpful quick peek at the img folder contents (first 20 entries)
                img_dir = Path.cwd() / "img"
                listing = ", ".join([p.name for p in sorted(img_dir.glob('*'))[:20]]) if img_dir.exists() else "(no img dir)"
                st.error(f"Image not found after resolution attempts: {raw}")
                st.caption(f"img/ listing: {listing}")

    except Exception as e:
        if debug:
            st.error(f"Image load error for '{raw}': {e}")

# ---------------------------
# Exercise renderers
# ---------------------------

def verdict_icon(ok: bool) -> str:
    return # "✅" if ok else "❌"


def exercise_heading_only(card: Dict):
    st.subheader(card["title"])
    st.caption(f"Topic: {card['topic']}")

    acronym_guess = st.text_input("Acronym", key="h_only_acronym")
    # ac_ok = normalize(acronym_guess) == normalize(card["acronym"])
    # st.write(f"Acronym: {verdict_icon(ac_ok)}")

    # answers_ok = []
    # for i, it in enumerate(card["items"]):
    #     ans = st.text_input(f"{it['letter']} →", key=f"h_only_item_{i}")
    #     answers_ok.append(normalize(ans) == normalize(it["text"]))

    # if st.button("Check answers", type="primary"):
        # st.write("---")
        # st.write(f"Acronym: {verdict_icon(ac_ok)}")
        # for i, it in enumerate(card["items"]):
        #     st.write(
        #         f"{it['letter']} → {verdict_icon(answers_ok[i])}  "
        #         f"**Your:** {st.session_state.get(f'h_only_item_{i}','')}  "
        #         f"**Correct:** {it['text']}"
        #     )
        # st.success("All correct! 🎉" if ac_ok else "Keep going!") #and all(answers_ok)


def exercise_heading_plus_acronym(card: Dict):
    st.subheader(card["title"])
    st.caption(f"Topic: {card['topic']}")
    st.markdown(f"**Acronym:** `{card['acronym']}`")

    answers_ok = []
    for i, it in enumerate(card["items"]):
        ans = st.text_input(f"{it['letter']} →", key=f"h_ac_item_{i}")
        answers_ok.append(normalize(ans) == normalize(it["text"]))

    # if st.button("Check answers", type="primary"):
    #     st.write("---")
    #     for i, it in enumerate(card["items"]):
    #         st.write(
    #             f"{it['letter']} → {verdict_icon(answers_ok[i])}  "
    #             f"**Your:** {st.session_state.get(f'h_ac_item_{i}','')}  "
    #             f"**Correct:** {it['text']}"
    #         )
    #     st.success("All correct! 🎉" if all(answers_ok) else "Keep going!")


def exercise_acronym_only(card: Dict):
    st.markdown(f"### Acronym: `{card['acronym']}`")
    st.caption(f"Topic: {card['topic']}")

    title_guess = st.text_input("Card heading/title", key="a_only_title")
    title_ok = normalize(title_guess) == normalize(card["title"])

    answers_ok = []
    for i, it in enumerate(card["items"]):
        ans = st.text_input(f"{it['letter']} →", key=f"a_only_item_{i}")
        answers_ok.append(normalize(ans) == normalize(it["text"]))

    # if st.button("Check answers", type="primary"):
    #     st.write("---")
    #     st.write(
    #         f"Title: {verdict_icon(title_ok)}  **Your:** {title_guess}  **Correct:** {card['title']}"
    #     )
    #     for i, it in enumerate(card["items"]):
    #         st.write(
    #             f"{it['letter']} → {verdict_icon(answers_ok[i])}  "
    #             f"**Your:** {st.session_state.get(f'a_only_item_{i}','')}  "
    #             f"**Correct:** {it['text']}"
    #         )
    #     st.success("All correct! 🎉" if title_ok and all(answers_ok) else "Keep going!")


def exercise_letters_blank(card: Dict):
    st.markdown(
        f"#### Letters: `{''.join([it['letter'] for it in card['items']])}`"
    )
    st.caption(f"Topic: {card['topic']} — Card: {card['title']}")

    answers_ok = []
    for i, it in enumerate(card["items"]):
        ans = st.text_input(f"{it['letter']} →", key=f"letters_blank_{i}")
        answers_ok.append(normalize(ans) == normalize(it["text"]))

    # if st.button("Check answers", type="primary"):
    #     st.write("---")
    #     for i, it in enumerate(card["items"]):
    #         st.write(
    #             f"{it['letter']} → {verdict_icon(answers_ok[i])}  "
    #             f"**Your:** {st.session_state.get(f'letters_blank_{i}','')}  "
    #             f"**Correct:** {it['text']}"
    #         )
    #     st.success("All correct! 🎉" if all(answers_ok) else "Keep going!")


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

    # if st.button("Check answers", type="primary"):
    #     st.write("---")
    #     for i, it in enumerate(card["items"]):
    #         if i in locked_idx:
    #             st.write(f"{it['letter']} → ✅ (prefilled) **{it['text']}**")
    #         else:
    #             st.write(
    #                 f"{it['letter']} → {verdict_icon(answers_ok[i])}  "
    #                 f"**Your:** {st.session_state.get(f'letters_some_{i}','')}  "
    #                 f"**Correct:** {it['text']}"
    #             )
    #     st.success("All correct! 🎉" if all(answers_ok) else "Keep going!")


def exercise_missing_keywords(card: Dict, difficulty: str = "medium"):
    st.caption(f"Topic: {card['topic']} — Card: {card['title']}")
    answers_ok = []
    for i, it in enumerate(card["items"]):
        masked = mask_text(it["text"], difficulty=difficulty)
        st.write(f"{it['letter']}: `{masked}`")
        ans = st.text_input("Full text:", key=f"mask_{i}")
        answers_ok.append(normalize(ans) == normalize(it["text"]))

    # if st.button("Check answers", type="primary"):
    #     st.write("---")
    #     for i, it in enumerate(card["items"]):
    #         st.write(
    #             f"{it['letter']} → {verdict_icon(answers_ok[i])}  "
    #             f"**Your:** {st.session_state.get(f'mask_{i}','')}  "
    #             f"**Correct:** {it['text']}"
    #         )
    #     st.success("All correct! 🎉" if all(answers_ok) else "Keep going!")

def mask_text(s: str, difficulty: str = "medium") -> str:
    """
    Simple keyword masking for the 'Missing key words' mode:
      - easy: mask vowels
      - medium: mask every second character (letters only)
      - hard: mask all letters (leave spaces/punct)
    """
    if difficulty == "easy":
        return re.sub(r"[aeiouAEIOU]", "_", s)
    elif difficulty == "hard":
        return re.sub(r"[A-Za-z]", "_", s)
    else:
        # medium
        out = []
        keep = True
        for ch in s:
            if ch.isalpha():
                out.append(ch if keep else "_")
                keep = not keep
            else:
                out.append(ch)
        return "".join(out)



# ---------------------------
# UI & Layout
# ---------------------------

def reset_inputs():
    for k in list(st.session_state.keys()):
        if k.startswith(("h_only_", "h_ac_", "a_only_", "letters_blank_", "letters_some_", "mask_")):
            st.session_state.pop(k, None)


st.set_page_config(page_title="Rote Cards", page_icon="🗂️", layout="centered")

# st.title("🗂️ Rote Learning Cards")
with st.sidebar:
    st.markdown("### Deck")
    deck_file = st.file_uploader(
        "Upload a deck (.xlsx or .xls)", type=["xlsx", "xls"]
    )

    # Reuse existing deck unless a new file is uploaded
    if "deck" in st.session_state and not deck_file:
        deck = st.session_state["deck"]
    elif deck_file:
        try:
            deck = load_any_deck(deck_file)
            st.session_state["deck"] = deck
            st.success(f"Loaded {len(deck['cards'])} cards from {deck_file.name}")
        except Exception as e:
            st.error(f"Failed to load deck: {e}")
            st.stop()
    else:
        # Start-up default: try the Excel template if present
        excel_default = Path("flashcards_template.xlsx")
        if excel_default.exists():
            deck = load_any_deck(excel_default)
            st.session_state["deck"] = deck
            st.caption("Loaded default deck from flashcards_template.xlsx")
        else:
            st.info("Upload an Excel deck (.xlsx/.xls) or add `flashcards_template.xlsx` next to app.py.")
            st.stop()

    topics = sorted({c["topic"] for c in deck["cards"]})
    topic = st.selectbox("Filter by topic", options=["(All)"] + topics)
    filtered = [c for c in deck["cards"] if topic == "(All)" or c["topic"] == topic]

    if "seed" not in st.session_state:
        st.session_state.seed = int(time.time())
    if "expander_key" not in st.session_state:
        st.session_state.expander_key = 0

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

    # show_img_debug = st.checkbox("Debug images", value=False)
    # st.markdown("---")


# Pick a card deterministically from seed
if not filtered:
    st.warning("No cards match the current filter.")
    st.stop()

idx = random.Random(st.session_state.seed).randint(0, len(filtered) - 1)
card = filtered[idx]

# Title banner
# st.markdown(
#     f"""
#     <div style="background:#166534; color:#fff; padding:12px 16px; border-radius:8px; font-weight:600; letter-spacing:0.2px;">{card['title']}</div>
#     """,
#     unsafe_allow_html=True,
# )

left, right = st.columns([2, 1], vertical_alignment="top")

with right:
    # Only render once; show debug details if enabled
    if mode != "Card heading only":
        show_card_image(card) #, debug=show_img_debug)
    if st.button("🎲 New Card"):
        st.session_state.seed = int(time.time())
        reset_inputs()
        st.session_state.expander_key = st.session_state.get("expander_key", 0) + 1

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
answer_box = st.empty()

# single expander; label carries a nonce so state doesn't persist

def hidden_nonce_label(base: str) -> str:
    # add N zero-width spaces (U+200B) – unique but invisible
    n = st.session_state.get("expander_key", 0)
    return base + ("\u200b" * n)

with answer_box.container():
    with st.expander(hidden_nonce_label("Show full answer"), expanded=False):
        st.markdown(f"**Title:** {card['title']}  \n**Acronym:** `{card['acronym']}`")
        st.write("**Items:**")
        for it in card["items"]:
            st.write(f"- **{it['letter']}** → {it['text']}")






