import json
import random
import re
import time
from pathlib import Path
from typing import List, Dict

import streamlit as st

# ---------------------------
# Data loading
# ---------------------------
@st.cache_data
def load_deck(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

# ---------------------------
# UI helpers
# ---------------------------
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

def verdict_icon(ok: bool) -> str:
    return "✅" if ok else "❌"

# ---------------------------
# Exercise renderers
# ---------------------------
def exercise_heading_only(card: Dict):
    """Show heading only; user supplies acronym + items."""
    st.subheader(card["title"])
    st.caption(f"Topic: {card['topic']}")

    # Guess length of acronym
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
            st.write(f"{it['letter']} → {verdict_icon(answers_ok[i])}  "
                     f"**Your:** {st.session_state.get(f'h_only_item_{i}','')}  "
                     f"**Correct:** {it['text']}")
        st.success("All correct! 🎉" if ac_ok and all(answers_ok) else "Keep going!")

def exercise_heading_plus_acronym(card: Dict):
    """Show heading + acronym; user supplies items."""
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
            st.write(f"{it['letter']} → {verdict_icon(answers_ok[i])}  "
                     f"**Your:** {st.session_state.get(f'h_ac_item_{i}','')}  "
                     f"**Correct:** {it['text']}")
        st.success("All correct! 🎉" if all(answers_ok) else "Keep going!")

def exercise_acronym_only(card: Dict):
    """Show acronym only; user supplies heading + items."""
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
        st.write(f"Title: {verdict_icon(title_ok)}  "
                 f"**Your:** {title_guess}  **Correct:** {card['title']}")
        for i, it in enumerate(card["items"]):
            st.write(f"{it['letter']} → {verdict_icon(answers_ok[i])}  "
                     f"**Your:** {st.session_state.get(f'a_only_item_{i}','')}  "
                     f"**Correct:** {it['text']}")
        st.success("All correct! 🎉" if title_ok and all(answers_ok) else "Keep going!")

def exercise_letters_blank(card: Dict):
    """Letters down the side, user fills all items (pure recall)."""
    st.markdown(f"#### Letters: `{''.join([it['letter'] for it in card['items']])}`")
    st.caption(f"Topic: {card['topic']} — Card: {card['title']}")

    answers_ok = []
    for i, it in enumerate(card["items"]):
        ans = st.text_input(f"{it['letter']} →", key=f"letters_blank_{i}")
        answers_ok.append(normalize(ans) == normalize(it["text"]))

    if st.button("Check answers", type="primary"):
        st.write("---")
        for i, it in enumerate(card["items"]):
            st.write(f"{it['letter']} → {verdict_icon(answers_ok[i])}  "
                     f"**Your:** {st.session_state.get(f'letters_blank_{i}','')}  "
                     f"**Correct:** {it['text']}")
        st.success("All correct! 🎉" if all(answers_ok) else "Keep going!")

def exercise_letters_some_prefilled(card: Dict, prefill_ratio: float = 0.4):
    """Some items are prefilled (locked); others blank for the user."""
    st.markdown(f"#### Letters: `{''.join([it['letter'] for it in card['items']])}`")
    st.caption(f"Topic: {card['topic']} — Card: {card['title']}")

    n = len(card["items"])
    k = max(1, int(n * prefill_ratio))
    rng = random.Random(st.session_state.get("seed", 0))
    locked_idx = set(rng.sample(range(n), k))

    answers_ok = []
    for i, it in enumerate(card["items"]):
        if i in locked_idx:
            st.text_input(f"{it['letter']} → (prefilled)", value=it["text"], disabled=True, key=f"pref_{i}")
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
                st.write(f"{it['letter']} → {verdict_icon(answers_ok[i])}  "
                         f"**Your:** {st.session_state.get(f'letters_some_{i}','')}  "
                         f"**Correct:** {it['text']}")
        st.success("All correct! 🎉" if all(answers_ok) else "Keep going!")

def exercise_missing_keywords(card: Dict, difficulty: str = "medium"):
    """
    Show masked versions of each item's text. User must supply the original.
    """
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
            st.write(f"{it['letter']} → {verdict_icon(answers_ok[i])}  "
                     f"**Your:** {st.session_state.get(f'mask_{i}','')}  "
                     f"**Correct:** {it['text']}")
        st.success("All correct! 🎉" if all(answers_ok) else "Keep going!")

# ---------------------------
# App
# ---------------------------
st.set_page_config(page_title="Rote Cards", page_icon="🗂️", layout="centered")

st.title("🗂️ Rote Learning Cards")
with st.sidebar:
    st.markdown("### Deck")
    deck_file = st.file_uploader("Upload a deck.json (optional)", type=["json"])
    if "seed" not in st.session_state:
        st.session_state.seed = int(time.time())  # new session seed

    rng = random.Random(st.session_state.seed)

    if deck_file:
        deck = json.load(deck_file)
    else:
        # Fallback to local file
        default_path = Path("deck.json")
        if not default_path.exists():
            st.error("No `deck.json` found. Upload a deck or add one beside app.py.")
            st.stop()
        deck = load_deck(default_path)

    topics = sorted({c["topic"] for c in deck["cards"]})
    topic = st.selectbox("Filter by topic", options=["(All)"] + topics)
    filtered = [c for c in deck["cards"] if topic == "(All)" or c["topic"] == topic]

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
        difficulty = st.select_slider("Masking difficulty", options=["easy", "medium", "hard"], value="medium")
    else:
        difficulty = None

    if mode == "Letters down the side (some prefilled)":
        prefill_ratio = st.slider("Prefill ratio", min_value=0.2, max_value=0.8, value=0.4, step=0.1)
    else:
        prefill_ratio = 0.4

    st.markdown("---")
    if st.button("🎲 New Card"):
        st.session_state.seed = int(time.time())

# Pick a card deterministically from seed
if not filtered:
    st.warning("No cards match the current filter.")
    st.stop()

idx = random.Random(st.session_state.seed).randint(0, len(filtered) - 1)
card = filtered[idx]

# Render the chosen exercise
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
