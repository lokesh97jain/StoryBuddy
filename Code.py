import os
import re
import tempfile
import time
from typing import Dict, List, Tuple, Iterable, Optional
import requests
import streamlit as st
from gtts import gTTS
from openai import OpenAI
import base64
import logging



try:
    # Optional: load .env if python-dotenv is available
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# ------------------ Config ------------------

# Prefer environment/Streamlit secrets for sensitive values. Never hardcode keys.
# Accessing st.secrets can raise if no secrets.toml exists, so guard it.
def _load_openai_key() -> Optional[str]:
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key
    try:
        return st.secrets["OPENAI_API_KEY"]  # type: ignore[index]
    except Exception:
        return None

OPENAI_API_KEY = _load_openai_key()

MEDIA_MODE = "tts"  # or "image" (currently both used if available)
MODEL = "gpt-4o-mini"


def get_openai_client() -> Optional[OpenAI]:
    """Lazily construct an OpenAI client if an API key is available.

    Returns None if the key is missing or client creation fails.
    """
    key = OPENAI_API_KEY
    if not key:
        return None
    try:
        return OpenAI(api_key=key)
    except Exception:
        return None


client = get_openai_client()

# Configure backend logging (printed to Streamlit server console)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("storybuddy")


# ------------------ Guardrails / Helpers ------------------

DEFAULT_BAD_WORDS = [
    "kill",
    "die",
    "blood",
    "gun",
    "knife",
    "bully",
    "violence",
    "hate",
    "racist",
    "drugs",
    "alcohol",
    "terror",
    "bomb",
    "abuse",
    "weapon",
    "stupid",
]

# Fallback artwork for when illustration generation is slow or fails.
PLACEHOLDER_IMAGE = "https://dummyimage.com/640x400/ebf4ff/1d4ed8&text=Illustration+loading..."
ERROR_IMAGE = "https://dummyimage.com/640x400/fef2f2/dc2626&text=Image+unavailable"


# ---------- Reading level presets & prompt builder ----------

LEVEL_PRESETS: Dict[str, Tuple[int, str]] = {
    "Age 3-4 (Pre-K)": (6, "very simple everyday words for toddlers"),
    "Age 5-6 (K-G1)": (8, "simple familiar words and short phrases"),
    "Age 7-8 (G2)": (10, "common words, some gentle new words"),
    "Age 9-10 (G3)": (12, "slightly longer sentences, a few new words"),
}


def level_from_age(age: int) -> Tuple[int, str]:
    if age <= 4:
        return LEVEL_PRESETS["Age 3-4 (Pre-K)"]
    if age <= 6:
        return LEVEL_PRESETS["Age 5-6 (K-G1)"]
    if age <= 8:
        return LEVEL_PRESETS["Age 7-8 (G2)"]
    return LEVEL_PRESETS["Age 9-10 (G3)"]


def build_system_prompt(max_words_per_sentence: int, vocab_hint: str) -> str:
    """Create a system prompt tuned to parent-chosen reading level."""
    return (
        "You are StoryBuddy, a gentle, playful narrator for young children.\n"
        "Constraints:\n"
        f"- Exactly 3 short sentences per turn; each sentence MUST be <= {max_words_per_sentence} words.\n"
        f"- Use {vocab_hint}. Avoid complex clauses.\n"
        "- Warm, kind, encouraging tone; curious and imaginative.\n"
        "- No fear, violence, bullying, injuries, weapons, scary themes, or sensitive topics.\n"
        "- End with exactly two choices labeled (A) and (B), each 3-6 words, age-appropriate.\n"
        "Output format:\n"
        "STORY: <3 sentences>\n"
        "CHOICES: (A) <option A> | (B) <option B>\n"
    )


def current_bad_words() -> set[str]:
    words = st.session_state.get("bad_words", DEFAULT_BAD_WORDS)
    return {w.strip().lower() for w in words if isinstance(w, str) and w.strip()}


def safe_check(text: str) -> bool:
    low = text.lower()
    bad_words = current_bad_words()
    return not any(b in low for b in bad_words)


def fallback_snippet() -> Dict[str, str]:
    return {
        "story": "STORY: We visit a sunny park. A blue butterfly dances near flowers. We feel brave and curious.\n",
        "choices": "CHOICES: (A) Follow the butterfly | (B) Build a tiny fort",
    }


def parse_snippet(raw: str) -> Tuple[str, str, str]:
    """Parse the model text into (story_text, choiceA, choiceB).

    Robust to spacing/casing; expects STORY: and CHOICES: lines.
    """
    text = raw.strip()

    # Extract STORY:
    story_match = re.search(r"(?is)story:\s*(.+?)\n\s*choices:", text)
    story = story_match.group(1).strip() if story_match else ""

    # Extract CHOICES: (A) ... | (B) ...
    choices_line_match = re.search(r"(?is)choices:\s*(.+)$", text)
    choiceA, choiceB = "", ""
    if choices_line_match:
        cl = choices_line_match.group(1).strip()
        # try split by |
        parts = [p.strip() for p in cl.split("|")]
        if len(parts) == 2:
            # remove leading labels
            a = re.sub(r"^\(?A\)?[:\-]?\s*", "", parts[0], flags=re.I).strip()
            b = re.sub(r"^\(?B\)?[:\-]?\s*", "", parts[1], flags=re.I).strip()
            choiceA, choiceB = a, b

    return story, choiceA, choiceB


def llm_snippet(
    theme: str,
    child_name: str,
    history: List[Dict],
    chosen: Optional[str],
) -> Dict[str, str]:
    """Call the LLM to produce a snippet. History holds prior turns as text.

    If chosen is 'A' or 'B', we continue; else we start.
    Returns a dictionary with raw output, parsed story, and choices A/B.
    Falls back to a safe snippet if unavailable.
    """
    if not client:
        fb = fallback_snippet()
        s, a, b = parse_snippet(fb["story"] + "\n" + fb["choices"])
        return {"story_raw": fb["story"] + "\n" + fb["choices"], "story": s, "choiceA": a, "choiceB": b}

    if chosen is None:
        user_prompt = (
            f"Child: {child_name or 'friend'}. Theme: {theme}.\n"
            "Start the story. Follow the exact output format."
        )
    else:
        prev_text = history[-1]["story_raw"]
        chosen_text = history[-1]["choiceA"] if chosen == "A" else history[-1]["choiceB"]
        user_prompt = (
            "Continue the same story in the same style.\n"
            f"Prior snippet:\n{prev_text}\n"
            f"Chosen option: ({chosen}) {chosen_text}\n"
            "Follow the exact output format."
        )

    max_words = st.session_state.get("target_max_words", 10)
    vocab_hint = st.session_state.get("target_vocab_hint", "common words, some gentle new words")
    system_prompt = build_system_prompt(max_words, vocab_hint)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=220,
        )
        raw = resp.choices[0].message.content.strip()

        def too_long(text: str, cap: int) -> bool:
            sentences = [seg.strip() for seg in re.split(r"[.!?]+", text) if seg.strip()]
            if not sentences:
                return False
            return any(len(sent.split()) > cap for sent in sentences)

        if too_long(raw, max_words):
            resp_retry = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt + f"\nIMPORTANT: Each sentence must be <= {max_words} words."},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.5,
                max_tokens=220,
            )
            raw = resp_retry.choices[0].message.content.strip()

        story_text, a, b = parse_snippet(raw)

        # Validate minimal shape; fallback if unsafe or malformed
        if not story_text or not a or not b or not safe_check(story_text + " " + a + " " + b):
            fb = fallback_snippet()
            story_text, a, b = parse_snippet(fb["story"] + "\n" + fb["choices"])

        return {"story_raw": raw, "story": story_text, "choiceA": a, "choiceB": b}
    except Exception:
        fb = fallback_snippet()
        s, a, b = parse_snippet(fb["story"] + "\n" + fb["choices"])
        return {"story_raw": fb["story"] + "\n" + fb["choices"], "story": s, "choiceA": a, "choiceB": b}


def tts_from_text(text: str) -> Optional[str]:
    """Generate a temporary mp3 from text and return its filesystem path.

    Returns None if TTS fails.
    """
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        # Close handle to allow gTTS to write on Windows
        tmp.close()
        gTTS(text).save(tmp.name)
        return tmp.name
    except Exception:
        return None


def extract_words(text: str, k: int = 3, exclude: Optional[Iterable[str]] = None) -> List[str]:
    """Very simple heuristic for demo: longer alphabetic tokens not in exclusion."""
    tokens = re.findall(r"[A-Za-z]+", text)
    tokens = [t.lower() for t in tokens if len(t) >= 5]

    excluded: set[str] = set()
    if exclude:
        for raw in exclude:
            if not raw:
                continue
            excluded.update(re.findall(r"[A-Za-z]+", raw.lower()))

    seen = set()
    out: List[str] = []
    for t in tokens:
        if t in seen or t in excluded:
            continue
        out.append(t)
        seen.add(t)
        if len(out) >= k:
            break

    if not out:
        out = ["adventure", "curious", "bravery"][:k]
    return out


def fetch_word_info(word: str) -> Tuple[Optional[str], Optional[str]]:
    """Retrieve a simple definition and audio URL for the word using a public dictionary API."""
    try:
        resp = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}", timeout=5)
        if resp.status_code != 200:
            return None, None
        data = resp.json()
        if not isinstance(data, list) or not data:
            return None, None

        entry = data[0]
        definition = None
        for meaning in entry.get("meanings", []):
            defs = meaning.get("definitions")
            if defs:
                definition = defs[0].get("definition")
                if definition:
                    break

        audio_url = None
        for phonetic in entry.get("phonetics", []):
            audio = phonetic.get("audio")
            if audio:
                audio_url = audio
                break

        return definition, audio_url
    except Exception:
        return None, None


def get_word_learning_resource(word: str) -> Dict[str, Optional[str]]:
    """Return a dictionary with definition and audio source data for the word."""
    cache = st.session_state.word_lookup_cache
    word_key = word.lower()
    if word_key in cache:
        return cache[word_key]

    definition, audio_url = fetch_word_info(word_key)
    audio_path = None
    if not audio_url:
        audio_path = st.session_state.word_audio_cache.get(word_key)
        if not audio_path:
            try:
                audio_path = tts_from_text(word_key)
                st.session_state.word_audio_cache[word_key] = audio_path
            except Exception:
                audio_path = None

    resource = {
        "definition": definition or "Definition currently unavailable.",
        "audio_url": audio_url,
        "audio_path": audio_path,
    }
    cache[word_key] = resource
    return resource


def collect_learning_words(
    turns: List[Dict], exclude: Optional[Iterable[str]] = None, max_words: int = 50
) -> List[str]:
    """Aggregate unique learning words across turns, respecting exclusions."""
    combined: List[str] = []
    seen = set()
    for turn in turns:
        story_text = turn.get("story", "")
        words = extract_words(story_text, k=max_words, exclude=exclude)
        for w in words:
            if w in seen:
                continue
            combined.append(w)
            seen.add(w)
    return combined

# Image Generation
def image_from_text(text: str):
    """Generate an illustration via OpenAI Images API and return bytes.

    Prefers base64 content; falls back to downloading from URL if provided.
    Stores a compact status in session for optional UI diagnostics.
    """ 
    # OPENAI_API_KEY = ""   # paste the actual key
    # client = OpenAI(api_key=OPENAI_API_KEY)
    
    env_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=env_key)
    prompt = f"Kid-friendly, colorful cartoon illustration: {text[:200]}"
    # Generate image and prefer base64 payload; fallback to URL if present
    result = client.images.generate(
        model="gpt-image-1",
        prompt=prompt, 
    ) 
    b64_image = result.data[0].b64_json        # <-- base64 string (no data URL prefix)
    img_bytes = base64.b64decode(b64_image)    # <-- decode to bytes

    return img_bytes


# ------------------ Streamlit UI ------------------

st.set_page_config(page_title="StoryBuddy (Ainia demo)", page_icon="SB", layout="centered")
st.title("StoryBuddy - Tiny Interactive Demo")

# Initialize session state
if "turns" not in st.session_state:
    st.session_state.turns = []  # list of dicts with story, choiceA, choiceB, media paths
if "bad_words" not in st.session_state:
    st.session_state.bad_words = list(DEFAULT_BAD_WORDS)
if "child_name" not in st.session_state:
    st.session_state.child_name = ""
if "child_age" not in st.session_state:
    st.session_state.child_age = 6
if "target_level_mode" not in st.session_state:
    st.session_state.target_level_mode = "by_age"
if "target_level_name" not in st.session_state:
    st.session_state.target_level_name = "Age 7-8 (G2)"
if "target_max_words" not in st.session_state:
    st.session_state.target_max_words = 10
if "target_vocab_hint" not in st.session_state:
    st.session_state.target_vocab_hint = LEVEL_PRESETS["Age 7-8 (G2)"][1]
if "show_parent_settings" not in st.session_state:
    st.session_state.show_parent_settings = False
if "word_lookup_cache" not in st.session_state:
    st.session_state.word_lookup_cache = {}
if "word_audio_cache" not in st.session_state:
    st.session_state.word_audio_cache = {}


tab1, tab2 = st.tabs(["Play", "Parent View"])

with tab1:
    st.caption("Pick a theme, hear a short story, make one choice, and continue once.")

    name = st.session_state.child_name
    age = st.session_state.child_age

    info_col, theme_col = st.columns(2)
    with info_col:
        st.markdown(f"**Child:** {name or 'Not set'} | **Age:** {age}")
        st.caption("Adjust child details in Parent View settings.")
    with theme_col:
        theme = st.selectbox("Theme", ["Space", "Animals", "Ocean", "Jungle", "Robots", "Fairy Garden"])

    start = st.button(
        "Start Story", type="primary", disabled=len(st.session_state.turns) > 0
    )

    if start:
        with st.spinner("Generating narration and illustration..."):
            # Safely get story, plus media fallbacks
            turn = llm_snippet(theme=theme, child_name=name, history=st.session_state.turns, chosen=None)
            audio_path = tts_from_text(turn["story"]) if MEDIA_MODE == "tts" else None
            img_bytes = image_from_text(turn["story"]) 
        turn["media_path"] = audio_path  
        turn["image_data"] = img_bytes 
        st.session_state.turns.append(turn)

    # Render current turns
    for i, t in enumerate(st.session_state.turns):
        st.subheader(f"Part {i+1}")
        st.write(t["story"])

        if t.get("media_path"):
            st.audio(t["media_path"])
 
        # show image if we have it
        if t.get("image_data"):
            st.image(t["image_data"], caption="Illustration")
        else:
            st.caption("Illustration unavailable.")

    learning_words: List[str] = []
    if st.session_state.turns:
        learning_words = collect_learning_words(st.session_state.turns, exclude=[name])
        disallowed = current_bad_words()
        learning_words = [w for w in learning_words if w not in disallowed]

    # Branching buttons (only if exactly 1 turn so far)
    if len(st.session_state.turns) == 1:
        t = st.session_state.turns[-1]
        a, b = t["choiceA"], t["choiceB"]
        c1, c2 = st.columns(2)
        with c1:
            pick_a = st.button(f"Choose A: {a}")
        with c2:
            pick_b = st.button(f"Choose B: {b}")

        if pick_a or pick_b:
            chosen = "A" if pick_a else "B"
            with st.spinner("Generating narration and illustration..."):
                # Continue story and generate media if available
                turn = llm_snippet(theme=theme, child_name=name, history=st.session_state.turns, chosen=chosen)
                audio_path = tts_from_text(turn["story"]) if MEDIA_MODE == "tts" else None
                img_bytes = image_from_text(turn["story"]) 
            turn["media_path"] = audio_path 
            turn["image_data"] = img_bytes 
            st.session_state.turns.append(turn)
            st.rerun()

    if learning_words:
        st.subheader("Words we're learning")
        for word in learning_words:
            info = get_word_learning_resource(word)
            st.markdown(f"**{word.capitalize()}** - {info['definition']}")
            if info.get("audio_url"):
                st.audio(info["audio_url"])
            elif info.get("audio_path"):
                st.audio(info["audio_path"])
            else:
                st.caption("Pronunciation audio unavailable.")

    st.divider()
    if st.button("Reset Story"):
        st.session_state.turns = []
        st.success("Story reset.")
        time.sleep(0.4)
        st.rerun()

with tab2:
    st.caption("Lightweight transparency card to build parent trust.")

    if st.button("Parent Settings", type="primary"):
        st.session_state.show_parent_settings = not st.session_state.show_parent_settings

    if st.session_state.show_parent_settings:
        with st.form("parent_settings_form"):
            name_col, age_col = st.columns([2, 1])
            with name_col:
                form_child_name = st.text_input("Child name", st.session_state.child_name)
            with age_col:
                form_child_age = st.number_input(
                    "Child age", min_value=3, max_value=12, value=int(st.session_state.child_age or 6), step=1
                )

            st.markdown("### Reading Level Controls")
            st.caption(
                "Reading level controls set how simple the story's language is: vocabulary and sentence length."
            )

            form_mode = st.radio(
                "How to set reading level?",
                ["by_age", "manual", "custom"],
                index=["by_age", "manual", "custom"].index(st.session_state.target_level_mode),
                horizontal=True,
                help="by_age = auto from age; manual = choose preset; custom = set your own max words/sentence",
            )

            form_level_name = st.selectbox(
                "Manual preset (if 'manual' is selected)",
                list(LEVEL_PRESETS.keys()),
                index=list(LEVEL_PRESETS.keys()).index(st.session_state.target_level_name),
            )

            form_max_words = st.slider(
                "Custom: Max words per sentence (if 'custom' is selected)",
                min_value=4,
                max_value=16,
                value=st.session_state.target_max_words,
                step=1,
            )

            default_bad_words = ", ".join(st.session_state.bad_words)
            form_bad_words = st.text_area("Blocked words (comma-separated)", default_bad_words, height=120)

            submitted = st.form_submit_button("Save parent settings")

        if submitted:
            # Persist child info
            st.session_state.child_name = form_child_name.strip()
            st.session_state.child_age = int(form_child_age)

            # Reading level configuration
            st.session_state.target_level_mode = form_mode
            if form_mode == "by_age":
                max_words, hint = level_from_age(st.session_state.child_age)
            elif form_mode == "manual":
                st.session_state.target_level_name = form_level_name
                max_words, hint = LEVEL_PRESETS[form_level_name]
            else:
                max_words, hint = form_max_words, "age-appropriate simple words"

            st.session_state.target_max_words = max_words
            st.session_state.target_vocab_hint = hint

            # Sanitize bad words list
            sanitized_words: List[str] = []
            seen = set()
            for candidate in re.split(r"[,\r\n]+", form_bad_words):
                word = candidate.strip().lower()
                if not word or word in seen:
                    continue
                sanitized_words.append(word)
                seen.add(word)
            st.session_state.bad_words = sanitized_words or list(DEFAULT_BAD_WORDS)

            st.success(f"Saved: max {max_words} words/sentence; vocab: {hint}")

    # Parent view metrics
    total_sentences = 0
    latest_text = ""
    for t in st.session_state.turns:
        # Count sentences (rough)
        sents = [s for s in re.split(r"[.!?]+", t["story"]) if s.strip()]
        total_sentences += len(sents)
        latest_text = t["story"]

    est_seconds = total_sentences * 2
    minutes = max(1, round(est_seconds / 60)) if est_seconds else 1

    words = collect_learning_words(st.session_state.turns, exclude=[st.session_state.child_name])
    disallowed_parent = current_bad_words()
    words = [w for w in words if w not in disallowed_parent]

    # Estimated level from generated story (rough heuristic)
    tokens = re.findall(r"[A-Za-z']+", latest_text)
    avg_len = round(len(tokens) / (total_sentences or 1), 1) if tokens else 7.0
    est_level = "Early Grade 2" if avg_len <= 9 else "Grade 3"

    target_words = st.session_state.get("target_max_words", 10)
    target_hint = st.session_state.get("target_vocab_hint", "age-appropriate simple words")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Today's reading time", f"{minutes} min")
    with c2:
        st.metric("Words surfaced", "")
        st.write(", ".join(words))
    with c3:
        st.metric(
            "Reading level (estimated)", est_level, delta=f"Target <= {target_words} words/sent"
        )
        st.caption(f"Target vocab: {target_hint}")

    st.caption(f"Child: {st.session_state.child_name or 'Not set'} (age {st.session_state.child_age})")
    st.info(
        "Guardrails: kid-safe system prompt, banned-word filter, fallback snippet on failure, fixed output shape, short snippets."
    )
