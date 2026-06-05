import cv2
import sys
import termios
from pathlib import Path
from nltk.corpus import wordnet as wn
import yaml

DISPLAY_WIDTH  = 960
DISPLAY_HEIGHT = 660
FPS = 6

# Arrow-key codes returned by cv2.waitKey on Linux after masking with 0xFF
KEY_LEFT  = 81
KEY_RIGHT = 83

def pick_synset_terminal(word: str) -> str | None:
    """ Print all WordNet synsets for word and prompt user to pick one. Returns synset name or None. """
    synsets = wn.synsets(word.replace(" ", "_"), pos=wn.NOUN)
    if not synsets:
        print(f"  No synsets found for '{word}'.")
        return None

    print(f"\n  Synsets for '{word}':")
    for i, syn in enumerate(synsets):
        print(f"  [{i+1}]  {syn.name():<28}  {syn.definition()}")

    raw = input("\n  Enter number to add (or 0 / Enter to cancel): ").strip()
    if not raw or raw == "0":
        return None
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(synsets):
            return synsets[idx].name()
        print("  Invalid selection.")
    except ValueError:
        print("  Invalid input.")
    return None


def _add_synset(name: str, synset_list: list[str]) -> bool:
    """ Add synset to list if not already present. Returns True if newly added. """
    if name not in synset_list:
        synset_list.append(name)
        print(f"  + Added:  {name:<28}  {wn.synset(name).definition()}")
        return True
    print(f"  (already in list: {name})")
    return False


def _get_hyponyms(syn, depth: int) -> list:
    """ Return all unique hyponyms up to `depth` levels below syn. """
    seen = set()
    frontier = [syn]
    for _ in range(depth):
        next_frontier = []
        for s in frontier:
            for h in s.hyponyms():
                if h not in seen:
                    seen.add(h)
                    next_frontier.append(h)
        frontier = next_frontier
    return list(seen)


def _get_hypernyms(syn, depth: int) -> list:
    """ Return all unique hypernyms up to `depth` levels above syn. """
    seen = set()
    frontier = [syn]
    for _ in range(depth):
        next_frontier = []
        for s in frontier:
            for h in s.hypernyms():
                if h not in seen:
                    seen.add(h)
                    next_frontier.append(h)
        frontier = next_frontier
    return list(seen)


def prompt_related_synsets(synset_name: str, synset_list: list[str]) -> None:
    """
    Show part meronyms and holonyms of synset_name and its hyponyms (4 levels deep),
    then prompt user to add any.
    """
    syn = wn.synset(synset_name)
    search_synsets = [syn] + _get_hyponyms(syn, depth=4) + _get_hypernyms(syn, depth=4)

    meronym_set: set = set()
    holonym_set: set = set()
    for s in search_synsets:
        meronym_set.update(s.part_meronyms())
        holonym_set.update(s.part_holonyms())

    meronyms = list(meronym_set)
    holonyms = list(holonym_set)

    if not meronyms and not holonyms:
        return

    related: list = []
    if meronyms:
        print(f"\n  Part meronyms (parts of '{synset_name}' or its hyponyms):")
        for s in meronyms:
            related.append(s)
            print(f"  [{len(related)}]  {s.name():<28}  {s.definition()}")
    if holonyms:
        print(f"\n  Part holonyms ('{synset_name}' or its hyponyms are part of):")
        for s in holonyms:
            related.append(s)
            print(f"  [{len(related)}]  {s.name():<28}  {s.definition()}")

    raw = input("\n  Add any related synsets? Enter numbers (comma-separated) or Enter to skip: ").strip()
    if not raw:
        return

    for token in raw.split(","):
        token = token.strip()
        try:
            idx = int(token) - 1
            if 0 <= idx < len(related):
                _add_synset(related[idx].name(), synset_list)
            else:
                print(f"  Invalid selection: {token}")
        except ValueError:
            print(f"  Invalid input: {token}")


def word_lookup(synset_list: list[str]) -> None:
    """
    Repeatedly prompt for words and add chosen synsets until the user cancels.
    Accepts multiple comma-separated words per line.
    """
    while True:
        raw = input("\nEnter word(s) to look up (comma-separated, or Enter to finish): ").strip().lower()
        if not raw:
            break
        words = [w.strip() for w in raw.split(",") if w.strip()]
        for word in words:
            name = pick_synset_terminal(word)
            if name:
                _add_synset(name, synset_list)
                prompt_related_synsets(name, synset_list)


def _load_size_constraints(path: Path) -> dict[str, tuple]:
    """Return {synset_name: (min_size, max_size)} from an existing YAML file."""
    if not path.exists():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f)
    constraints = {}
    for entry in data.get("initial_synset_dict", []):
        if isinstance(entry, dict):
            constraints[entry["name"]] = (entry.get("min_size"), entry.get("max_size"))
    return constraints


def load_synset_list(path: Path) -> list[str]:
    """Load existing synset list from YAML (initial_synset_dict format), or return an empty list."""
    if not path.exists():
        return []
    with open(path) as f:
        data = yaml.safe_load(f)
    entries = data.get("initial_synset_dict", [])
    result = []
    for entry in entries:
        if isinstance(entry, str):
            result.append(entry)
        elif isinstance(entry, dict):
            result.append(entry["name"])
    return result


def save_synset_list(path: Path, synset_list: list[str]) -> None:
    """Save synset list as YAML in initial_synset_dict format, preserving existing size constraints."""
    existing = _load_size_constraints(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    synset_list = sorted(synset_list)
    max_name_len = max(len(name) for name in synset_list) if synset_list else 0
    lines = ["initial_synset_dict:\n"]
    for name in synset_list:
        min_size, max_size = existing.get(name, (None, None))
        min_str = str(min_size) if min_size is not None else "null"
        max_str = str(max_size) if max_size is not None else "null"
        padding = " " * (max_name_len - len(name))
        lines.append(f"  - {{name: \"{name}\",{padding} min_size: {min_str}, max_size: {max_str}}}\n")
    with open(path, "w") as f:
        f.writelines(lines)
    print(f"\nSaved {len(synset_list)} synset(s) to {path}")


def run_viewer(img_data, synset_list: list[str], title: str = "Viewer", fps: int = FPS,
               frame_skip: int = 1) -> list[str]:
    """
    Display images from img_data with interactive controls.
    Returns the (possibly modified) synset_list.

    Controls:
      SPACE       : pause / resume
      LEFT / A    : step back one frame (pauses automatically)
      RIGHT / D   : step forward one frame
      W           : open word-lookup dialog to add a synset
      Q / ESC     : quit and save
    """
    times = img_data.times[::frame_skip]
    n = len(times)

    WIN = f"{title}  |  SPACE=pause  LEFT/RIGHT=step  W=lookup  Q=quit"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    idx    = 0
    paused = False
    delay  = max(1, int(1000 / fps))  # ms between frames during playback

    print("Controls: SPACE=pause/resume  LEFT/A=back  RIGHT/D=forward  W=word lookup  Q/ESC=quit\n")

    while 0 <= idx < n:
        img = img_data.img(times[idx])
        if img is None:
            idx += 1
            continue

        # Resize for display and overlay status
        display = cv2.resize(img, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        label = (f"Frame {idx+1}/{n}  |  {'PAUSED' if paused else 'PLAYING'}"
                 f"  |  Synsets in list: {len(synset_list)}")
        cv2.putText(display, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(WIN, display)

        # Block on key when paused, otherwise advance at playback rate
        key = cv2.waitKey(0 if paused else delay) & 0xFF

        if key in (ord('q'), 27):               # Q / ESC: quit
            break
        elif key == ord(' '):                   # SPACE: toggle pause
            paused = not paused
        elif key in (KEY_LEFT, ord('a')):       # LEFT / A: step back
            idx = max(0, idx - 1)
            paused = True
        elif key in (KEY_RIGHT, ord('d')):      # RIGHT / D: step forward
            idx = min(n - 1, idx + 1)
        elif key == ord('w'):                   # W: open word lookup in terminal
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
            word_lookup(synset_list)
            continue

        if not paused:
            idx += 1

    cv2.destroyAllWindows()
    return synset_list
