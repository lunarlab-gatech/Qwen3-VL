from __future__ import annotations

import clip
try:
    import cupy as cp
except ImportError:
    cp = None
import logging
logger = logging.getLogger(__name__)
import nltk
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset, Lemma
import numpy as np
import os
from pathlib import Path
from PIL import Image  # Required by clip.load, even if not used
import torch

class LemmaWrapper():
    """Wrapper for a WordNet Lemma"""
    
    def __init__(self, lemma: Lemma):
        self.lemma = lemma

    def __str__(self) -> str:
        str_rep = f"Name: {self.get_word()}\n"
        str_rep += f"Synset: {self.lemma.synset()}\n"
        antonyms = self.lemma.antonyms()
        if len(antonyms) > 0:str_rep += f"Antonyms: {self.lemma.antonyms()}\n"
        return str_rep
    
    def get_word(self) -> str:
        """ Assuming the lemma name (ex. angle-closure_glaucoma), return the corresponding word (ex. angle-closure glaucoma)"""
        lemma_str: str = self.lemma.name()
        return lemma_str.replace("_", " ")

class SynsetWrapper():
    """Wrapper for a WordNet Synset"""

    _hyponym_cache: dict[tuple[str, int], set[SynsetWrapper]] = {}

    def __init__(self, synset: Synset):
        self.synset: Synset = synset

    def get_word(self) -> str:
        """ Return the synset name (ex. montezuma.n.01) with underscores replaced by spaces (ex. montezuma.n.01)"""
        return self.synset.name().replace("_", " ")

    def __repr__(self) -> str:
        return self.synset.name()

    def __eq__(self, other) -> bool:
        if isinstance(other, SynsetWrapper):
            return self.synset == other.synset
        return False

    def __hash__(self) -> int:
        return hash(self.synset)

    def is_hyponym_of(self, other: SynsetWrapper) -> bool:
        """Returns True if self is a (direct or indirect) hyponym of other,
        i.e. other is a hypernym of self."""
        return other.synset in self.synset.closure(lambda s: s.hypernyms())

    def get_all_meronyms(self, include_hypernyms: bool, meronym_levels: int = 1) -> set[SynsetWrapper]:
        """
        Parameters:
            include_hypernyms: Whether to include meronyms from direct hypernyms as well.
            meronym_levels: The number of levels of meronyms to retrieve.
        """
        meronyms: set[SynsetWrapper] = set()

        if meronym_levels <= 0: return meronyms

        meronyms = {SynsetWrapper(s) for s in self.synset.part_meronyms()}
        # meronyms.update(SynsetWrapper(s) for s in self.synset.substance_meronyms())
        # meronyms.update(SynsetWrapper(s) for s in self.synset.member_meronyms())

        lower_level_meronyms: set[SynsetWrapper] = set()
        for meronym in meronyms:
            lower_level_meronyms.update(meronym.get_all_meronyms(include_hypernyms, meronym_levels-1))
        meronyms.update(lower_level_meronyms)

        if include_hypernyms:
            for hypernym in self.synset.hypernyms():
                meronyms.update(SynsetWrapper(hypernym).get_all_meronyms(False, meronym_levels))
        return meronyms

    def get_all_holonyms(self, include_hypernyms: bool, holonym_levels: int = 1) -> set[SynsetWrapper]:
        """
        Parameters:
            include_hypernyms: Whether to include holonyms from direct hypernyms as well.
            holonym_levels: The number of levels of holonyms to retrieve.
        """
        holonyms: set[SynsetWrapper] = set()

        if holonym_levels <= 0: return holonyms

        holonyms = {SynsetWrapper(s) for s in self.synset.part_holonyms()}
        # holonyms.update(SynsetWrapper(s) for s in self.synset.substance_holonyms())
        # holonyms.update(SynsetWrapper(s) for s in self.synset.member_holonyms())

        higher_level_holonyms: set[SynsetWrapper] = set()
        for holonym in holonyms:
            higher_level_holonyms.update(holonym.get_all_holonyms(include_hypernyms, holonym_levels-1))
        holonyms.update(higher_level_holonyms)

        if include_hypernyms:
            for hypernym in self.synset.hypernyms():
                holonyms.update(SynsetWrapper(hypernym).get_all_holonyms(False, holonym_levels))
        return holonyms
    
    def get_all_hyponyms(self, hyponym_levels: int = 4) -> set[SynsetWrapper]:
        
        # Get hyponyms from cache if we've already had to look it pu
        cache_key = (self.synset.name(), hyponym_levels)
        if cache_key in SynsetWrapper._hyponym_cache:
            return SynsetWrapper._hyponym_cache[cache_key]

        # If there are no hyponym levels left, return an empty set
        hyponyms: set[SynsetWrapper] = set()
        if hyponym_levels <= 0:
            SynsetWrapper._hyponym_cache[cache_key] = hyponyms
            return hyponyms

        # Get direct hyponyms
        hyponyms: set[SynsetWrapper] = {SynsetWrapper(s) for s in self.synset.hyponyms()}

        # Recursively get lower level hyponyms as well
        lower_level_hyponyms: set[SynsetWrapper] = set()
        for hyponym in hyponyms:
            lower_level_hyponyms.update(hyponym.get_all_hyponyms(hyponym_levels-1))
        hyponyms.update(lower_level_hyponyms)

        SynsetWrapper._hyponym_cache[cache_key] = hyponyms
        return hyponyms

    @staticmethod
    def synsets_as_strings(synsets: list[Synset] | list[SynsetWrapper], max_len: int = 7) -> list[str]:
        if synsets and isinstance(synsets[0], Synset):
            wrapped: list[SynsetWrapper] = [SynsetWrapper(s) for s in synsets]
        else:
            wrapped: list[SynsetWrapper] = synsets
        strings = [h.get_word() for h in wrapped]
        if len(strings) > max_len:
            return strings[0:max_len - 1] + ["..."] + [strings[-1]]
        else:
            return strings

    def __str__(self) -> str:
        methods_str_to_call = ["Root Hypernyms", "Hypernyms", "Instance Hypernyms",
                            "Hyponyms", "Instance Hyponyms",
                            "Part Holonyms", "Substance Holonyms", "Member Holonyms",
                            "Part Meronyms", "Substance Meronyms", "Member Meronyms",
                            "In Region Domains", "In Topic Domains", "In Usage Domains"]
        str_rep = f"Name: {self.synset.name()}\n"

        # Extract lemmas
        lemma_names = [n.replace("_", " ") for n in self.synset.lemma_names()]
        str_rep += f"Lemma Names: {lemma_names}\n"

        # Print Def/Examples
        str_rep += f"Def: {self.synset.definition()}\n"
        examples = self.synset.examples()
        if len(examples) > 0:
            str_rep += f"Examples: {examples}\n"

        # Print other related words
        for method_str in methods_str_to_call:
            method_exact = method_str.lower().replace(" ", "_")
            synsets = getattr(self.synset, method_exact)()
            if len(synsets) >= 1:
                str_rep += f"{method_str}: {SynsetWrapper.synsets_as_strings(synsets)}\n"

        # Print Upwards Meronyms & Holonyms
        meronyms_upwards = sorted(self.get_all_meronyms(True), key=lambda sw: sw.synset.name())
        holonyms_upwards = sorted(self.get_all_holonyms(True), key=lambda sw: sw.synset.name())
        if len(meronyms_upwards) > 0:
            str_rep += f"Upwards Meronyms: {SynsetWrapper.synsets_as_strings(meronyms_upwards, 20)}\n"
        if len(holonyms_upwards) > 0:
            str_rep += f"Upwards Holonyms: {SynsetWrapper.synsets_as_strings(holonyms_upwards, 20)}\n"

        # Print path to root hypernym with depths
        min, max = self.synset.min_depth(), self.synset.max_depth()
        if min == max: str_rep += f"Depth: {min}\n"
        else: str_rep += f"Min Depth: {min}; Max Depth: {max}\n"
        paths = [SynsetWrapper.synsets_as_strings(x, 20) for x in self.synset.hypernym_paths()]
        str_rep += f"Hypernym Paths: {paths}\n"
        str_rep += "\n"

        return str_rep

class WordNetWrapper():

    def __init__(self):
        self.model = None

    def set_initial_word_list(self, word_list: list | None):
        """
        Parameters:
            word_list: List of SynsetDictEntry objects (or plain synset name strings, e.g.
                "pole.n.06") to build the dictionary from. Pass None to use all WordNet noun
                synsets (no size constraints applied).
        """
        from roman.params.meronomy_graph_params import SynsetDictEntry

        wordnet_emb_path = Path(__file__).resolve().parent / "files" / "synset_features.npy"
        wordnet_word_path = Path(__file__).resolve().parent / "files" / "synset_list.npy"

        # Build a map of synset name -> (min_size, max_size) from the input list
        size_constraints: dict[str, tuple[float | None, float | None]] = {}

        if word_list is None:
            # Use all noun synsets directly; no size constraints
            all_synsets: list[Synset] = list(wn.all_synsets(pos=wn.NOUN))
            synset_set: set[SynsetWrapper] = {SynsetWrapper(s) for s in all_synsets}
        else:
            synset_set: set[SynsetWrapper] = set()
            for item in word_list:
                if isinstance(item, str):
                    name, min_size, max_size = item, None, None
                elif isinstance(item, SynsetDictEntry):
                    name, min_size, max_size = item.name, item.min_size, item.max_size
                else:
                    raise ValueError(f"Unsupported word list entry type: {type(item)}")

                sw = SynsetWrapper(wn.synset(name))
                synset_set.add(sw)
                size_constraints[name] = (min_size, max_size)

                # Ignore for now, seems to add words that only increase spurious synset assignments
                # synset_set.update(sw.get_all_meronyms(True))
                # synset_set.update(sw.get_all_holonyms(True))

        self.synset_list: list[SynsetWrapper] = sorted(
            list(synset_set), key=lambda sw: sw.synset.name())
        self.num_of_synsets: int = len(self.synset_list)

        # Build parallel size-constraint arrays aligned to the sorted synset_list
        self._synset_min_size: list[float | None] = [
            size_constraints.get(sw.synset.name(), (None, None))[0] for sw in self.synset_list
        ]
        self._synset_max_size: list[float | None] = [
            size_constraints.get(sw.synset.name(), (None, None))[1] for sw in self.synset_list
        ]
        logger.debug(f"Final Synset List: {[sw.synset.name() for sw in self.synset_list]}")
        logger.debug(f"Number of Synsets in dictionary: {self.num_of_synsets}")

        # Check if we have access to cupy
        try:
            _ = cp.zeros(1)
            self.use_cupy = True
        except Exception:
            self.use_cupy = False

        # Calculate embeddings if we haven't already
        synset_names = [sw.synset.name() for sw in self.synset_list]
        features_loaded = False
        if wordnet_emb_path.exists() and wordnet_word_path.exists():
            saved_names: list[str] = np.load(str(wordnet_word_path), allow_pickle=True).tolist()
            if saved_names == synset_names:
                self.word_features = np.load(str(wordnet_emb_path))
                features_loaded = True
                logger.info(f"CLIP Features loaded successfully for dictionary")

        if not features_loaded:
            self.word_features = self._calculate_word_embeddings(self.synset_list)
            os.makedirs(os.path.dirname(wordnet_emb_path), exist_ok=True)
            np.save(str(wordnet_emb_path), self.word_features)
            np.save(str(wordnet_word_path), synset_names)
            logger.info(f"Saving New CLIP Features for dictionary")

        # Save word features into CuPy array
        if self.use_cupy:
            self.word_features_cupy = cp.asarray(self.word_features)

    @staticmethod
    def get_prompts(synsets: list[SynsetWrapper]) -> list[str]:
        """Return the CLIP/detection text prompt for each synset (single source of truth)."""
        return [
            f"{sw.synset.lemma_names()[0].replace('_', ' ')}"
            for sw in synsets
        ]

    def _calculate_word_embeddings(self, synsets: list[SynsetWrapper] | list[str]) -> np.ndarray:
        """ Convert WordNet synsets into CLIP embeddings using 'word: definition' prompts """

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.model is None:
            self.model, _ = clip.load("ViT-L/14", device=device)
            self.model.eval()

        batch_size = 1000
        word_features = np.zeros((len(synsets), 768), dtype=np.float16)
        for i in range(int(np.ceil(len(synsets) / batch_size))):
            synset_batch = synsets[i*batch_size:(i+1)*batch_size]
            if isinstance(synsets[0], SynsetWrapper):
                prompts = WordNetWrapper.get_prompts(synset_batch)
            else:
                prompts = synset_batch

            # Tokenize with truncation (CLIP max is 77 tokens)
            tokens = clip.tokenize(prompts, truncate=True).to(device)

            with torch.no_grad():
                text_features = self.model.encode_text(tokens).cpu().numpy()
                text_features /= np.linalg.norm(text_features, axis=1, keepdims=True)
            word_features[i*batch_size:(i+1)*batch_size] = text_features

        return word_features

    def map_embedding_to_synset(self, emb: np.ndarray, min_cos_sim: float = 0.0,
                                object_size: float | None = None) -> SynsetWrapper | None:
        """Returns the best-matching SynsetWrapper within min_cos_sim, or None if none qualify.

        Parameters:
            emb: CLIP embedding to match against the synset dictionary.
            min_cos_sim: Minimum cosine similarity threshold; synsets below this are excluded.
            object_size: Longest-line size of the object in meters. Synsets whose min_size or
                max_size constraints are violated are excluded before finding the best match.
        """
        emb = emb / np.linalg.norm(emb)

        if self.use_cupy:
            similarities = (self.word_features_cupy @ cp.asarray(emb)).get()
        else:
            similarities = self.word_features @ emb

        # Mask out synsets that violate size constraints
        if object_size is not None:
            min_sizes = np.array([s if s is not None else -np.inf for s in self._synset_min_size])
            max_sizes = np.array([s if s is not None else  np.inf for s in self._synset_max_size])
            invalid = (object_size < min_sizes) | (object_size > max_sizes)
            similarities[invalid] = -np.inf

        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])

        if best_sim < min_cos_sim:
            return None
        return self.synset_list[best_idx]

    def get_embedding_for_synset(self, synset: SynsetWrapper) -> np.ndarray:
        """ Returns the CLIP embedding for a synset, computing it on-the-fly if not in the list. """
        try:
            idx = self.synset_list.index(synset)
            return self.word_features[idx]
        except ValueError:
            pass

        feat = self._calculate_word_embeddings([synset])
        self.synset_list.append(synset)
        self._synset_min_size.append(None)
        self._synset_max_size.append(None)
        self.word_features = np.vstack((self.word_features, feat))
        if self.use_cupy:
            self.word_features_cupy = cp.asarray(self.word_features)

        return self.word_features[-1]


def main():
    synset = SynsetWrapper(wn.synset("pole.n.01"))
    print(synset)

if __name__ == "__main__":
    main()