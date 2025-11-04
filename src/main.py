from typing import Set, List, Tuple, Dict, Optional
from collections import defaultdict, Counter
import math
import pandas as pd
from tqdm import tqdm
from jiwer import wer
from fire import Fire


SMALL_KANA: Set[str] = set(list("ゃゅょぁぃぅぇぉャュョァィゥェォ"))
SOKUON: str = "っッ"
CHOON: str = "ー"
SOW: Set[str] = set(SMALL_KANA) | set(list(SOKUON)) | set(list(CHOON))


def tokenize_mora(kana: str) -> Tuple[List[str], Set[int]]:
    """
    Splits Japanese text into moras while preserving word boundaries.
    Mora tokenizer that:
    - treats combined kana like きゃ/きゅ/きょ as one mora (small kana following),
    - treats small "っ" (sokuon) as its own mora (but may merge with following mora in phoneme mapping),
    - treats "ー" (chōon) as part of previous mora,
    - treats "ん" as a mora.

    Args:
        kana (str): Input string in kana with optional spaces between words

    Returns:
        tuple: (list of moras, set of word boundary indices)
        - list of moras: each element is a single mora
        - word boundaries: indices in the mora list after which a word boundary exists
    """
    kana = kana.strip()
    words = kana.split()
    mora_list = []
    word_boundaries = []  # indices in mora_list after which a word boundary exists
    idx = 0
    for w_i, w in enumerate(words):
        i = 0
        local_moras = []
        while i < len(w):
            c = w[i]
            # if small kana follows, merge
            if i + 1 < len(w) and w[i + 1] in SMALL_KANA:
                local_moras.append(w[i : i + 2])
                i += 2
            elif c in CHOON:
                # attach to previous mora if exists, else standalone
                if local_moras:
                    local_moras[-1] = local_moras[-1] + c
                else:
                    local_moras.append(c)
                i += 1
            else:
                local_moras.append(c)
                i += 1
        mora_list.extend(local_moras)
        idx += len(local_moras)
        # mark boundary after this word (unless last)
        if w_i != len(words) - 1:
            word_boundaries.append(idx)  # boundary is after index idx-1 (0-based)
    return mora_list, set(word_boundaries)


def viterbi_align(
    moras: List[str],
    phonemes: List[str],
    probs: Dict[str, Dict[Tuple[str, ...], float]],
    max_ph_len: int = 6,
) -> Optional[List[Tuple[str, ...]]]:
    """
    Finds optimal alignment between moras and phonemes using Viterbi algorithm.

    Args:
        moras (list): List of mora tokens
        phonemes (list): List of phoneme tokens
        probs (dict): Probability mapping mora -> {phoneme sequence -> probability}
        max_ph_len (int): Maximum length of phoneme sequence for a mora

    Returns:
        list: List of phoneme tuples aligned with each mora, or None if alignment impossible
    """
    N = len(moras)
    M = len(phonemes)
    # dp[pos_mora][pos_ph] = (logprob, backpointer_len)
    NEG_INF = -1e9
    dp = [[(NEG_INF, None) for _ in range(M + 1)] for _ in range(N + 1)]
    dp[0][0] = (0.0, None)

    for i in range(N):
        m = moras[i]
        for p in range(M + 1):
            if dp[i][p][0] <= NEG_INF / 2:
                continue
            base_logp = dp[i][p][0]
            # try to assign length l from 1..max_ph_len such that p+l <= M
            for l in range(1, max_ph_len + 1):
                if p + l > M:
                    break
                subseq = tuple(phonemes[p : p + l])
                # get prob of m -> subseq
                prob = probs.get(m, {}).get(subseq, None)
                if prob is None:
                    # backoff: if unseen, use small epsilon
                    prob = 1e-8
                logp = math.log(prob)
                newp = base_logp + logp
                if newp > dp[i + 1][p + l][0]:
                    dp[i + 1][p + l] = (newp, l)
    # check feasible end
    if dp[N][M][0] <= NEG_INF / 2:
        return None  # no feasible alignment
    # backtrack
    alignment = [None] * N
    p = M
    for i in range(N, 0, -1):
        logp, l = dp[i][p]
        if l is None:
            # should not happen
            return None
        alignment[i - 1] = tuple(phonemes[p - l : p])
        p = p - l
    return alignment


def em_train(
    pairs: List[Tuple[str, List[str]]],
    max_iters: int = 30,
    max_ph_len: int = 6,
    smoothing: float = 0.1,
) -> Dict[str, Dict[Tuple[str, ...], float]]:
    """
    Trains mora-to-phoneme alignment model using EM algorithm.

    Args:
        pairs (list): List of (kana_with_spaces, phoneme_list) pairs
        max_iters (int): Maximum number of EM iterations
        max_ph_len (int): Maximum length of phoneme sequence for a mora
        smoothing (float): Smoothing parameter for probabilities

    Returns:
        dict: Trained probability mapping mora -> {phoneme sequence -> probability}
    """
    mora_set = set()
    for kana, _ in pairs:
        moras, _ = tokenize_mora(kana)
        mora_set.update(moras)

    probs = defaultdict(lambda: defaultdict(float))

    # but better init: collect candidate subseqs from data with naive proportional split
    candidate_counts = defaultdict(Counter)
    for kana, phonemes in pairs:
        moras, _ = tokenize_mora(kana)
        if len(moras) == 0:
            continue
        # naive even split to get candidates: distribute roughly M/len(moras) phonemes to each
        M = len(phonemes)
        avg = max(1, int(round(M / len(moras))))
        p = 0
        for i, m in enumerate(moras):
            take = avg
            # adjust for last
            if i == len(moras) - 1:
                take = M - p
            subseq = tuple(phonemes[p : p + take]) if take > 0 else tuple()
            candidate_counts[m][subseq] += 1
            p += take

    # seed probs from candidate_counts
    for m, counter in candidate_counts.items():
        total = sum(counter.values())
        probs[m] = {}
        for subseq, c in counter.items():
            probs[m][subseq] = (c + 1e-3) / (total + 1e-3 * len(counter))

    # EM iterations
    for it in tqdm(range(max_iters)):
        counts = defaultdict(Counter)
        total_alignments = 0
        no_align = 0
        for kana, phonemes in pairs:
            moras, word_boundaries = tokenize_mora(kana)
            # try Viterbi alignment under current probs
            alignment = viterbi_align(moras, phonemes, probs, max_ph_len=max_ph_len)
            if alignment is None:
                # fallback: naive even split
                M = len(phonemes)
                if len(moras) == 0:
                    continue
                avg = max(1, int(round(M / len(moras))))
                p = 0
                alignment = []
                for i, m in enumerate(moras):
                    take = avg
                    if i == len(moras) - 1:
                        take = M - p
                    alignment.append(tuple(phonemes[p : p + take]))
                    p += take
                no_align += 1
            # accumulate counts
            for m, subseq in zip(moras, alignment):
                counts[m][subseq] += 1.0
                total_alignments += 1
        # M-step: update probs with smoothing
        max_change = 0.0
        for m, cnts in counts.items():
            denom = sum(cnts.values()) + smoothing * (len(cnts) + 1)
            newd = {}
            for subseq, c in cnts.items():
                newd[subseq] = (c + smoothing) / denom
            # also keep prior unseen short sequences with tiny prob
            # compute change
            old_map = probs.get(m, {})
            # measure total variation distance roughly
            old_keys = set(old_map.keys())
            new_keys = set(newd.keys())
            # include an epsilon for previously-known but now-absent
            tv = 0.0
            all_keys = old_keys.union(new_keys)
            for k in all_keys:
                oldv = old_map.get(k, 0.0)
                newv = newd.get(k, 0.0)
                tv += abs(oldv - newv)
            max_change = max(max_change, tv)
            probs[m] = newd

        if max_change < 1e-5:
            break
    return probs


def segment_phonemes_by_kana(
    kana_with_spaces: str,
    phonemes: List[str],
    probs: Dict[str, Dict[Tuple[str, ...], float]],
    max_ph_len: int = 6,
) -> List[str]:
    """
    Segments phoneme sequence according to kana mora boundaries.

    Args:
        kana_with_spaces (str): Japanese text in kana with spaces between words
        phonemes (list): List of phonemes to segment
        probs (dict): Probability mapping mora -> {phoneme sequence -> probability}
        max_ph_len (int): Maximum length of phoneme sequence for a mora

    Returns:
        list: List of segmented phonemes with added space tokens at word boundaries
    """
    moras, word_boundaries = tokenize_mora(kana_with_spaces)
    alignment = viterbi_align(moras, phonemes, probs, max_ph_len=max_ph_len)
    if alignment is None:
        # fallback naive split
        M = len(phonemes)
        if len(moras) == 0:
            return phonemes
        avg = max(1, int(round(M / len(moras))))
        p = 0
        alignment = []
        for i, m in enumerate(moras):
            take = avg
            if i == len(moras) - 1:
                take = M - p
            alignment.append(tuple(phonemes[p : p + take]))
            p += take

    # insert spaces where kana word boundaries are
    segmented = []
    p_pos = 0
    for i, block in enumerate(alignment):
        # append phonemes in block
        for ph in block:
            segmented.append(ph)
        p_pos += len(block)
        # if there was a word boundary after mora i, insert space token
        # recall word_boundaries set holds mora indices after which boundary exists (1-based index count of moras)
        if (i + 1) in word_boundaries:
            segmented.append("<space>")
    return segmented


def main() -> None:
    train = pd.read_csv("../dataset/train.csv")
    test = pd.read_csv("../dataset/test.csv")
    trainpairs = [
        (i, j.split())
        for i, j in zip(
            train["split_phrase"].tolist() + test["split_phrase"].tolist(),
            train["ipa"].tolist() + test["ipa"].tolist(),
        )
    ]
    testpairs = [
        (i, j.split(), k.split())
        for i, j, k in zip(
            train["split_phrase"].tolist(),
            train["ipa"].tolist(),
            train["split_ipa"].tolist(),
        )
    ]

    probs = em_train(trainpairs, max_iters=50, max_ph_len=4, smoothing=0.5)

    num_eq = 0.0
    num_all = len(testpairs)
    ref = []
    hyp = []
    for kana, phon, true in testpairs:
        segmented = segment_phonemes_by_kana(kana, phon, probs, max_ph_len=4)
        if segmented == true:
            num_eq += 1.0
        print("KANA:", kana)
        print("PHON (raw):", " ".join(phon))
        print("PHON (segmented):", " ".join(segmented))
        print("PHON (true):", " ".join(true))
        print("Equal? -> ", segmented == true)
        print("-----------------------------------\n")
        ref.append(" ".join(true))
        hyp.append(" ".join(segmented))

    print("WER on test set:", wer(ref, hyp))
    print("accuracy on test set:", num_eq / num_all)


def submission_gen(save2dir="../dataset") -> None:
    train = pd.read_csv("../dataset/train.csv")
    test = pd.read_csv("../dataset/test.csv")
    trainpairs = [
        (i, j.split())
        for i, j in zip(
            train["split_phrase"].tolist(),
            train["ipa"].tolist(),
        )
    ]
    testpairs = [
        (i, j.split())
        for i, j in zip(
            test["split_phrase"].tolist(),
            test["ipa"].tolist(),
        )
    ]

    probs = em_train(trainpairs, max_iters=50, max_ph_len=4, smoothing=0.5)
    segmented = []
    for kana, phon in testpairs:
        segmented.append(" ".join(segment_phonemes_by_kana(kana, phon, probs, max_ph_len=4)))
    
    ans = pd.DataFrame({'split_ipa': segmented})
    ans.to_csv(save2dir+"/submission2.csv")


def router(mode: str = "main", save2dir: str = "../dataset") -> None:
    if mode == "main":
        main()
    elif mode == "submission":
        submission_gen(save2dir=save2dir)


if __name__ == "__main__":
    Fire(router)
