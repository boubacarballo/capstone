from functools import lru_cache
from typing import Optional
import threading

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

from bert_score import BERTScorer
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import sent_tokenize, word_tokenize
from rank_bm25 import BM25Okapi
import nltk
import ssl

try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

_model = SentenceTransformer("all-mpnet-base-v2")

_scorer_lock = threading.Lock()
_scorer_cache = {}


@lru_cache(maxsize=10)
def get_scorer(model_type: str = "roberta-large", device: Optional[str] = None, idf: bool = False, lang: str = "en") -> BERTScorer:
    """Get or create a cached BERTScorer instance with thread safety."""
    cache_key = (model_type, device, idf, lang)
    with _scorer_lock:
        if cache_key not in _scorer_cache:
            _scorer_cache[cache_key] = BERTScorer(
                model_type=model_type,
                device=device,
                idf=idf,
                rescale_with_baseline=True,
                lang=lang,
            )
        return _scorer_cache[cache_key]


def compute_bert_score(summary: str, ground_truth: str, *, model_type: str = "roberta-large", lang: str = "en"):
    """Compute BERT score using a shared, cached scorer for thread safety."""
    candidates = [summary] if isinstance(summary, str) else summary
    references = [ground_truth] if isinstance(ground_truth, str) else ground_truth

    scorer = get_scorer(model_type=model_type, lang=lang)
    P, R, F1 = scorer.score(candidates, references)
    return F1.mean().item()


def compute_score(agent_summary: str, gt: str, threshold: float = 0.5) -> float:
    """Semantic similarity: cosine sim between agent summary and ground-truth sentences."""
    gt_sents = gt
    summary_sents = agent_summary

    if len(gt_sents) == 0 or len(summary_sents) == 0:
        return 0.0

    gt_embeds = _model.encode(gt_sents, convert_to_tensor=True, show_progress_bar=False)
    summary_embeds = _model.encode(summary_sents, convert_to_tensor=True, show_progress_bar=False)

    sim_matrix = util.cos_sim(summary_embeds, gt_embeds)  # [n_summary, n_gt]
    best_scores_per_fact, _ = torch.max(sim_matrix, dim=1)

    average_raw_score = torch.mean(best_scores_per_fact).item()
    return average_raw_score


def compute_similarity_matrix(agent_summary: str, gt_facts) -> dict:
    """
    Compute the full cosine similarity matrix between each sentence in the
    agent's summary and each ground-truth fact.

    Returns a dict with:
      - matrix: list of lists  (shape [n_summary_sents, n_gt_facts])
      - best_per_fact: best similarity for each GT fact (column-wise max)
      - gt_facts: list of GT fact strings  (column labels)
      - summary_sentences: list of summary sentence strings (row labels)
    """
    if isinstance(gt_facts, str):
        gt_facts = sent_tokenize(gt_facts)

    summary_sents = sent_tokenize(agent_summary) if isinstance(agent_summary, str) else list(agent_summary)
    summary_sents = [s.strip() for s in summary_sents if s.strip()]

    if not summary_sents or not gt_facts:
        return {
            "matrix": [],
            "best_per_fact": [],
            "gt_facts": gt_facts if isinstance(gt_facts, list) else [],
            "summary_sentences": summary_sents,
        }

    gt_embeds = _model.encode(gt_facts, convert_to_tensor=True, show_progress_bar=False)
    summary_embeds = _model.encode(summary_sents, convert_to_tensor=True, show_progress_bar=False)

    sim_matrix = util.cos_sim(summary_embeds, gt_embeds)  # [n_summary_sents, n_gt_facts]
    best_per_fact, _ = torch.max(sim_matrix, dim=0)

    return {
        "matrix": sim_matrix.cpu().tolist(),
        "best_per_fact": best_per_fact.cpu().tolist(),
        "gt_facts": gt_facts,
        "summary_sentences": summary_sents,
    }


# ── NLI (entailment) scoring ─────────────────────────────────────────────────

_DEFAULT_NLI_MODEL = "roberta-large-mnli"
_nli_lock = threading.Lock()
_nli_cache = {}


def _load_nli(model_name: str = _DEFAULT_NLI_MODEL):
    with _nli_lock:
        if model_name in _nli_cache:
            return _nli_cache[model_name]
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device).eval()
        _nli_cache[model_name] = (tokenizer, model, device)
        return tokenizer, model, device


def _nli_entailment_matrix(summary_sents, gt_sents, batch_size=32):
    """Return a numpy array of shape [n_summary, n_gt] with P(entailment)."""
    tokenizer, model, device = _load_nli()

    premises = []
    hypotheses = []
    for s in summary_sents:
        for g in gt_sents:
            premises.append(s)
            hypotheses.append(g)

    entailment_probs = []
    for i in range(0, len(premises), batch_size):
        batch_p = premises[i:i + batch_size]
        batch_h = hypotheses[i:i + batch_size]
        inputs = tokenizer(
            batch_p, batch_h,
            padding=True, truncation=True, max_length=512,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        entailment_probs.extend(probs[:, 2].cpu().tolist())

    n_gt = len(gt_sents)
    matrix = [entailment_probs[i * n_gt:(i + 1) * n_gt] for i in range(len(summary_sents))]
    return np.array(matrix)


def compute_nli_score(agent_summary, gt_facts) -> float:
    """SummaC-ZS style: for each GT fact, find the max P(entailment) across
    all summary sentences, then return the average."""
    if isinstance(gt_facts, str):
        gt_sents = [s.strip() for s in sent_tokenize(gt_facts) if s.strip()]
    else:
        gt_sents = [s.strip() for s in gt_facts if s.strip()]

    if isinstance(agent_summary, str):
        summary_sents = [s.strip() for s in sent_tokenize(agent_summary) if s.strip()]
    else:
        summary_sents = [s.strip() for s in agent_summary if s.strip()]

    if not gt_sents or not summary_sents:
        return 0.0

    mat = _nli_entailment_matrix(summary_sents, gt_sents)
    best_per_gt = mat.max(axis=0)  # column-wise max: best entailment per GT fact
    return float(best_per_gt.mean())


def compute_nli_matrix(agent_summary, gt_facts) -> dict:
    """Full NLI entailment probability matrix, same structure as
    compute_similarity_matrix / compute_bm25_matrix for heatmap compatibility."""
    if isinstance(gt_facts, str):
        gt_facts = sent_tokenize(gt_facts)

    summary_sents = sent_tokenize(agent_summary) if isinstance(agent_summary, str) else list(agent_summary)
    summary_sents = [s.strip() for s in summary_sents if s.strip()]

    if not summary_sents or not gt_facts:
        return {
            "matrix": [],
            "best_per_fact": [],
            "gt_facts": gt_facts if isinstance(gt_facts, list) else [],
            "summary_sentences": summary_sents,
        }

    mat = _nli_entailment_matrix(summary_sents, gt_facts)
    best_per_fact = mat.max(axis=0).tolist()

    return {
        "matrix": mat.tolist(),
        "best_per_fact": best_per_fact,
        "gt_facts": gt_facts,
        "summary_sentences": summary_sents,
    }


# ── BM25 (lexical) scoring ──────────────────────────────────────────────────

def _bm25_score(gt_sents, sum_sents):
    if not gt_sents or not sum_sents:
        return 0.0
    token_sum = [word_tokenize(s.lower()) for s in sum_sents]
    token_gt = [word_tokenize(s.lower()) for s in gt_sents]
    bm25 = BM25Okapi(token_sum)
    best = []
    for q in token_gt:
        scores = bm25.get_scores(q)
        best.append(float(np.max(scores)) if len(scores) else 0.0)
    if not best:
        return 0.0
    normalized = [float(1.0 / (1.0 + np.exp(-score))) for score in best]
    return float(np.mean(normalized))


def compute_bm25_matrix(agent_summary: str, gt_facts) -> dict:
    """
    Compute the full BM25 score matrix between each summary sentence and
    each ground-truth fact, with sigmoid normalization into (0, 1).

    Returns a dict with:
      - matrix: list of lists  (shape [n_summary_sents, n_gt_facts])
      - best_per_fact: best normalized score per GT fact (column-wise max)
      - gt_facts: list of GT fact strings  (column labels)
      - summary_sentences: list of summary sentence strings (row labels)
    """
    if isinstance(gt_facts, str):
        gt_facts = sent_tokenize(gt_facts)

    summary_sents = sent_tokenize(agent_summary) if isinstance(agent_summary, str) else list(agent_summary)
    summary_sents = [s.strip() for s in summary_sents if s.strip()]

    if not summary_sents or not gt_facts:
        return {
            "matrix": [],
            "best_per_fact": [],
            "gt_facts": gt_facts if isinstance(gt_facts, list) else [],
            "summary_sentences": summary_sents,
        }

    token_sum = [word_tokenize(s.lower()) for s in summary_sents]
    token_gt = [word_tokenize(s.lower()) for s in gt_facts]

    bm25 = BM25Okapi(token_sum)

    raw_matrix = []
    for q in token_gt:
        raw_matrix.append(bm25.get_scores(q).tolist())
    # raw_matrix is [n_gt, n_summary] — transpose to [n_summary, n_gt]
    raw_matrix = list(map(list, zip(*raw_matrix)))

    sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
    norm_matrix = [[float(sigmoid(v)) for v in row] for row in raw_matrix]

    best_per_fact = [max(norm_matrix[i][j] for i in range(len(summary_sents)))
                     for j in range(len(gt_facts))]

    return {
        "matrix": norm_matrix,
        "best_per_fact": best_per_fact,
        "gt_facts": gt_facts,
        "summary_sentences": summary_sents,
    }


# ── Combined scorer ─────────────────────────────────────────────────────────

def compute_final_score(agent_summary, ground_truth, alpha: float = 0.7) -> float:
    """
    Combined semantic + lexical similarity metric.

    alpha = weight on semantic similarity (default 0.7)
    final_score = alpha * semantic + (1 - alpha) * lexical
    """
    if isinstance(ground_truth, str):
        gt_sents = [s.strip() for s in sent_tokenize(ground_truth) if s.strip()]
    else:
        gt_sents = [s.strip() for s in ground_truth if s.strip()]

    if isinstance(agent_summary, str):
        sum_sents = [s.strip() for s in sent_tokenize(agent_summary) if s.strip()]
    else:
        sum_sents = [s.strip() for s in agent_summary if s.strip()]

    semantic = compute_score(agent_summary, ground_truth)
    lexical = _bm25_score(gt_sents, sum_sents)
    return alpha * semantic + (1 - alpha) * lexical


# ── Heatmap visualisation ───────────────────────────────────────────────────

def _truncate(text: str, max_chars: int = 80) -> str:
    """Truncate to max_chars on a word boundary, adding ellipsis if needed."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut + " …"


def save_similarity_heatmap(sim_data: dict, save_path, agent_id: str = "", metric_label: str = "Cosine Similarity"):
    """
    Render the similarity matrix as an annotated heatmap and save to disk.

    Parameters
    ----------
    sim_data : dict returned by compute_similarity_matrix() or compute_bm25_matrix()
    save_path : str or Path — output PNG file path
    agent_id : str — used in the plot title
    metric_label : str — label for the colorbar and title
    """
    matrix = sim_data.get("matrix", [])
    gt_facts = sim_data.get("gt_facts", [])
    summary_sents = sim_data.get("summary_sentences", [])

    if not matrix or not gt_facts or not summary_sents:
        return

    mat = np.array(matrix)
    n_rows, n_cols = mat.shape

    row_labels = [f"S{i+1}:  {_truncate(s, 72)}" for i, s in enumerate(summary_sents)]
    col_labels = [f"F{j+1}:  {_truncate(f, 72)}" for j, f in enumerate(gt_facts)]

    cell_h = max(0.7, min(1.2, 10.0 / n_rows))
    cell_w = max(1.4, min(2.2, 18.0 / n_cols))
    fig_w = max(10, n_cols * cell_w + 6)
    fig_h = max(5, n_rows * cell_h + 4)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    cmap = LinearSegmentedColormap.from_list(
        "blue_red", ["#1e3a5f", "#3b7dd8", "#f0f0f0", "#e8644a", "#c0392b"]
    )
    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    for i in range(n_rows):
        for j in range(n_cols):
            val = mat[i, j]
            color = "white" if val < 0.35 or val > 0.75 else "#1a1a2e"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=max(7, min(10, 120 // max(n_rows, n_cols))),
                    color=color, fontweight="bold")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=7.5, color="#c9d1d9")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=7.5, color="#c9d1d9")

    ax.set_xlabel("Ground Truth Facts", fontsize=11, color="#c9d1d9", labelpad=10)
    ax.set_ylabel("Agent Summary Sentences", fontsize=11, color="#c9d1d9", labelpad=10)

    title = f"{metric_label} Matrix — Agent {agent_id}" if agent_id else f"{metric_label} Matrix"
    ax.set_title(title, fontsize=14, color="#ffffff", fontweight="bold", pad=12)

    best = sim_data.get("best_per_fact", [])
    if best:
        subtitle = f"Best-per-fact avg: {np.mean(best):.3f}  |  Covered (>0.5): {sum(1 for b in best if b > 0.5)}/{len(best)}"
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes,
                fontsize=9, color="#8b949e", ha="center", va="bottom")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label(metric_label, fontsize=10, color="#c9d1d9")
    cbar.ax.tick_params(colors="#8b949e", labelsize=8)

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor="#0d1117", edgecolor="none", bbox_inches="tight")
    plt.close(fig)


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ground_truth = """Students meet recruiters from many companies. Recruiters offer internships and full-time jobs.
    Students hand out resumes and practice elevator pitches. Workshops and panels help students prepare for interviews.
    Career counselors guide students to suitable employers. The fair is energetic and full of opportunity."""

    summary = """Students prepared for career fairs by developing resumes, business cards, and elevator pitches.
    Recruiters from consulting, technology, and startup firms conducted pitch practice sessions and attended the fairs.
    University counselors facilitated small group discussions and distributed branded tote bags.
    Recruiters, university counselors, and career counselors connected students with startups and technology firms."""

    result = compute_final_score(summary, ground_truth)
    print(result)
