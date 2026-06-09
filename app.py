"""
============================================================================
 HMM Part-of-Speech Tagger — Web Client (SINGLE FILE, no other files needed)
============================================================================
 Everything is in this one file: the Hidden Markov Model built from scratch,
 the improved unknown-word handling, and the Flask web server.

 How to run (in Git Bash / terminal, inside this file's folder):
     pip install flask nltk
     python app.py
 Then open the link it prints:  http://127.0.0.1:5000
============================================================================
"""

import math
import random
import re
from collections import defaultdict, Counter

from flask import Flask, request, jsonify, render_template_string
import json

import nltk
nltk.download("brown", quiet=True)
nltk.download("universal_tagset", quiet=True)
from nltk.corpus import brown


# ===========================================================================
# 1) DATA: load the Brown Corpus and split 80% train / 20% test
# ===========================================================================
def load_and_split(test_ratio=0.2, seed=42):
    sentences = list(brown.tagged_sents(tagset="universal"))
    random.seed(seed)
    random.shuffle(sentences)
    cut = int(len(sentences) * (1 - test_ratio))
    return sentences[:cut], sentences[cut:]


# ===========================================================================
# 2) THE HMM (built from scratch) with improved unknown-word handling
# ===========================================================================
EPS = 1e-12
CLOSED_CLASS = {"DET", "PRON", "ADP", "CONJ", "PRT", "."}


class HMM:
    """First-order HMM POS tagger: pi, A, B estimated by counting, Laplace
    smoothing for unseen words, and a log-space Viterbi decoder.  Unknown words
    are tagged using suffix + capitalization clues learned from rare words."""

    def __init__(self, alpha=0.01, rare_threshold=10, max_suffix=3):
        self.alpha = alpha
        self.rare_threshold = rare_threshold
        self.max_suffix = max_suffix

    # -- Training ---------------------------------------------------------
    def train(self, train_data):
        tag_count = Counter()
        initial_count = Counter()
        transition_count = defaultdict(Counter)
        emission_count = defaultdict(Counter)
        self.vocab = set()

        for sentence in train_data:
            prev = None
            for i, (word, tag) in enumerate(sentence):
                tag_count[tag] += 1
                emission_count[tag][word] += 1
                self.vocab.add(word)
                if i == 0:
                    initial_count[tag] += 1
                else:
                    transition_count[prev][tag] += 1
                prev = tag

        self.tags = sorted(tag_count)
        V = len(self.vocab)
        n_sent = len(train_data)
        T = len(self.tags)
        a = self.alpha

        # pi : P(tag starts a sentence)
        self.pi = {t: initial_count[t] / n_sent for t in self.tags}

        # A : P(t_j | t_i)  (light smoothing over the small tag set)
        self.A = {}
        for ti in self.tags:
            total = sum(transition_count[ti].values())
            self.A[ti] = {tj: (transition_count[ti][tj] + a) / (total + a * T)
                          for tj in self.tags}

        # B : P(word | tag) with Laplace smoothing + an "<OOV>" key
        self.B = {}
        for t in self.tags:
            denom = tag_count[t] + a * V
            d = {w: (c + a) / denom for w, c in emission_count[t].items()}
            d["<OOV>"] = a / denom
            self.B[t] = d

        # ---- learn suffix + capitalization stats from RARE words ----
        word_total = Counter()
        for s in train_data:
            for w, _ in s:
                word_total[w] += 1

        self.suffix_count = {L: defaultdict(Counter) for L in range(1, self.max_suffix + 1)}
        self.suffix_total = {L: Counter() for L in range(1, self.max_suffix + 1)}
        self.suffix_vocab = {L: set() for L in range(1, self.max_suffix + 1)}
        self.cap_count, self.rare_total = Counter(), Counter()

        for s in train_data:
            for w, t in s:
                if word_total[w] > self.rare_threshold:
                    continue
                lw = w.lower()
                self.rare_total[t] += 1
                if w[:1].isupper():
                    self.cap_count[t] += 1
                for L in range(1, self.max_suffix + 1):
                    if len(lw) >= L:
                        suf = lw[-L:]
                        self.suffix_count[L][t][suf] += 1
                        self.suffix_total[L][t] += 1
                        self.suffix_vocab[L].add(suf)

    # -- Emission probability P(word | tag) -------------------------------
    def emission_prob(self, word, tag):
        if word in self.vocab:                       # known word
            return self.B[tag].get(word, self.B[tag]["<OOV>"])
        if tag in CLOSED_CLASS:                       # unknowns aren't closed-class
            return 1e-10
        lw = word.lower()                             # suffix backoff: 3 -> 2 -> 1
        p_suffix = None
        for L in range(min(self.max_suffix, len(lw)), 0, -1):
            s = lw[-L:]
            if s in self.suffix_vocab[L]:
                num = self.suffix_count[L][tag][s] + self.alpha
                den = self.suffix_total[L][tag] + self.alpha * len(self.suffix_vocab[L])
                p_suffix = num / den
                break
        if p_suffix is None:
            return self.B[tag]["<OOV>"]
        cap_p = (self.cap_count[tag] + self.alpha) / (self.rare_total[tag] + 2 * self.alpha)
        cap_factor = cap_p if word[:1].isupper() else (1.0 - cap_p)
        return p_suffix * cap_factor

    # -- Log-space Viterbi decoder ---------------------------------------
    def log_viterbi(self, sentence):
        n = len(sentence)
        if n == 0:
            return []
        log = lambda p: math.log(p + EPS)
        viterbi = [dict() for _ in range(n)]
        backptr = [dict() for _ in range(n)]

        w0 = sentence[0]
        for tag in self.tags:
            viterbi[0][tag] = log(self.pi[tag]) + log(self.emission_prob(w0, tag))
            backptr[0][tag] = None

        for t in range(1, n):
            word = sentence[t]
            for tj in self.tags:
                best_prev, best_score = None, -math.inf
                for ti in self.tags:
                    sc = viterbi[t - 1][ti] + log(self.A[ti][tj])
                    if sc > best_score:
                        best_score, best_prev = sc, ti
                viterbi[t][tj] = best_score + log(self.emission_prob(word, tj))
                backptr[t][tj] = best_prev

        last = max(self.tags, key=lambda tag: viterbi[n - 1][tag])
        path = [last]
        for t in range(n - 1, 0, -1):
            last = backptr[t][last]
            path.append(last)
        path.reverse()
        return path


def tokenize(text):
    """Split text into word/punctuation tokens, AND expand contractions into
    full words first, because the Brown Corpus does not store forms like "'m".

      "I'm not sure!"  ->  ['I', 'am', 'not', 'sure', '!']
      "don't"          ->  ['do', 'not']
      "we're happy"    ->  ['we', 'are', 'happy']
    """
    # Whole-word contractions that need a special expansion.
    SPECIAL = {
        "won't": "will not", "can't": "can not", "cannot": "can not",
        "shan't": "shall not", "ain't": "is not", "let's": "let us",
        "it's": "it is", "he's": "he is", "she's": "she is",
        "that's": "that is", "there's": "there is", "what's": "what is",
        "who's": "who is", "here's": "here is",
    }
    # Generic suffixes (apply after the special cases above).
    SUFFIX = {
        "n't": " not", "'re": " are", "'ve": " have",
        "'ll": " will", "'m": " am", "'d": " would",
    }

    out_words = []
    for token in text.split():
        low = token.lower()
        if low in SPECIAL:
            replacement = SPECIAL[low]
        else:
            replacement = token
            for suf, full in SUFFIX.items():
                if low.endswith(suf):
                    replacement = token[: -len(suf)] + full
                    break
        out_words.append(replacement)
    text = " ".join(out_words)

    # Now split words from the remaining punctuation.
    tokens = re.findall(r"\w+|[^\w\s]", text)
    # The pronoun "I" is always capitalized in English (Brown tags "I" as PRON,
    # but lowercase "i" is unknown), so normalize a lone "i" to "I".
    tokens = ["I" if t == "i" else t for t in tokens]
    return tokens


# ===========================================================================
# 3) TRAIN THE MODEL ONCE AT STARTUP
# ===========================================================================
print("Training the HMM model (a few seconds)...")
_train_data, _ = load_and_split()
MODEL = HMM(alpha=0.01)
MODEL.train(_train_data)
print("Model ready. Open http://127.0.0.1:5000 in your browser.")

# Each tag maps to [short name, description-with-example shown on hover].
TAG_INFO = {
    "NOUN": ["Noun", "A person, place, thing or idea \u2014 e.g. dog, Cambodia, idea"],
    "VERB": ["Verb", "An action or state \u2014 e.g. run, is, loves"],
    "ADJ":  ["Adjective", "Describes a noun \u2014 e.g. quick, blue, happy"],
    "ADV":  ["Adverb", "Describes a verb or adjective \u2014 e.g. quickly, very, often"],
    "PRON": ["Pronoun", "Replaces a noun \u2014 e.g. I, he, they, it"],
    "DET":  ["Determiner", "Introduces a noun \u2014 e.g. the, a, this, her"],
    "ADP":  ["Preposition", "Shows relation or position \u2014 e.g. in, on, of, with"],
    "CONJ": ["Conjunction", "Joins words or clauses \u2014 e.g. and, but, or"],
    "NUM":  ["Number", "A numeral \u2014 e.g. one, 2025, third"],
    "PRT":  ["Particle", "Small function word \u2014 e.g. to (to go), not, 's"],
    "X":    ["Other", "Foreign words, symbols or unclear tokens"],
    ".":    ["Punctuation", "All punctuation marks: . , ? ! : ; ' \" ( ) - \u2014"],
}


# ===========================================================================
# 4) THE WEB APP
# ===========================================================================
app = Flask(__name__)

PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HMM POS Tagger</title>
<style>
  :root { --bg:#0f172a; --card:#1e293b; --text:#e2e8f0; --muted:#94a3b8; --accent:#38bdf8; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         background: var(--bg); color: var(--text); min-height:100vh; padding:32px 16px; }
  .wrap { max-width: 860px; margin: 0 auto; }
  h1 { font-size: 1.6rem; margin: 0 0 4px; }
  p.sub { color: var(--muted); margin: 0 0 24px; }
  .card { background: var(--card); border-radius: 14px; padding: 20px; margin-bottom: 20px;
          box-shadow: 0 8px 24px rgba(0,0,0,.25); }
  .row { display:flex; gap:10px; }
  input[type=text] { flex:1; padding:14px 16px; border-radius:10px; border:1px solid #334155;
        background:#0b1220; color:var(--text); font-size:1rem; outline:none; }
  input[type=text]:focus { border-color: var(--accent); }
  button { padding:14px 22px; border:0; border-radius:10px; background:var(--accent);
        color:#04293a; font-weight:700; font-size:1rem; cursor:pointer; }
  button:hover { filter:brightness(1.08); }
  .examples { margin-top:10px; color:var(--muted); font-size:.9rem; }
  .examples span { cursor:pointer; color:var(--accent); margin-right:14px; text-decoration:underline; }
  .tokens { display:flex; flex-wrap:wrap; gap:10px; margin-top:6px; }
  .token { display:flex; flex-direction:column; align-items:center; background:#0b1220;
        border-radius:10px; padding:10px 12px; min-width:60px; }
  .token .w { font-weight:600; font-size:1.05rem; }
  .token .t { margin-top:6px; font-size:.72rem; font-weight:700; padding:3px 8px;
        border-radius:999px; letter-spacing:.04em; }
  .token .sub { margin-top:5px; font-size:.68rem; color:var(--muted); text-align:center; }
  .t.NOUN{background:#1d4ed8;color:#dbeafe}.t.VERB{background:#b91c1c;color:#fee2e2}
  .t.ADJ{background:#15803d;color:#dcfce7}.t.ADV{background:#a16207;color:#fef9c3}
  .t.PRON{background:#7c3aed;color:#ede9fe}.t.DET{background:#0e7490;color:#cffafe}
  .t.ADP{background:#be185d;color:#fce7f3}.t.CONJ{background:#4338ca;color:#e0e7ff}
  .t.NUM{background:#0f766e;color:#ccfbf1}.t.PRT{background:#9333ea;color:#f3e8ff}
  .t.X{background:#475569;color:#e2e8f0}.t[class~="."]{background:#334155;color:#cbd5e1}
  .legend { display:flex; flex-wrap:wrap; gap:8px; font-size:.75rem; color:var(--muted); }
  .legend b { color: var(--text); }
</style>
</head>
<body>
<div class="wrap">
  <h1>HMM Part-of-Speech Tagger</h1>
  <p class="sub">Built from scratch (Hidden Markov Model + log-space Viterbi). Type a sentence and the model predicts the tag for each word.</p>

  <div class="card">
    <div class="row">
      <input id="sentence" type="text" placeholder="e.g. my watch is broken" autofocus>
      <button onclick="predict()">Tag it</button>
    </div>
    <div class="examples">
      Try:
      <span onclick="setEx('my watch is broken')">my watch is broken</span>
      <span onclick="setEx('watch the dog run')">watch the dog run</span>
      <span onclick="setEx('the quick brown fox jumps over the lazy dog')">the quick brown fox...</span>
    </div>
  </div>

  <div class="card" id="result" style="display:none;">
    <div class="tokens" id="tokens"></div>
  </div>

  <div class="card">
    <div class="legend" id="legend"></div>
  </div>
</div>

<script>
const TAG_INFO = {{ tag_info | safe }};
const legend = document.getElementById('legend');
for (const [tag, info] of Object.entries(TAG_INFO)) {
  const name = info[0], desc = info[1];
  const el = document.createElement('span');
  el.setAttribute('title', desc);          // hover to see description + example
  el.style.cursor = 'help';
  el.innerHTML = '<span class="t ' + tag + '">' + tag + '</span> <b>' + name + '</b>';
  legend.appendChild(el);
}
function setEx(s){ document.getElementById('sentence').value = s; predict(); }

// The model tags ALL punctuation as ".". This map identifies the specific
// mark from the character itself (a simple, exact lookup).
const PUNCT_NAMES = {
  '.': 'Period', ',': 'Comma', '?': 'Question Mark', '!': 'Exclamation Mark',
  ':': 'Colon', ';': 'Semicolon', "'": 'Apostrophe', '\u2019': 'Apostrophe',
  '"': 'Quotation Mark', '\u201c': 'Quotation Mark', '\u201d': 'Quotation Mark',
  '`': 'Quotation Mark', '(': 'Parenthesis', ')': 'Parenthesis',
  '[': 'Bracket', ']': 'Bracket', '{': 'Brace', '}': 'Brace',
  '-': 'Hyphen', '\u2013': 'Dash', '\u2014': 'Dash', '/': 'Slash'
};

function esc(s){ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

async function predict() {
  const text = document.getElementById('sentence').value.trim();
  if (!text) return;
  const res = await fetch('/predict', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ sentence: text })
  });
  const data = await res.json();
  const box = document.getElementById('tokens');
  box.innerHTML = '';
  data.result.forEach(([word, tag]) => {
    const t = document.createElement('div');
    t.className = 'token';
    let extra = '';
    if (tag === '.') {                          // punctuation -> name the mark
      const name = PUNCT_NAMES[word] || 'Punctuation';
      extra = '<span class="sub">' + name + '</span>';
    }
    t.innerHTML = '<span class="w">' + esc(word) + '</span>' +
                  '<span class="t ' + tag + '">' + tag + '</span>' + extra;
    box.appendChild(t);
  });
  document.getElementById('result').style.display = 'block';
}
document.getElementById('sentence').addEventListener('keydown', e => {
  if (e.key === 'Enter') predict();
});
</script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(PAGE, tag_info=json.dumps(TAG_INFO))


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    sentence = (data or {}).get("sentence", "")
    words = tokenize(sentence)
    tags = MODEL.log_viterbi(words)
    return jsonify({"result": list(zip(words, tags))})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)