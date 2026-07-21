# %% [markdown]
# # Lab 3b: Text Embeddings and Semantic Search
#
# ICPSR 2026 — The Social Science Data Pipeline
# Instructor: Andy Halterman
#
# This lab puts the morning's chapter into practice. The plan: turn text into
# vectors of numbers with a real embedding model, and then similarity, search, and classification are all
# ordinary numpy and sklearn on those numbers.
#
# A note on the data, since it's a detour. We spent Day 1 on Ukraine, but
# for the embeddings lab I'm using a dataset of Indian news articles instead:
# the **India Police Events** corpus, which I've worked with extensively so I know
# the data well. 
#
# This is also, for some of you, only the second time writing Python. I've
# tried to keep it gentle: we look at every object before we use it, and I
# explain the Python bits (methods, attributes, slicing, loops) as they come
# up. When in doubt, you should print() and type() everything before you try 
# to work with it.
#
# By the end you'll have: embeddings you computed yourself, a semantic search
# engine over ~1,300 news articles in a handful of lines, a document
# classifier, and a bake-off between embeddings and 1995-era
# bag-of-words that shows that modern methods aren't always better!

# %%
# Setup (this same block opens every lab this week)
import os

def is_colab():
    """Detect if running in Google Colab"""
    try:
        import google.colab
        return True
    except ImportError:
        return False

IN_COLAB = is_colab()
print(f"Environment detected: {'Colab' if IN_COLAB else 'Local/hosted Jupyter'}")

if IN_COLAB:
    # On Colab, grab the course repo (cached data included) if it isn't there.
    if not os.path.exists("/content/icpsr-pipeline"):
        !git clone -q https://github.com/ahalterman/icpsr-pipeline.git /content/icpsr-pipeline
    COURSE_DIR = "/content/icpsr-pipeline"
else:
    # Hosted Jupyter / local: assume we're inside the repo.
    COURSE_DIR = os.path.dirname(os.getcwd()) if os.path.basename(os.getcwd()) in ("labs", "solutions") else os.getcwd()

DATA_DIR = os.path.join(COURSE_DIR, "data", "cached")
OUTPUTS_DIR = os.path.join(COURSE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
print(f"Course dir: {COURSE_DIR}")

# %% [markdown]
# The one install that matters today is `sentence-transformers`, the library
# that runs embedding models. It pulls in `torch` (the CPU build is all we
# need). On a fresh machine this takes a minute or two.

# %%
!pip install -q sentence-transformers scikit-learn pandas numpy matplotlib

# %% [markdown]
# ## Step 1: Get the model
#
# An **embedding model** takes a piece of text and returns a list of numbers
# (a *vector*) that represents its meaning. We'll use `all-MiniLM-L6-v2`, the
# sentence transformer from the chapter: it's small (~90 MB) and CPU-friendly, 
# but keep in mind that this is probably a *floor* on the performance you'd 
# expect from embedding models.
#
# The first line downloads the model the first time you run it (that ~90 MB),
# then loads it. After that it's on your machine and loading is instant (until
# the kernel restarts).

# %%
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Model loaded.")

# %% [markdown]
# Now let's actually use it on one sentence. A model in Python is an *object*,
# and objects have **methods** (things they can *do*, written with parentheses
# like `model.encode(...)`) and **attributes** (things they *are*, written
# without parentheses). `encode` is the method that turns text into a vector.

# %%
one_vector = model.encode("Police arrested dozens of protesters in the city.")

print("Type of what came back:", type(one_vector))
print("Shape:", one_vector.shape)      # .shape is an ATTRIBUTE: no parentheses
print("First 8 numbers:", one_vector[:8].round(3))

# %% [markdown]
# Three things to notice, which you saw yesterday in a pretty intense first exposure to python:
#
# - **`type(...)`** tells you it's a numpy `ndarray` — an array of numbers,
#   the standard data type for this kind of data.
# - **`.shape`** is `(384,)`: a single row of 384 numbers. Shape is how you
#   check what you're working with *before* you try to compute with it. Looking
#    at the shape is a greathabit to get into.
# - **`one_vector[:8]`** is *slicing*: "give me elements 0 through 7." Square
#   brackets index into an array; `[:8]` means "from the start up to (not
#   including) position 8." We only print 8 because 384 numbers is a lot to look at
#   and there's nothing very human readable in it (there's no "arrest"
#   column). The *meaning* lives in the whole vector's direction, which is
#   what the next step is about.

# %% [markdown]
# `encode` also takes a **list** of texts and embeds all of them at once,
# which is how you'll normally call it. A list is written with square brackets
# and commas: `["first text", "second text"]`.

# %%
few = model.encode([
    "Police arrested dozens of protesters in the city.",
    "Officers detained many demonstrators downtown.",
    "The cricket team won the national championship.",
])
print("Shape now:", few.shape)   # (3, 384): three rows, 384 numbers each

# %% [markdown]
# `(3, 384)` is a 2-D array: **3 rows** (one per text) and **384 columns**.
# `few[0]` is the first text's vector, `few[1]` the second, and so on. The
# first two sentences say nearly the same thing in different words; the third
# is about something else entirely. In a moment we'll measure that.

# %% [markdown]
# ## Step 2: Two texts, one number
#
# The standard way to measure how similar two vectors are is **cosine
# similarity**: scale each vector to length 1, then take their dot product. It
# runs from about 0 (unrelated) up to 1 (same direction, i.e. same meaning).
# It's one short function.

# %%
import numpy as np

def cosine(a, b):
    """Cosine similarity between vector `a` and vector-or-matrix `b`."""
    a = a / np.linalg.norm(a)                       # scale a to length 1
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)   # scale each row of b
    return b @ a                                    # dot product(s)

# A sentence compared with itself is exactly 1.0:
print("self-similarity:", cosine(few[0], few[0]).round(3))

# %% [markdown]
# Now the two sentences that *mean* the same thing but *share few words*
# ("arrested dozens of protesters" vs. "detained many demonstrators"), against
# the unrelated cricket sentence.

# %%
print("arrest vs. detain  (same meaning):   ", cosine(few[0], few[1]).round(3))
print("arrest vs. cricket (unrelated):      ", cosine(few[0], few[2]).round(3))

# %% [markdown]
# The paraphrase pair scores high even though the words barely overlap — a
# bag-of-words model would score it near zero, because "arrested" and
# "detained" are just different words to it. This is what embeddings are for!
#
# But here's a cautionary pair. 
#
# - "The government must expand offshore wind subsidies."
# - "The government must end offshore wind subsidies."
#
# How similar are these documents?

# %%
stance = model.encode([
    "The government must expand offshore wind subsidies.",
    "The government must end offshore wind subsidies.",
])
print("expand vs. end (opposite stance):", cosine(stance[0], stance[1]).round(3))

# %% [markdown]
# The two sentences are about the same thing in almost the same words,
# and cosine similarity is tracking topic and phrasing, not stance.
# Similarity on this specific emebdding model measures **aboutness**, not stance or agreement,
# Keep this in mind every time you're reading papers that don't discuss the embedding similarity critically.

# %% [markdown]
# ## Step 3: The corpus
#
# Now some real text. The **India Police Events** corpus is 1,257 news
# articles from the *Times of India* around the 2002 Gujarat violence, each
# hand-labeled for which kinds of police activity it reports (Halterman et
# al. 2021). The labels come from a codebook with five event types:
#
# - `ARREST` — police arrest or detain people
# - `KILL` — police kill someone
# - `FORCE` — police use force (baton charges, tear gas, firing)
# - `FAIL` — police fail to act against violence
# - `ANY_ACTION` — the article reports *any* police activity at all
#
# Each article can carry several labels, or none. We load the file with pandas
# and, as always, **look at the data before doing anything to it.**

# %%
import pandas as pd

df = pd.read_json(os.path.join(DATA_DIR, "india_police_events.jsonl"), lines=True)
print("Shape:", df.shape)          # (rows, columns)
df.head(3)

# %% [markdown]
# Three columns: `doc_id`, `doc_text` (the article), and `doc_labels` (a list
# of the event types in that article). Read a few articles in full to get a sense
# of what's in them. (Reading the docs is a superpower). Rerun this cell a couple of times to see
# different ones.

# %%
for _, row in df.sample(3).iterrows():
    print("labels:", row["doc_labels"])
    print(row["doc_text"][:400], "...\n")

# %% [markdown]
# The `doc_labels` column holds a Python **list** per row. To model these, it's
# easiest to turn each event type into its own 0/1 column ("does this article
# have this label?"). The loop below does that for the five event types.

# %%
event_types = ["ANY_ACTION", "ARREST", "KILL", "FORCE", "FAIL"]

for event in event_types:
    # For each row, ask: is `event` in that row's list of labels? -> 1 or 0.
    df[event] = df["doc_labels"].apply(lambda labels: 1 if event in labels else 0)

# How common is each label? (.mean() of a 0/1 column is the share of 1s.)
print("Share of articles with each label:")
for event in event_types:
    print(f"  {event:11s} {df[event].mean():.0%}   ({df[event].sum()} articles)")

# %% [markdown]
# Notice the different base rates: about a third of articles report *some* police
# action, but `KILL` shows up in only 4%. That imbalance is normal for event
# data (most documents are *not* about your rare event) and that's why
# we'll measure with **F1** rather than plain accuracy: a classifier
# that predicts "no KILL" every time is 96% accurate.

# %% [markdown]
# ## Step 4: Embed the whole corpus
#
# Same `model.encode` as Step 1, now on all 1,257 articles at once. This is
# the one slow-ish step in the lab: a minute or so on a laptop CPU. Passing a
# list of texts and `show_progress_bar=True` gives you a progress bar.

# %%
import time

texts = df["doc_text"].tolist()     # the article column, as a plain Python list

t0 = time.time()
X = model.encode(texts, show_progress_bar=True)
print(f"\nEmbedded {len(texts)} articles in {time.time() - t0:.0f}s -> array {X.shape}")

# %% [markdown]
# `X` is now a `(1257, 384)` array: one row per article, row `i` lined up with
# `df` row `i`. Every article is a point in the same 384-dimensional space,
# and from here on the text is behind us — everything is arithmetic on `X`.
#
# A note on downloads: this lab needs the model itself (the ~90 MB from Step
# 1), because Step 5 embeds your search queries live too. That download is
# required — the same deal as this afternoon's image lab, which pulls a bigger
# model. What's *optional* is re-running this embed: if the corpus embed above
# is painfully slow, the identical vectors are cached, so you can skip it with
# `X = np.load(os.path.join(DATA_DIR, "india_embeddings.npz"))["embeddings"]`.
# Run it live if you can, though — watching it embed is half the point.

# %% [markdown]
# ## Step 5: Semantic search in a few lines
#
# Here's where embeddings are really cool. To search the corpus by *meaning*:
# embed your query into the same space, compute its cosine similarity to all
# 1,257 articles, and sort.

# %%
def search(query, k=5):
    """Return the k articles most similar to a text query."""
    query_vec = model.encode(query)         # the query becomes a vector too
    sims = cosine(query_vec, X)             # similarity to every article
    top = np.argsort(-sims)[:k]             # indices of the k highest
    results = df.iloc[top].copy()
    results["similarity"] = sims[top]
    return results

hits = search("police opened fire on a crowd of protesters")
for _, row in hits.iterrows():
    print(f"{row['similarity']:.3f}  labels={row['doc_labels']}")
    print("  ", row["doc_text"][:150], "...\n")

# %% [markdown]
# Look at what it found: firing, baton charges, crowds — and notice the top
# hits don't need to contain the words "opened fire" or "protesters." That's
# the benefit over exact keyword search, and it needed **no labels at all**: the model's
# general knowledge of language did the work. (This is also, at heart, how
# retrieval-augmented LLM systems find their source documents.)
#
# One technical detail: `np.argsort(-sims)` sorts by *negative* similarity, which
# is the standard trick for "sort highest-first" (argsort only goes
# lowest-first). 
#
# **Exercise.** Try two or three of your own queries. A good one to try:
# search for something using words that *don't* appear in the articles (e.g.
# "detained demonstrators" when the articles say "arrested" or "taken into
# custody") and see whether meaning-based search still finds them.

# %%
# try it here
# search("...")

# %% [markdown]
# ## Step 6: What the context window throws away
#
# The chapter's warning: this model has a **context window** of 256 tokens
# (very roughly 200 words) and *silently ignores everything after that*. Our
# articles are not short. Let's see what happens.

# %%
df["n_words"] = df["doc_text"].str.split().str.len()   # split each text on spaces, count
print(df["n_words"].describe().round(0))
print(f"\nShare of articles longer than ~200 words: {(df['n_words'] > 200).mean():.0%}")

# %% [markdown]
# The median article is ~300 words and three-quarters are over 200, so for
# most of this corpus "embed the article" really means "embed the first
# half-ish of the article." But the model won't tell us! Sometimes that's fine (the
# key facts are up top); sometimes the police action is in a sentence on
# paragraph nine that's outside the window. This is a real limitation, and
# it's part of why the classifier below struggles on the rarest events: the
# one sentence that names a killing can fall off the end. When you embed your
# own documents you should always check their length distribution.

# %% [markdown]
# ## Step 7: A classifier — does this article report police action?
#
# Now the workhorse from the chapter: logistic regression where the 384
# embedding numbers are the covariates, predicting a label. We'll predict
# `ANY_ACTION` (does this article report police activity at all?), the most
# common and most clearly *topical* of our labels.
#
# First, a **60/20/20 split**: 60% of the articles to train on, 20% to check
# our work on as we go (the *validation* set), and 20% held out as a final
# *test* set that we don't look at until the very end. Keeping a test set
# untouched is the only honest way to
# report how well your instrument works.

# %%
from sklearn.model_selection import train_test_split

idx = np.arange(len(df))
train_idx, temp_idx = train_test_split(idx, test_size=0.4, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
print(f"train: {len(train_idx)}   validation: {len(val_idx)}   test: {len(test_idx)}")



# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report

y = df["ANY_ACTION"].values          # the 0/1 label for every article

# class_weight="balanced" tells the model to care about the rarer class
# proportionally more, so it doesn't just learn to say "no" all the time.
clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X[train_idx], y[train_idx])

# Check our work on the validation set (NOT the test set).
val_pred = clf.predict(X[val_idx])
print(f"Validation F1: {f1_score(y[val_idx], val_pred):.3f}")
print()
print(classification_report(y[val_idx], val_pred, target_names=["no action", "action"]))

# %% [markdown]
# The `classification_report`
# breaks down the performance: precision (when it says "action," how often is it right?)
# and recall (of the real "action" articles, how many did it catch?).

# %% [markdown]
# ## Step 8: Read some errors
#
# Overall scores hide *what kind* of mistakes the model makes. Let's pull up
# some. With ~38 errors on the validation set we won't read all of them, but a
# sample tells us a lot. We'll look separately at the two kinds:
#
# - **False positives**: predicted "action," truly no action.
# - **False negatives**: predicted "no action," truly action.

# %%
val_df = df.iloc[val_idx].copy()
val_df["pred"] = val_pred
val_df["true"] = y[val_idx]

false_neg = val_df[(val_df["true"] == 1) & (val_df["pred"] == 0)]
false_pos = val_df[(val_df["true"] == 0) & (val_df["pred"] == 1)]
print(f"{len(false_neg)} false negatives (missed action), "
      f"{len(false_pos)} false positives (false alarm)\n")

print("=== A few MISSED action articles (false negatives): ===\n")
for _, row in false_neg.head(3).iterrows():
    print(row["doc_text"][:300], "...\n")

# %% [markdown]
# **Exercise.** Read the three above (and change `.head(3)` to see more, or
# look at `false_pos`). For each, think about: is
# the mistake because (a) the police action is mentioned only briefly, late in
# a long article () the context-window problem from Step 6),  (b) the article is
# genuinely ambiguous, or (c) the model just missed something clear? The mix of
# these is what tells you whether a better model or a different *representation*
# would help.

# %%
# try it here — look at false_pos, or read more of false_neg

# %% [markdown]
# ## Step 9: Does the fancy method actually win?
#
# In the chapter, I'm very enthusastic about embeddings, and semantic search (Step 5) shows one of their major benefits.
# But "fancier" doens't necessarily mean "better". A good habit from computer science is to always compare 
#  against a simple baseline: **TF-IDF**, the bag-of-words
# representation from the 1990s. This counts the words in each document (weighted so
# common words matter less). Then we can train the same logistic
# regression on this (modified) bag of words.

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

# Build TF-IDF features from the SAME articles. This is a sparse matrix of
# word counts, one row per article, aligned with df just like X.
tfidf = TfidfVectorizer(max_features=5000, stop_words="english",
                        ngram_range=(1, 2), min_df=2)
X_tfidf = tfidf.fit_transform(df["doc_text"])
print("Embeddings shape:", X.shape, "  TF-IDF shape:", X_tfidf.shape)

# %% [markdown]
# Now train the same classifier on each representation, for each event type,
# and compare F1 on the validation set. One small loop does it.

# %%
def f1_on_val(features, event):
    """Train logistic regression on `features` to predict `event`, F1 on val."""
    y = df[event].values
    m = LogisticRegression(max_iter=1000, class_weight="balanced")
    m.fit(features[train_idx], y[train_idx])
    return f1_score(y[val_idx], m.predict(features[val_idx]))

print(f"{'event':11s} {'embeddings':>11s} {'TF-IDF':>8s}")
for event in event_types:
    emb_f1 = f1_on_val(X, event)
    tfidf_f1 = f1_on_val(X_tfidf, event)
    print(f"{event:11s} {emb_f1:11.3f} {tfidf_f1:8.3f}")

# %% [markdown]
# **TF-IDF wins on every single event type**! (Though narrowly on
# the broad `ANY_ACTION` task), and by a huge margin on the specific rare ones
# (`KILL`, `FORCE`). 
#
# Why? 
#

# %% [markdown]
# Finally, the number we've been saving: the **test set**. We used validation
# to compare representations and decide TF-IDF is the better classifier here.
# Now we report its F1 once, on the 20% we held out, as our honest estimate of how well the model would perform in the real world.

# %%
best_features = X_tfidf     # TF-IDF won the comparison above
y = df["ANY_ACTION"].values
final = LogisticRegression(max_iter=1000, class_weight="balanced")
final.fit(best_features[train_idx], y[train_idx])
test_f1 = f1_score(y[test_idx], final.predict(best_features[test_idx]))
print(f"Final held-out TEST F1 for ANY_ACTION (TF-IDF): {test_f1:.3f}")

# %% [markdown]
# ## Capstone variant
#
# You can use this code pretty much as-is on your own text.
#
# 1. **Load your text** into a dataframe with a text column (and label columns
#    if you have them).
# 2. **Embed it**: `X = model.encode(df["text"].tolist(), show_progress_bar=True)`,
#    then cache the array so you don't have to recompute it.
# 3. **Search it** with the Step 5 function: this works with *no labels* and
#    is often the fastest way to explore a corpus you haven't read.
# 4. **If you have labels, classify**,  and run the Step 9 head-to-head comparison. Whether
#    embeddings or TF-IDF wins depends on your task (topical vs. keyword) and
#    your document lengths. Alternatively, you can experiment with the much larger and 
#    better sentence-transformer models.
#

# %% [markdown]
# ## If you finish early
#
# - **Look at the space.** PCA the 384-dim embeddings down to 2
#   (`sklearn.decomposition.PCA`), scatter-plot the articles, and color the
#   points by `ANY_ACTION`. Do the action/no-action articles separate at
#   all? 
# - **Query engineering.** Find the search query that best surfaces `KILL`
#   articles. How much does wording matter? "police killing" vs. "deaths in
#   police firing" vs. "shot dead by police"? 
