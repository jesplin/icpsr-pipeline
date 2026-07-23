# %% [markdown]
# # Lab 11: Files, Long Documents, and Multi-Step LLM Pipelines
#
# ICPSR 2026 — The Social Science Data Pipeline
# Instructor: Andy Halterman
#
# Bonus lab! I tried to pull together some code to answer questions that
# people had at the end of the day yesterday. 
#
# 1. **Files and folders.** Reading a whole directory of files into Python,
#    filtering to the ones you want, and saving your results back out (including
#     Colab-specific "download to my laptop" code).
# 2. **Long documents and embeddings.** On Wednesday we found that the embedding
#    model silently ignores everything past ~200 words. We'll talk about how to handle
#    documents that are longer than that: chunking, and combining the chunks.
#    Plus some embedding visualization.
# 3. **LLMs on hard documents.** PDFs, documents that are longer than the model's
#    context window, and a three-step LLM pipeline (classify, extract,
#    summarize) on the India data.
#
# In a few places I skip things you'd want
# in real production code (retries, error handling) because they'd get in the way
# of reading what the code actually does.

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
    if not os.path.exists("/content/icpsr-pipeline"):
        !git clone -q https://github.com/ahalterman/icpsr-pipeline.git /content/icpsr-pipeline
    COURSE_DIR = "/content/icpsr-pipeline"
else:
    COURSE_DIR = os.path.dirname(os.getcwd()) if os.path.basename(os.getcwd()) in ("labs", "solutions") else os.getcwd()

DATA_DIR = os.path.join(COURSE_DIR, "data", "cached")
OUTPUTS_DIR = os.path.join(COURSE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# %% [markdown]
# # Part 1: Files and folders
#
# ## Step 1: Make a folder of files to practice on
#
#
# To practice, we'll make our own folder filled with different files. We'll take 25 of the India articles and
# write each one out as its own `.txt` file, and also drop in a couple of files that are *not* articles (a README
# and a spreadsheet). 

# %%
import pandas as pd

df = pd.read_json(os.path.join(DATA_DIR, "india_police_events.jsonl"), lines=True)

# The folder we'll write into. os.path.join builds the path with the right slash
# for your operating system (/ on Mac/Linux, \ on Windows) so you never hard-code
# separators.
ARTICLE_DIR = os.path.join(OUTPUTS_DIR, "article_folder")
os.makedirs(ARTICLE_DIR, exist_ok=True)

# Write 25 articles, one per .txt file.
for _, row in df.head(25).iterrows():
    filename = f"article_{row['doc_id']}.txt"           # e.g. "article_11.txt"
    path = os.path.join(ARTICLE_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(row["doc_text"])

# And two files that are NOT articles, so filtering has something to filter out.
with open(os.path.join(ARTICLE_DIR, "README.md"), "w") as f:
    f.write("# Scraped articles\nTimes of India, 2002.\n")
pd.DataFrame({"note": ["do not include me in the corpus"]}).to_csv(
    os.path.join(ARTICLE_DIR, "notes.csv"), index=False)

print("Wrote files to", ARTICLE_DIR)

# %% [markdown]
# ## Step 2: Traverse the directory
#
# `os.listdir(folder)` gives you a list of every file/dir name in a folder

# %%
all_names = os.listdir(ARTICLE_DIR)
print(len(all_names))     # how many things are in the folder
print(all_names[:6])      # the first few names

# %% [markdown]
# That list has our 25 `.txt` files plus the README and the CSV. We only want
# the articles. The tool for "does this name end with `.txt`?" is the string
# method `.endswith()`, which returns `True` or `False`:

# %%
print("article_11.txt".endswith(".txt"))   # True
print("README.md".endswith(".txt"))        # False

# %% [markdown]
# So we filter the list down to just the `.txt` files. You can do this with a
# `for` loop and an `if`:

# %%
txt_names = []
for name in all_names:
    if name.endswith(".txt"):
        txt_names.append(name)

txt_names = sorted(txt_names)   # os.listdir order is arbitrary; sort for sanity
print(len(txt_names))     # how many .txt files we kept
print(txt_names[:5])

# %% [markdown]
# Now we can make a loop to read in the files. For each filename we (1) build its full path with `os.path.join`,
# (2) open and read the file, and (3) store the result in a list.

# %%
records = []
for name in txt_names:
    # os.path.join glues the folder and the filename into one full path.
    # (You could paste them together yourself, ARTICLE_DIR + "/" + name, but that
    #  hard-codes the "/" -- it breaks on Windows, and doubles up if ARTICLE_DIR
    #  already ends in a slash. os.path.join handles all that, so it's good to use.)
    path = os.path.join(ARTICLE_DIR, name)
    with open(path, encoding="utf-8") as f:
        text = f.read()                        # the whole file, as one string
    records.append({"filename": name, "text": text})

# A list of dicts turns straight into a dataframe -- back on familiar ground.
corpus = pd.DataFrame(records)
print("Shape:", corpus.shape)
corpus.head(3)

# %% [markdown]
# ## Step 3: Working with directories, generally
#
# A few directory functions from `os` and `os.path` that are helpful to know:

# %%
# Does a path exist yet? 
print(os.path.exists(ARTICLE_DIR))                              # True: we just made it
print(os.path.exists(os.path.join(ARTICLE_DIR, "nope.txt")))    # False: made-up file

# Make a folder (and any missing parent folders). exist_ok=True means "don't
# complain if it's already there" -- without it, a second run would error.
os.makedirs(os.path.join(OUTPUTS_DIR, "subfolder", "deeper"), exist_ok=True)

# Split a full path into its pieces.
example = os.path.join(ARTICLE_DIR, "article_11.txt")
print("basename (just the file name):", os.path.basename(example))
print("dirname  (just the folder):   ", os.path.dirname(example))

# %% [markdown]
# One shortcut worth knowing: `glob` does the "list a folder and filter by
# pattern" of Step 2 in a single line. `*` means "anything," so `*.txt` means
# "every name ending in `.txt`." It even hands you the full paths already, so you
# skip the `os.path.join`.

# %%
import glob

txt_paths = glob.glob(os.path.join(ARTICLE_DIR, "*.txt"))
print(len(txt_paths))     # same 25 files as the loop found
print(txt_paths[0])       # note: glob gives back the FULL path, not just the name

# I still like writing the `os.listdir` + `endswith` loop when I'm teaching or
# debugging, because you can see every step.

# %% [markdown]
# ## Step 4: Saving files
#
# We've been skimming over the "saving" part of the labs, but 
# obviously it's importat. On your computer, you can save your dataframes in a few
# different formats:

# %%
corpus.to_csv(os.path.join(OUTPUTS_DIR, "corpus.csv"), index=False)
corpus.to_json(os.path.join(OUTPUTS_DIR, "corpus.jsonl"), orient="records", lines=True)
corpus.to_parquet(os.path.join(OUTPUTS_DIR, "corpus.parquet"))
print("Saved corpus.csv, corpus.jsonl, corpus.parquet to", OUTPUTS_DIR)

# %% [markdown]
# ### Saving in Colab vs. downloading to your computer
#
# When you run `to_csv(...)` above, the file is written to Colab's own temporary
# hard drive, but that
# storage is temporary. When your Colab runtime disconnects (you close the tab,
# leave it idle too long, or it resets), those files are deleted. So writing a
# file in Colab is not the same as having the file.
#
# To actually get a file onto your own computer, you have to explicitly
# download it. `files.download()` triggers your browser's normal download
# prompt, and the file lands in your Downloads folder like any other download.
#
# (There's also a Google Drive integration that mounts your Drive as a permanent
# folder, but we're not setting that up today since it's kind of a pain)

# %%
if IN_COLAB:
    from google.colab import files
    files.download(os.path.join(OUTPUTS_DIR, "corpus.csv"))
    # ^ this pops up a browser download of corpus.csv onto YOUR computer.
else:
    print("Not in Colab -- files are already on this machine's real disk,")
    print("so there's nothing to 'download'. Just open them where you saved them.")



# %% [markdown]
# # Part 2: Long documents and embeddings
#
# Now the embedding follow-up. We'll need the sentence-transformer model again.

# %%
!pip install -q sentence-transformers pandas numpy scikit-learn plotly

# %%
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def cosine(a, b):
    """Cosine similarity between vector `a` and vector-or-matrix `b`."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return b @ a

# %% [markdown]
# ## Step 5: Chunking long documents
#
# Recall Wednesday's problem: `all-MiniLM-L6-v2` has a context window of 256
# tokens (roughly 200 words) and drops everything after that. Let's construct
# a bad example:  a long article about a
# political meeting (no police action at all), followed by a shorter one about
# the army and police firing on a crowd. The police-firing content is now in the
# end of the document, so it'll get missed

# %%
# .str.split() breaks each article into a list of words, then count them with .str.len().
df["n_words"] = df["doc_text"].str.split().str.len()
# For each row, is "FORCE" in that row's list of labels? -> 1 or 0.
df["FORCE"] = df["doc_labels"].apply(lambda labels: 1 if "FORCE" in labels else 0)
# Same idea for the broad "any police action at all" label.
df["ANY_ACTION"] = df["doc_labels"].apply(lambda labels: 1 if "ANY_ACTION" in labels else 0)

# %%
# Find a long article with NO police action, for the front of our document.
no_action = df[(df["ANY_ACTION"] == 0) & (df["n_words"] > 250)]
intro = no_action.iloc[0]["doc_text"]     # .iloc[0] = the first matching row

# Find a short article about police firing, to put at the end.
firing = df[(df["FORCE"] == 1) & (df["n_words"] < 160)]
tail = firing.iloc[0]["doc_text"]

# Paste them together into one long document, intro first.
long_doc = intro + "\n\n" + tail

print("total words:", len(long_doc.split()))
print("(the first", len(intro.split()), "words are the non-police setup)")
print("\nThe buried tail (the part the model won't reach):")
print("  ", tail[:180], "...")

# %% [markdown]
# Now the smoking gun! If the model truncates, then embedding the whole
# document should give the same vector as embedding only the intro, because
# the model just never reads past the intro. Let's check:

# %%
whole_vec = model.encode(long_doc)     # "embed the whole thing"
intro_vec = model.encode(intro)        # embed only the front part

print("cosine(whole document, intro-only) =",
      round(float(cosine(intro_vec, whole_vec[None, :])[0]), 4))
# ~1.0 means the two are the same vector: the 133-word police-firing tail
# contributed literally nothing to the "whole document" embedding.

# %% [markdown]
# It comes back essentially identical. The police-firing tail contributed *nothing*
# to the embedding. So if you search this document for police firing, you'll miss
# it — not because embeddings are bad, but because the model never saw the
# relevant words.
#
# ### A fix: chunk and combine
#
# The idea is to cut the document into pieces small enough that each fits in
# the window, embed each piece, and then combine the piece-embeddings into one
# vector for the document:

# %%
# (a) Fixed-length chunks: split every N words.
def chunk_by_words(text, size=180):
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]

# (b) By paragraph: split on blank lines. This matches the document's origianl structure.
# but paragraphs vary in length, so a very long paragraph can still overflow.
def chunk_by_paragraph(text):
    paras = text.split("\n\n")
    chunks = []
    for p in paras:
        if p.strip():           # skip empty / whitespace-only paragraphs
            chunks.append(p)
    return chunks

word_chunks = chunk_by_words(long_doc, 180)
para_chunks = chunk_by_paragraph(long_doc)
print(f"fixed-length: {len(word_chunks)} chunks   by paragraph: {len(para_chunks)} chunks")

# %% [markdown]
# Now embed the chunks and combine them. You embed the chunks with the same
# `model.encode` as always. 
# 
# The
# question is then how to squash those several rows into one document vector. Two options
# are the elementwise mean or max.

# %%
query = "police opened fire on a crowd"
query_vec = model.encode(query)

chunk_vecs = model.encode(word_chunks)      # one vector per chunk

mean_vec = chunk_vecs.mean(axis=0)          # average the chunks, dimension by dimension
max_vec  = chunk_vecs.max(axis=0)           # take the max in each dimension

print(f"query vs. whole-doc (truncated):   {float(cosine(query_vec, whole_vec[None,:])[0]):.3f}")
print(f"query vs. chunk MEAN:              {float(cosine(query_vec, mean_vec[None,:])[0]):.3f}")
print(f"query vs. chunk MAX:               {float(cosine(query_vec, max_vec[None,:])[0]):.3f}")

# %% [markdown]
# On my run, the truncated whole-document embedding scores **0.08** against the
# query (it never saw the firing), while the **mean** of the chunks scores
# **0.31** and the **max** scores **0.24**. Chunking works!
#
# In this case, "mean-pooling", which  gives you
# the "average meaning" of the document, works best.
#
# There's also a third option that's often the *best* for search: don't combine at
# all. Keep every chunk's embedding, score the query against each chunk, and take
# the best-matching chunk as the document's score. That's how real
# retrieval systems (and RAG pipelines) do it:

# %%
per_chunk_sims = cosine(query_vec, chunk_vecs)
print("similarity to each chunk:", per_chunk_sims.round(3))
print("best chunk score:        ", round(float(per_chunk_sims.max()), 3))
# The best single chunk (~0.37) beats both pooling methods, because it isn't
# diluted by the two irrelevant intro chunks. The tradeoff: you store more
# vectors (one per chunk, not one per document).

# %% [markdown]
# SUmmary:
#
# - If you need one vector per document (e.g., to feed a classifier like Lab
#   3b's), chunk and mean-pool.
# - If you're doing search or retrieval, keep the chunks separate and score
#   against the best-matching chunk.
# - Either way, chunk if your documents are longer than ~200 words or you'll miss the end.
#


# %% [markdown]
# ## Step 6: Which *sentence* matched? Search at the sentence level
#
# When we did semantic search in Lab 3b over whole articles, but it
# was often hard to see *why* an article came back: somewhere in 300 words there
# was a relevant sentence, but you had to hunt for it. If you
# embed at the *sentence* level and search those, the search points you straight at the matching
# sentence, and it's much easier to see what the model found relevant.
# 
# First we need a way to break an article into sentences. There's no perfect rule,
# but "split after a `.`, `?`, or `!` that's followed by a space" gets you most of
# the way for news text. THere are fancier methods for doing this (e.g., spaCy or NLTK), 
# but that's too much for today.

# %%
import re

def split_sentences(text):
    """Dumb sentence splitter. (Think about how this would fail!)"""
    # re.split cuts the text wherever a . ? or ! is followed by a space.
    parts = re.split(r"(?<=[.!?])\s+", text)
    sentences = []
    for p in parts:
        if len(p.split()) >= 4:      # skip tiny fragments ("p. 3", stray bits)
            sentences.append(p.strip())
    return sentences

# Try it on one article:
print(split_sentences(df.iloc[0]["doc_text"])[:3])

# %% [markdown]
# Now build a pool of sentences from 150 articles, remembering which article each
# sentence came from (so we can carry its label along for the plot in Step 7).

# %%
sample = df.sample(150, random_state=1).reset_index(drop=True)

rows = []
for _, article in sample.iterrows():
    for sentence in split_sentences(article["doc_text"]):
        rows.append({
            "doc_id": article["doc_id"],
            "ANY_ACTION": article["ANY_ACTION"],
            "sentence": sentence,
        })
sents = pd.DataFrame(rows)
print(len(sents), "sentences from", len(sample), "articles")

# %% [markdown]
# Embed every sentence, then run
# the exact same cosine-similarity search from Lab 3b, just over sentences instead
# of documents.

# %%
sent_vecs = model.encode(sents["sentence"].tolist(), show_progress_bar=True)

query_vec = model.encode("police detained protesters")
sims = cosine(query_vec, sent_vecs)
top = np.argsort(-sims)[:6]             # the 6 best-matching sentences

for i in top:
    print(round(float(sims[i]), 3), "|", sents.iloc[i]["sentence"])



# %% [markdown]
# ## Step 7: Seeing your embeddings (a fun aside)
#
# Embeddings are points in 384-dimensional space, which you obviously can't
# picture. But you can "project" them down to 2 dimensions  with PCA and make a scatter plot, 
# which is a nice way to explore a corpus. Are there distinct cluster? Do your labels
# separate? This is a stripped-down version of an interactive embedding explorer I
# built for a synthetic-text project.
#
# We'll plot the *sentences* we just embedded. Plotting sentences instead of whole
# articles has a practical payoff for the next step: when you hover a point, you
# see one complete, readable sentence, rather than a long article.

# %%
from sklearn.decomposition import PCA

# PCA finds the 2 directions of most variation and projects onto them.
xy = PCA(n_components=2, random_state=42).fit_transform(sent_vecs)
sents["x"] = xy[:, 0]
sents["y"] = xy[:, 1]

# %% [markdown]
# Now an *interactive* scatter with plotly: one point per sentence, colored by
# whether its article reports police action, and you can hover to read the
# whole sentence.

# %%
import plotly.express as px

# A readable label for the color legend.
sents["from_article"] = sents["ANY_ACTION"].map({0: "no-action article", 1: "police-action article"})

fig = px.scatter(
    sents, x="x", y="y",
    color="from_article",
    hover_data={"sentence": True, "x": False, "y": False},
    opacity=0.5,
    title="~2,000 sentences, embeddings projected to 2-D (hover to read)",
)
fig.update_traces(marker=dict(size=5))
fig.show()

# %% [markdown]
# One note on the color: it's the *article's* label, not the sentence's. A
# police-action article still contains plenty of ordinary background sentences, so
# the two colors mix more than a document-level plot would 

# %% [markdown]
# # Part 3: LLMs on hard documents
#
# The last part moves to the OpenRouter API for a few things people asked about yesterday:
# reading PDFs, handling documents longer than the model's context window, and
# chaining several LLM calls into a pipeline.

# %%
!pip install -q openai pymupdf pandas

# %% [markdown]
# ## Connecting 
#
# Same setup as Lab 5. 

# %%
if not os.environ.get("OPENROUTER_API_KEY"):
    import getpass
    os.environ["OPENROUTER_API_KEY"] = getpass.getpass("Paste your OpenRouter API key: ")

from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
MODEL = "qwen/qwen3-30b-a3b"

import re

def chat(prompt):
    """Send one prompt, return the response text. Qwen3 sometimes 'thinks out
    loud' in <think>...</think> before answering, so we strip that. (No retry
    logic here, to keep it readable -- Lab 5's version has it.)"""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    text = resp.choices[0].message.content
    clean_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return clean_text

print(chat("Reply with exactly one word: ready"))

# %% [markdown]
# ## Working with PDFs
#
# (Only big models handle PDFs natively. We'll extract text outselves.)
#


# %%
import fitz   # this is what the pymupdf library is called when you import it

pdf_path = os.path.join(COURSE_DIR, "labs", "papers",
                        "2022_mmchived-multimodal-chile-and-venezuela-protest-event-data.pdf")
doc = fitz.open(pdf_path)

pages = [page.get_text() for page in doc]     # one string per page
pdf_text = "\n".join(pages)

print(f"{doc.page_count} pages, {len(pdf_text.split())} words extracted")
print("\nFirst 400 characters:\n")
print(pdf_text[:400])

# %% [markdown]
# PDF text extraction is often
# imperfect and library quality varies a lot. I first tried the popular `pypdf`
# library on this same file and it mangled it (dropping the spaces between words,
#  turned `&` into `F`, etc). `pymupdf`
# worked better for me here.
#
# And if you have *scans* of text, we'll come back to that in a bit.
#
# Once you have the text, the LLM part is ordinary — it's just a long string in
# your prompt:

# %%
question = "In one sentence, what data source does this paper use to measure protests?"
answer = chat(f"{question}\n\nPaper text:\n{pdf_text}")
print(answer)

# %% [markdown]
# ### The image path (for figures, tables, scanned pages)
#
# Text extraction throws away the figures, and the tables often come out as a mess
# of numbers with no structure. When the *visual* content matters, render each
# page to an image and send it to a vision model (prev lab). `pymupdf`
# can convert a page to a PNG for a vision model:

# %%
page_image_path = os.path.join(OUTPUTS_DIR, "page_1.png")
doc[0].get_pixmap(dpi=100).save(page_image_path)
print("Rendered page 1 to", page_image_path)

# %% [markdown]
# To send that image to a vision model, you base64-encode it and put it in the
# message the same way Lab 6 did. (We hid the actual API call before, but it's useful to see it.) 

# %%
import base64

with open(page_image_path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

resp = client.chat.completions.create(
    model="google/gemma-4-26b-a4b-it",     # a vision model, not our text MODEL
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "What is the title of this paper?"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ]}],
    temperature=0.0,
)
print(resp.choices[0].message.content)


# %% [markdown]
# ## Step 9: Documents longer than the context window
#
# One of you has a few hundred documents that are 500–1,000 pages each. Can you
# just paste one into the LLM? Let's do the math!
#
# Every model has a **context window**: the maximum number of tokens (input +
# output) it can handle at once. Many open models cap around **128,000 tokens**.
# Recall the rough conversion: tokens \approx words \times 1.3.

# %%
words_per_page = 500      # a rough average 
for n_pages in [10, 128, 500, 1000]:
    est_tokens = n_pages * words_per_page * 1.3
    fits = "fits" if est_tokens < 128_000 else "TOO BIG for a 128k window"
    print(f"{n_pages:5d} pages  ~ {est_tokens:10,.0f} tokens   {fits}")

# %% [markdown]
# A 500-page document is on the order of **300,000+ tokens**, several times over a
# 128k window. 
#
# And even for documents that *do* fit, or for the very-large-context models
# (some go to 1M tokens), stuffing in a whole book is usually a bad idea. Models
# get measurably worse at finding information buried in the middle of a huge
# context (the "lost in the middle" problem), and you pay for every token on every
# call.
#
# ### Instead, "map-reduce"
#
# The standard fix is the same chunking idea from above:
#
# 1. **Map**: split the document into chunks that comfortably fit in context, and run the LLM
#    on each chunk independently (summarize it, or extract what you need).
# 2. **Reduce**: combine the per-chunk outputs, maybe with one more LLM call that
#    summarizes the summaries.
#
# We'll demo it on the PDF text, treating it as our stand-in for a long document.
# First, split into word-chunks, then summarize
# each chunk:

# %%
def chunk_by_words(text, size=600):
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]

chunks = chunk_by_words(pdf_text, size=600)
print(f"Split the paper into {len(chunks)} chunks.\n")

# MAP: summarize each chunk on its own.
chunk_summaries = []
for i, chunk in enumerate(chunks):
    summary = chat(f"Summarize this section of a research paper in 1-2 sentences:\n\n{chunk}")
    chunk_summaries.append(summary)
    print(f"[chunk {i}] {summary}\n")

# %% [markdown]
# Now the **reduce** step: hand all the little summaries to the model at once and ask for one combined summary. 

# %%
joined = "\n".join(f"- {s}" for s in chunk_summaries)
final_summary = chat(f"Here are section summaries of one paper. Write a 3-sentence "
                     f"overall summary of the paper:\n\n{joined}")
print(final_summary)

# %% [markdown]
# For a real 500-page document you'd use bigger chunks and maybe a middle layer
# (summarize groups of chunks, then summarize those), but the shape is the same:
# map over pieces, then reduce. And if you're *extracting* rather than
# summarizing, then use the map step to return a list per chunk
# and the reduce step to concatenate and de-duplicate them.

# %% [markdown]
# ## Step 10: A multi-step LLM pipeline
#
# Here's an example of a three-step LLM pipeline.
#
# 1. **Classify**: is this article relevant (does it report police action)? A
#    cheap yes/no gate.
# 2. **Extract**: for the relevant ones only, pull structured details — which kind
#    of action, and a supporting quote.
# 3. **Summarize**: aggregate over all the extractions into a final dataset.
#

# %%
import json

sample = df.sample(15, random_state=2026).reset_index(drop=True)
print(f"Working on {len(sample)} articles.")

def get_json(raw):
    """Pull the first {...} block out of a response and parse it. Returns {} if
    there's nothing parseable (kept simple -- no elaborate error handling)."""
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}

# %% [markdown]
# ### Pass 1: the relevance gate

# %%
GATE_PROMPT = """Does this news article report the police (or army/paramilitary
acting as police) doing or saying anything? Answer with ONLY JSON:
{"relevant": "yes" or "no"}

Article:
"""

sample["relevant"] = ""
for i, row in sample.iterrows():
    result = get_json(chat(GATE_PROMPT + row["doc_text"][:2000]))
    sample.loc[i, "relevant"] = result.get("relevant", "no")

relevant = sample[sample["relevant"] == "yes"].copy()
print(f"{len(relevant)} of {len(sample)} articles passed the gate as relevant.")

# %% [markdown]
# ### Pass 2: extraction (only on the relevant articles)
#
# Notice we only loop over `relevant`, not the whole sample. For each one we ask for the *type* of action and a short **supporting
# quote** from the article. 

# %%
EXTRACT_PROMPT = """This article reports police activity. Extract, as ONLY JSON:
{"action_type": "arrest" | "kill" | "force" | "other",
 "supporting_quote": "<the exact phrase from the article that shows it>"}

Article:
"""

extractions = []
for i, row in relevant.iterrows():
    result = get_json(chat(EXTRACT_PROMPT + row["doc_text"][:2000]))
    extractions.append({
        "doc_id": row["doc_id"],
        "action_type": result.get("action_type", "other"),
        "supporting_quote": result.get("supporting_quote", ""),
    })

extracted = pd.DataFrame(extractions)
extracted



# %% [markdown]
# ### Pass 3: summarize over the extractions into a final dataset
#
# %%
# The structured part: how many of each action type? 
counts = extracted["action_type"].value_counts()
print("Action types found across the sample:")
print(counts)

# The synthesis part: one LLM call over the extracted quotes.
quote_list = "\n".join(f"- ({r.action_type}) {r.supporting_quote}"
                       for r in extracted.itertuples())
synthesis = chat(f"""Here are extracted police actions from a set of news articles
about the 2002 Gujarat violence. Write a 3-sentence summary of what kinds of
police activity these articles describe:

{quote_list}""")
print("\n--- Synthesis ---")
print(synthesis)


# ## If you finish early
#
# - **Paragraph vs. fixed chunking.** Redo Step 5's mean-pool with
#   `chunk_by_paragraph` instead of `chunk_by_words`. Does respecting the
#   document's paragraph boundaries help or hurt on this document?
# - **Audit the pipeline.** In Pass 2, add a `page`-style check: for a few rows,
#   confirm the `supporting_quote` actually appears in `doc_text`
#   (`quote in text`). How often does the model quote something that isn't
#   really there? (This is a cheap, powerful hallucination check.)
# - **Cost the gate.** The relevance gate saved you running extraction on the
#   irrelevant articles. Using Lab 5's cost arithmetic, how much does that save on
#   a 100,000-document corpus where only 30% are relevant?
