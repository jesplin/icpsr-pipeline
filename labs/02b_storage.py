# %% [markdown]
# # Lab 2b: Storage — SQLite and Streaming
#
# ICPSR 2026 — The Social Science Data Pipeline
# Instructor: Andy Halterman
#
# This lab is the hands-on half of the storage chapter (Part 2: "when data
# outgrow memory"). 
#
#  We'll briefly cover two things, but these are largely intended to be reference
#  code for you to come back to in the future.
#
# 1. **SQLite**: get a dataframe into a database and get answers back out with
#    SQL. This is likely your first real SQL, so
#    the point is to write queries yourself and check them against pandas.
# 2. **Streaming**: process a file that's too big to hold in memory by handing
#    one record at a time through a chain of generators, and measure that it
#    actually uses less memory.
#
# Everything runs offline from `data/cached/`.

# %%
# Setup (this same block opens every lab this week)
import os

def is_colab():
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

# %%
# sqlite3, json, and tracemalloc are all in the standard library — nothing to
# install for those. We only need pandas/pyarrow to read the cached files.
!pip install -q pandas pyarrow

# %% [markdown]
# ## Part 1: SQLite
#
# A SQLite database is just one file on disk. `sqlite3` is in Python's standard
# library, so there's no server to run and nothing to install. We'll use Lab
# 2's Telegram collection: 632 posts from five public Ukrainian
# energy/infrastructure channels.
#
# ### 1a. Getting a dataframe into a database
#
# This is one line. `df.to_sql(name, con)` writes a dataframe as a table. (The
# `date` column is a pandas datetime; SQLite has no native datetime type, so we
# store it as text — `date()` and string comparisons still work on ISO strings,
# which is one reason ISO date formatting from the storage chapter matters.)

# %%
import sqlite3
import pandas as pd

tg = pd.read_parquet(os.path.join(DATA_DIR, "telegram_infra.parquet"))

con = sqlite3.connect(os.path.join(OUTPUTS_DIR, "lab2.db"))
tg.astype({"date": str}).to_sql("messages", con, if_exists="replace", index=False)

# what columns did we get?
pd.read_sql("SELECT name FROM pragma_table_info('messages')", con)["name"].tolist()

# %% [markdown]
# ### 1b. Four verbs
#
# The chapter's claim is that four verbs cover most real queries. Here they are
# on the `messages` table, one at a time. `pd.read_sql(query, con)` runs a query
# and hands the result back as a dataframe, so you stay in pandas for anything
# after the query.

# %%
# Verb 1 — SELECT ... WHERE: which columns, which rows. Posts from dtek_ua
# (the private utility) that mention substations (Ukrainian підстанці-).
pd.read_sql("""
    SELECT date, text
    FROM messages
    WHERE channel = 'dtek_ua' AND text LIKE '%підстанці%'
""", con)

# %%
# Verb 2 — ORDER BY ... LIMIT: the top of a ranking. The five most-viewed posts.
pd.read_sql("""
    SELECT channel, views, text
    FROM messages
    ORDER BY views DESC
    LIMIT 5
""", con)

# %%
# Verb 3 — GROUP BY: aggregate, exactly like pandas .groupby(). Posts and
# average views per channel.
pd.read_sql("""
    SELECT channel, COUNT(*) AS posts, CAST(AVG(views) AS INT) AS avg_views
    FROM messages
    GROUP BY channel
    ORDER BY posts DESC
""", con)

# %% [markdown]
# ### 1c. JOIN: bring in a lookup table
#
# The one verb Lab 2 didn't show. A `JOIN` combines two tables on a shared key.
# The common research pattern is a small hand-built "lookup" table — one row per
# channel, recording something you know about it — joined onto the big table.
# Here's a lookup table classifying each channel by who runs it:

# %%
channels = pd.DataFrame({
    "channel":      ["Ukrenergo", "dtek_ua", "energoatom_ua", "dsns_telegram", "kyivoda"],
    "channel_type": ["state_operator", "company", "state_operator", "government", "government"],
})
channels.to_sql("channels", con, if_exists="replace", index=False)

# Now JOIN it on: posts per channel_type, which we couldn't get from
# `messages` alone because `messages` doesn't know what type each channel is.
pd.read_sql("""
    SELECT c.channel_type, COUNT(*) AS posts
    FROM messages m
    JOIN channels c ON m.channel = c.channel
    GROUP BY c.channel_type
    ORDER BY posts DESC
""", con)

# %% [markdown]
# One warning specific to this data, repeated from the chapter because it's
# easy to get wrong: Telegram message ids restart in every channel, so a join
# on `msg_id` alone will match posts across different channels. The key is
# `(channel, msg_id)`. "What uniquely identifies a row?" is worth asking of
# every table you build.

# %% [markdown]
# ### 1d. Exercise: write a query, check it against pandas
#
# The reliable way to trust a SQL query you wrote is to compute the same thing
# in pandas on the loaded dataframe and confirm the two agree. That's the
# exercise. Write a query that returns, per channel, the total `forwards`
# summed across all its posts, ordered from most to least. Then compute the
# same thing with `tg.groupby(...)` and compare.

# %%
# your SQL version:
# by_channel_sql = pd.read_sql("""
#     SELECT channel, SUM(forwards) AS total_forwards
#     FROM messages
#     GROUP BY ...
# """, con)

# your pandas version:
# by_channel_pd = tg.groupby("channel")["forwards"].sum()...

# then compare the two — do the per-channel totals match?

# try it here

# %% [markdown]
# ### 1e. Indexes: the payoff for repeated lookups
#
# If you query the same column over and over (all posts from one channel, say),
# an index turns each lookup from a full scan of the table into roughly instant.
# It's one line, and on a table this small you won't feel the difference — but
# on a 50 GB table it's the difference between a query returning now and in a
# minute.

# %%
con.execute("CREATE INDEX IF NOT EXISTS idx_channel ON messages(channel)")
con.commit()
# Queries with `WHERE channel = ...` now use the index automatically; you don't
# change the query, just add the index once.
print("index created")

# %% [markdown]
# ## Part 2: Streaming with generators
#
# The idea from the chapter: if you're processing records one at a time (filter
# them, label them, count them), you never need all of them in memory at once —
# only the one you're working on. A Python generator is a function that `yield`s
# one item at a time instead of building a whole list and `return`ing it. You
# can chain generators together, and nothing between the source file and the
# final result ever holds the full dataset.
#
# We'll use the Lab 3b corpus: 1,257 *Times of India* news articles from around
# the 2002 Gujarat violence, stored as JSONL (one JSON object per line). It's
# small enough to run instantly and big enough to measure (Part 3), and it
# stands in for the year-of-Telegram-posts case where this actually matters.
#
# **Provenance** (the habit this chapter preaches, applied to our own file):
# scraped from the *Times of India* archive for a hand-labeled event-extraction
# dataset; fetched 2026-07-21; released for research use with attribution. Full
# note in `data/cached/README.md`, acquisition script in
# `data/acquisition/get_india_police_events.py`.

# %%
import json

jsonl_path = os.path.join(DATA_DIR, "india_police_events.jsonl")

# Reading a file line by line is *already* a generator: `for line in open(path)`
# reads one line at a time, not the whole file. So this counts every line
# without ever holding more than one line in memory.
n_lines = sum(1 for line in open(jsonl_path, encoding="utf-8"))
print(f"{n_lines} documents in the file")

# %% [markdown]
# ### 2a. A chain of generators
#
# Three small generators, each doing one job and each `yield`ing to the next:
# read raw lines, parse each into a dict, keep only the ones we want. Because
# they're chained, a document flows all the way through (read → parse → filter)
# and is discarded before the next one is read.

# %%
def read_lines(path):
    """Yield one raw line at a time."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield line

def parse(lines):
    """Yield one parsed record (dict) at a time."""
    for line in lines:
        yield json.loads(line)

def containing(records, keyword):
    """Yield only records whose text mentions `keyword`."""
    for rec in records:
        if keyword in rec["doc_text"].lower():
            yield rec

# Nothing has run yet — generators are lazy. The chain only does work when
# something consumes it, e.g. a `for` loop or `sum`. Count the matches:
matches = containing(parse(read_lines(jsonl_path)), "police")
n_police = sum(1 for _ in matches)
print(f"{n_police} of {n_lines} documents mention 'police'")

# %% [markdown]
# ### 2b. Writing output as you go
#
# The other half of streaming is on the *output* side: instead of accumulating
# a list and dumping it at the end, write each record as you produce it. One
# JSON object per line, appended immediately. If the job crashes at record
# 900,000, everything up to there is already safely on disk — this is the
# crash-safe collector pattern from the chapter and Lab 2's "finish early."

# %%
out_path = os.path.join(OUTPUTS_DIR, "police_docs.jsonl")

n_written = 0
with open(out_path, "w", encoding="utf-8") as out:
    for rec in containing(parse(read_lines(jsonl_path)), "police"):
        out.write(json.dumps({"doc_id": rec["doc_id"]}) + "\n")
        n_written += 1

print(f"wrote {n_written} records to {os.path.basename(out_path)}")

# %% [markdown]
# ### 2c. Exercise: add a stage to the chain
#
# Add one more generator to the chain: a `truncate(records, n_chars)` stage that
# yields each record with its `doc_text` cut to the first `n_chars` characters
# (imagine you only need the lede, not the whole article). Insert it into the
# chain so the flow is read → parse → filter → truncate, and write the truncated
# text out. The point: adding a processing step is just adding one link, and the
# memory story doesn't change — still one record at a time.

# %%
def truncate(records, n_chars):
    # your code here
    ...

# build the chain and write the output here

# try it here

# %% [markdown]
# ## Part 3: Does streaming actually use less memory?
#
# Let's measure it instead of taking it on faith. `tracemalloc` is a
# standard-library tool that records how much memory Python allocated, including
# the *peak* — the high-water mark during a computation. We'll run the same
# computation two ways and compare peaks:
#
# 1. **Load everything**: read the whole file into a list of dicts, then count.
# 2. **Stream**: run the generator chain, holding one record at a time.
#
# Same input, same answer — the only thing that differs is whether the full
# dataset is in memory at once. Keeping both sides pure Python (a list of dicts,
# not a dataframe) matters here: `tracemalloc` tracks Python's own allocations,
# and it's what makes the comparison clean.

# %%
import tracemalloc

def count_load_all(path, keyword):
    data = [json.loads(line) for line in open(path, encoding="utf-8")]  # whole file in RAM
    return sum(1 for r in data if keyword in r["doc_text"].lower())

def count_stream(path, keyword):
    return sum(1 for _ in containing(parse(read_lines(path)), keyword))  # one at a time

tracemalloc.start()

count_load_all(jsonl_path, "police")
_, peak_load_all = tracemalloc.get_traced_memory()

tracemalloc.reset_peak()  # forget the high-water mark, keep measuring

count_stream(jsonl_path, "police")
_, peak_stream = tracemalloc.get_traced_memory()

tracemalloc.stop()

file_mb = os.path.getsize(jsonl_path) / 1e6
print(f"file on disk:          {file_mb:5.2f} MB")
print(f"load everything, peak: {peak_load_all/1e6:5.2f} MB")
print(f"stream, peak:          {peak_stream/1e6:5.3f} MB")
print(f"ratio:                 {peak_load_all/peak_stream:.0f}x")

# %% [markdown]
# On this 2.7 MB file the absolute numbers are small (a few megabytes either
# way, nothing your laptop notices). The lesson is the *ratio*, and where it
# goes as the file grows: the load-everything peak grows with the file (100x
# the data, ~100x the memory), while the streaming peak stays roughly flat,
# because it only ever holds one record no matter how long the file is. So at a
# few megabytes streaming is a curiosity, but at the scale where your year of
# Telegram collection stops fitting in RAM, that flat line is the difference
# between the job running and a `MemoryError`.
#
# `pandas` gives you the same trick without writing generators by hand:
# `pd.read_csv(path, chunksize=500)` and `pd.read_parquet` on one row-group at
# a time both hand you pieces to aggregate as you go, instead of one giant
# dataframe. Same idea, one argument.

# %% [markdown]
# ## Capstone variant
#
# 1. Take whatever you've collected so far (even 50 records) and load it into a
#    SQLite database with `to_sql`. Write the three or four queries you already
#    know you'll want to run repeatedly — per-source counts, a date-range
#    filter, whatever your analysis needs — and save them somewhere. If you
#    can't yet name those queries, that's a useful sign about what your
#    measurement target still needs pinned down.
# 2. Which of your processing steps could be a generator chain? Anything of the
#    form "for each record, do X and write the result" qualifies. You don't need
#    it yet at your current data size, but sketch where it would go, so it's a
#    small rewrite and not a panic later.
# 3. Estimate: rows × columns × bytes-per-value (the chapter's back-of-envelope)
#    for your capstone at full size. Does three times that fit in your RAM? Your
#    answer decides whether Part 2 is optional for you or required.
#
# %% [markdown]
# ## If you finish early
#
# - Add a `CREATE INDEX` on `date` and time a `WHERE date >= ...` query with and
#   without it (`%%timeit` in a cell). On 632 rows you won't see much; write
#   down what you'd expect at 632 million.
# - Point `duckdb` at the cached parquet directly — `duckdb.sql("SELECT channel,
#   COUNT(*) FROM 'data/cached/telegram_infra.parquet' GROUP BY channel")` — and
#   confirm you get the same counts as the SQLite `GROUP BY` above. Same SQL, no
#   database to build, running straight on the file.
# - Break the streaming reader on purpose: truncate the JSONL file's last line
#   (a realistic crash mid-write) and make `parse` skip a line it can't decode
#   instead of dying, so one bad record doesn't lose the other 1,256.
