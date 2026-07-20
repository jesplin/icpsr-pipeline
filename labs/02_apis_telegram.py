# %% [markdown]
# # Lab 2: APIs and Telegram
#
# ICPSR 2026 — The Social Science Data Pipeline
# Instructor: Andy Halterman
#
# This lab has two halves, matching the chapter: first the front door
# (APIs — JSON, authentication, pagination, and the modern reality that
# even open data wants to know who you are), then the platform where the
# Ukraine war's information environment actually lives (Telegram — channel
# ecology, metadata, and source criticism with real teeth).
#
# As always: every required cell runs offline from `data/cached/`. Cells
# marked **OPTIONAL — live** need network, and sometimes credentials.

# %%
# Setup (same block as every lab)
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
!pip install -q requests pandas pyarrow matplotlib telethon nest_asyncio

# %% [markdown]
# ## Part 1: APIs
#
# Part 1 moves at demonstration pace: read and run the worked cells, and
# predict each result before you run it. Your hands-on time is Part 2, Telegram.
#
# ### 1a. What an API response looks like
#
# Yesterday you wrestled HTML written for browsers. An API hands you the
# data structure directly. Let's hit one with no key, no registration, and
# direct course relevance: DeepStateMap's territorial-control API — the
# polygons behind the map a million people check daily.

# %%
# OPTIONAL — live network. The cached fallback is two cells down.
import requests

resp = requests.get(
    "https://deepstatemap.live/api/history/last",
    headers={"User-Agent": "icpsr-pipeline-lab (academic course)"},
    timeout=30,
)
resp.raise_for_status()
raw = resp.json()          # JSON → Python dicts and lists. That's it.
type(raw), list(raw.keys())

# %% [markdown]
# Real-world detail #1: the GeoJSON you want is wrapped inside a `"map"`
# key — every API has these undocumented quirks, and `resp.json()` +
# poking at `.keys()` is how you find them. Real-world detail #2: this API
# publishes no documentation and no terms — it *works*, but building a
# dissertation on an undocumented endpoint is a risk you should price in
# (the chapter's provenance lesson: snapshot what you depend on).

# %%
# Offline fallback: we cached a cleaned snapshot on 2026-06-10
# (see data/acquisition/get_deepstate.py for exactly what "cleaned" means).
import json

with open(os.path.join(DATA_DIR, "deepstate_control.geojson")) as f:
    control = json.load(f)

print(f"{len(control['features'])} features")
control["features"][0]["properties"]

# %% [markdown]
# ### 1b. Nested JSON → flat dataframe
#
# JSON is trees; analysis wants tables. `pd.json_normalize` is the bridge.

# %%
import pandas as pd

features = pd.json_normalize(control["features"])
features[["properties.name", "geometry.type"]].head(8)

# %%
# Worked inline (predict before you run): how many features of each
# geometry.type, and how many property names mention "occupied"?
# `.value_counts()` and `.str.contains` are the two tools.
print(features["geometry.type"].value_counts())
n_occupied = features["properties.name"].str.contains("occupied", case=False, na=False).sum()
print(f"\nFeature names containing 'occupied': {n_occupied}")

# %% [markdown]
# ### 1c. Event data APIs: the registered front door
#
# The serious conflict-event APIs all want to know who you are:
#
# - **ACLED**: free academic registration → OAuth token → paginated JSON.
#   Their license *prohibits redistributing the data*, which is why there
#   is no ACLED file in `data/cached/` — an access-terms lesson in itself.
#   The complete, documented pull script is
#   `data/acquisition/get_acled.py`; run it with your own credentials
#   tonight if you registered.
# - **UCDP**: CC-BY licensed (we *can* and do cache it), API token free by
#   email. Our cached extract below came from their bulk download.
#
# The pattern both share — authenticate, request a page, append, repeat
# until empty, sleep between calls — is in `get_acled.py`. Read it now;
# it's 60 lines and it is every event-data pull you will ever write.

# %%
# The cached UCDP extract: Kharkiv oblast region, May 2024 (the Vovchansk
# offensive). 529 vetted, georeferenced, fatality-anchored events.
ucdp = pd.read_csv(os.path.join(DATA_DIR, "ucdp_ged_kharkiv_may2024.csv"))
print(ucdp.shape)
ucdp[["date_start", "adm_1", "type_of_violence", "best", "source_office"]].head()

# %%
# Worked inline (predict first: does the API-vetted event stream see the
# offensive?). Events per day, with May 10 -- the offensive's start -- marked.
import matplotlib.pyplot as plt

ucdp["date_start"] = pd.to_datetime(ucdp["date_start"])
per_day = ucdp.resample("D", on="date_start").size()

fig, ax = plt.subplots(figsize=(9, 3))
per_day.plot(ax=ax)
ax.axvline(pd.Timestamp("2024-05-10"), color="red", linestyle="--", label="May 10")
ax.set_ylabel("events / day")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Part 2: Telegram
#
# ### 2a. How collection works (Telethon)
#
# Telegram's MTProto API gives an authenticated client the full history of
# any public channel. The `telethon` code below is complete and real — it
# needs the `api_id`/`api_hash` from my.telegram.org (participant guide)
# and a phone-number login the first time it runs.

# %%
# OPTIONAL — live network + Telegram credentials.
# The five public energy/infrastructure channels this lab uses — large,
# institutional, unambiguously public broadcast channels (the chapter
# discusses why we collect channels, not chats):
CHANNELS = ["Ukrenergo", "dtek_ua", "energoatom_ua", "dsns_telegram", "kyivoda"]

if os.environ.get("TELEGRAM_API_ID"):
    import nest_asyncio
    nest_asyncio.apply()        # lets Telethon's event loop run in Jupyter
    from telethon.sync import TelegramClient

    rows = []
    with TelegramClient("lab2", os.environ["TELEGRAM_API_ID"],
                        os.environ["TELEGRAM_API_HASH"]) as client:
        for ch in CHANNELS:
            for msg in client.iter_messages(ch, limit=200):
                rows.append({"channel": ch, "date": msg.date,
                             "text": msg.text, "views": msg.views,
                             "forwards": msg.forwards})
    live_tg = pd.DataFrame(rows)
    print(f"Collected {len(live_tg)} posts")
else:
    print("No TELEGRAM_API_ID in environment — skipping live collection.")

# %% [markdown]
# ### 2b. The cached channel export — read this cell, it matters
#
# This is a *real* Telethon export: 632 posts from five public Ukrainian
# energy/infrastructure channels, May-July 2024 (the spring energy-strike
# campaign). The five, and why each is here:
#
# - `Ukrenergo` — the national grid operator (outages, damage to the system)
# - `dtek_ua` — DTEK, the largest private utility
# - `energoatom_ua` — the state nuclear operator
# - `dsns_telegram` — the State Emergency Service (response at strike sites)
# - `kyivoda` — the Kyiv Oblast administration (regional government)
#
# `data/acquisition/get_telegram.py` shows exactly how it was pulled. Two
# caveats before you use it. First, these are real posts by real
# institutions — public broadcast, but still real words, so use them for
# source criticism, not to profile anyone. Second, the posts are in
# Ukrainian: the `text_en_mt` column is an English **machine translation**
# (done by an LLM, not human-checked). It's good enough to work with and
# wrong often enough that you should treat it as a rough guide rather than
# ground truth — we'll come back to machine-generated labels on Day 4.

# %%
tg = pd.read_parquet(os.path.join(DATA_DIR, "telegram_infra.parquet"))
print(f"{len(tg)} posts, {tg['channel'].nunique()} channels, "
      f"{tg['date'].min():%Y-%m-%d} → {tg['date'].max():%Y-%m-%d}")
tg.sample(5, random_state=1)[["channel", "date", "text", "text_en_mt", "views", "forwards"]]

# %% [markdown]
# ### 2c. Channel ecology
#
# Before reading any individual post, profile the *channels*: who posts
# how much, when, to whom? This is the metadata layer Telethon gives you
# for free, and it's analytically rich before any NLP happens.

# %%
import matplotlib.pyplot as plt

profile = tg.groupby("channel").agg(
    posts=("text", "size"),
    median_views=("views", "median"),
    fwd_rate=("forwards", lambda s: s.sum()),
).assign(fwd_per_1k_views=lambda d: 1000 * d.fwd_rate / tg.groupby("channel")["views"].sum())
profile

# %%
# Posting tempo: posts per day per channel (3-day rolling average).
tempo = (tg.set_index("date").groupby("channel")
           .resample("D").size().unstack(0).fillna(0))
tempo.rolling(3).mean().plot(figsize=(10, 4))
plt.ylabel("posts/day (3-day avg)")
plt.title("Posting tempo by channel")
plt.show()

# %%
# Exercise: hour-of-day posting profiles per channel (df.date.dt.hour,
# groupby, unstack, plot). Which channel never sleeps, and why does that
# make sense given its role?

# try it here

# %% [markdown]
# ### 2d. The forward graph: who amplifies whom
#
# Forwards are Telegram's citation network. The `fwd_from` column records
# where a forwarded post came from. Build the channel-to-source matrix and
# look at who these official channels amplify.
#
# The chapter described *laundering loops*: channel A cites B, B cites A, and
# repetition starts to look like confirmation. You won't find a clean loop
# among these five — they're official channels, so they forward *upstream*
# sources (the President, ministries, individual power plants), not each
# other. The related pattern you can find: a single source that several of
# these nominally-independent channels all relay. If a reader sees the same
# claim carried by five official channels, it's easy to read repetition as
# confirmation, even though it's one source amplified five times.

# %%
fwd_matrix = pd.crosstab(tg["channel"], tg["fwd_from"])
fwd_matrix

# %%
# Exercise: which single `fwd_from` source is forwarded by the most of our
# five channels? (Hint: `(fwd_matrix > 0).sum()` counts, for each source,
# how many channels forwarded it.) Pull a few of those forwarded posts and
# read them. When several official channels all carry the same source, what
# would a reader who sees it repeated conclude — and would they be right?

# try it here

# %% [markdown]
# ### 2e. Source criticism, operationalized
#
# Read 10 posts each (use the `text_en_mt` column) from `dtek_ua` — a
# private utility that wants customers and investors to see it as competent
# and in control — and from `dsns_telegram`, the State Emergency Service,
# whose posts foreground rescue and response. For each post, ask the
# chapter's question: what does this channel's *incentive structure* do to
# what it reports, and to what it leaves out? Then do the exercise that
# turns this from vibes into method:

# %%
# Exercise: add a column `claim_type` to 15 posts of your choosing, coded
# by hand as: "own_report" (channel reports own side's action/observation),
# "enemy_claim" (characterizes the other side), "relay" (forwards/quotes a
# third party), "warning/admin" (alerts, logistics). What's the mix per
# channel? Thursday morning the whole class hand-codes a shared sample and
# measures how much *human* coders disagree with each other; keep your
# claim_type notes, since the hard calls you run into here are what that
# exercise is about.

sample_posts = tg.sample(15, random_state=42).copy()

# try it here

# %% [markdown]
# ### 2f. Store it like you mean it
#
# Closing the loop with this afternoon's storage session: append-safe raw
# (the parquet is our "raw" here), plus a queryable SQLite copy.

# %%
import sqlite3

con = sqlite3.connect(os.path.join(OUTPUTS_DIR, "lab2.db"))
tg.astype({"date": str}).to_sql("messages", con, if_exists="replace", index=False)
pd.read_sql("""
    SELECT channel, COUNT(*) AS n, AVG(views) AS avg_views
    FROM messages GROUP BY channel ORDER BY n DESC
""", con)

# %% [markdown]
# ## Capstone variant
#
# 1. Which of this lab's two doors does your measurement target need —
#    a documented API (which one? what auth? what terms?) or platform
#    collection (which channels/accounts? public broadcast or something
#    more sensitive?)? Write down the access path and its constraints.
# 2. If Telegram is relevant to your target: list 3–5 candidate public
#    channels, and for each write ONE sentence on its incentive structure
#    (who runs it, what does it want you to believe?).
# 3. Pull or load *something* today — even 50 records — and save it with a
#    provenance note. Stand-up tomorrow: your source, your access path,
#    your first surprise.
#
# %% [markdown]
# ## If you finish early
#
# - Run `data/acquisition/get_acled.py` with your own ACLED credentials
#   and compare its Kharkiv May-2024 event count to the cached UCDP
#   extract's 529. (Thursday's lab does this comparison properly; getting
#   the raw counts tonight will make you appropriately suspicious early.)
# - Are channel views lognormal? The `views` here are real, so check
#   directly: a histogram of `np.log(tg["views"])` is a fast start. Do the
#   big institutional channels and the smaller ones differ in shape, or just
#   in scale?
# - Write the JSONL "collector" pattern from the storage chapter: a
#   function that appends each post as one JSON line, crash-safe, then a
#   reader that recovers cleanly from a truncated final line.
