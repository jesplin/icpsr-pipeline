# %% [markdown]
# # Lab 1: Scraping the Humanitarian Web
#
# ICPSR 2026 -- The Social Science Data Pipeline
# Instructor: Andy Halterman
#
# In this lab you'll build a complete web scraper for ReliefWeb, the UN
# humanitarian information service, collecting its stream of Ukraine
# situation updates. By the end you'll have a parser for listing pages, a
# pagination loop, a parser for article pages, a saved parquet dataset, and
# some practice delegating scraper-writing to an LLM and checking its work.
#
# Why ReliefWeb? We checked (and you should always check): its robots.txt
# allows crawling the listing and report pages, its terms permit downloading
# for non-commercial use, the HTML is server-rendered and stable, and it
# doesn't fight automated clients.

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

# %%
!pip install -q requests beautifulsoup4 lxml pandas pyarrow openai

# %% [markdown]
# ## Offline-first: how this lab (and every lab) handles the network
#
# Live websites are unreliable under classroom conditions: rate limits, slow
# responses, the occasional outage right when you need the page. So we
# snapshotted ReliefWeb's listing and article pages into
# `data/cached/reliefweb_html/` (the script that did it is
# `data/acquisition/get_reliefweb_pages.py` -- four short functions, read it
# later). Everything below runs from those files; the cells that touch the
# live site are clearly marked **OPTIONAL -- live network**.
#
# This is the Day 1 storage lesson in action: *keep the raw HTML*. Our
# parsers run against saved files, so when we improve a parser we don't
# re-download anything.

# %%
CACHE = os.path.join(DATA_DIR, "reliefweb_html")
listing_html = open(os.path.join(CACHE, "listing_page_0.html")).read()
print(f"{len(listing_html):,} characters of HTML")
listing_html[:500]

# %% [markdown]
# ## Step 1: From HTML soup to a tree
#
# That wall of text is a *tree*: nested tags with attributes. BeautifulSoup
# parses it, and **CSS selectors** are how we address parts of it:
#
# - `article`: every `<article>` tag
# - `.rw-river-article`: anything with that class
# - `h3 a`: every link *inside* an `<h3>`
#
# How did we know ReliefWeb wraps each listed report in
# `<article class="rw-river-article ...">`? Browser dev tools: open the
# listing page, right-click a report title, "Inspect," and read the tags
# around it. That's the actual workflow, most of the time, on most sites.

# %%
from bs4 import BeautifulSoup

soup = BeautifulSoup(listing_html)
articles = soup.select("article.rw-river-article")
print(f"Found {len(articles)} articles on this listing page")

# %%
# Look at one. .prettify() re-indents the HTML so humans can read it.
print(articles[0].prettify()[:1200])

# %%
# Exercise (required -- the next section is off-limits until these three
# are filled in): find where the TITLE, the SOURCE (publishing
# organization), and the DATE live inside an article element. The method,
# not the answer: read the prettify() output above, or open the listing
# page in your browser, right-click each piece of information, and Inspect.
# Read the tag and class around it, write a selector, test it on
# articles[0] with the cell below, then check it matches on EVERY article
# on the page.

# %%
TITLE_SELECTOR = ""    # fill in, e.g. "tag.some-class a"
SOURCE_SELECTOR = ""   # fill in
DATE_SELECTOR = ""     # fill in

for name, sel in [("title", TITLE_SELECTOR), ("source", SOURCE_SELECTOR),
                  ("date", DATE_SELECTOR)]:
    if not sel:
        print(f"{name:7s}: selector not written yet")
        continue
    hits = [a.select_one(sel) for a in articles]
    n_match = sum(h is not None for h in hits)
    first = hits[0].get_text(strip=True) if hits[0] else "NO MATCH on articles[0]"
    print(f"{name:7s}: {n_match}/{len(articles)} articles match -- {first}")

# %% [markdown]
# ## Step 2: One article, one dictionary
#
# A scraper is just a function from HTML to records. For each article we
# want title, URL, the publishing organization, the document format, and
# the date. Compare your three selectors to the ones below: if yours differ
# but matched every article, yours are fine, there's usually more than one
# working selector for a field. Two details worth noticing:
#
# - The date is in a `<time datetime="2026-06-10T...">` attribute
#   (machine-readable ISO format, far better than parsing "10 Jun 2026").
#   Always prefer attributes over displayed text when both exist.
# - We use `.select_one()` plus a None-check rather than assuming every
#   field exists. Real listings have gaps, and a scraper that crashes on
#   record #847 of 20,000 is a scraper you get to babysit.

# %%
def parse_listing_item(article):
    """Turn one <article> element into a dict (None for missing fields)."""
    title_el = article.select_one("h3.rw-river-article__title a")
    source_el = article.select_one("dd.rw-entity-meta__tag-value--source")
    format_el = article.select_one("dd.rw-entity-meta__tag-value--format")
    time_el = article.select_one("time")
    return {
        "title": title_el.get_text(strip=True) if title_el else None,
        "url": title_el["href"] if title_el else None,
        "source": source_el.get_text(strip=True) if source_el else None,
        "format": format_el.get_text(strip=True) if format_el else None,
        "date": time_el["datetime"] if time_el else None,
    }

parse_listing_item(articles[0])

# %%
import pandas as pd

records = [parse_listing_item(a) for a in articles]
df = pd.DataFrame(records)
df.head()

# %% [markdown]
# ## Step 3: Pagination
#
# A single page is enough to test the parser, but the loop over pages is
# what actually builds the dataset. ReliefWeb paginates with a `&page=N`
# URL parameter (look at the cached filenames -- we snapshotted pages 0-2).
# The pattern below works on the cache; the live version just swaps
# `read_cached_page` for a `requests.get` call.
#
# Note the stopping condition: we stop when a page yields nothing, never at
# a hardcoded page count, since sites grow and shrink over time.

# %%
def read_cached_page(page_num):
    """Return the HTML of cached listing page N, or None if not cached."""
    path = os.path.join(CACHE, f"listing_page_{page_num}.html")
    if not os.path.exists(path):
        return None
    return open(path).read()

all_records = []
page = 0
while True:
    html = read_cached_page(page)
    if html is None:
        print(f"No page {page} in cache -- stopping.")
        break
    page_articles = BeautifulSoup(html).select("article.rw-river-article")
    all_records.extend(parse_listing_item(a) for a in page_articles)
    print(f"Page {page}: {len(page_articles)} articles")
    page += 1

df = pd.DataFrame(all_records)
print(f"\nTotal: {len(df)} records")

# %% [markdown]
# ### OPTIONAL -- live network: the same loop against the real site
#
# Three changes turn the cache loop into a live scraper, and all three are
# *politeness infrastructure*:
#
# 1. A User-Agent that says who you are and how to reach you. Anonymous
#    high-volume clients are what get IP ranges banned.
# 2. `raise_for_status()`: a 404 or 503 should be an ERROR, not silently
#    parsed as an empty page.
# 3. `time.sleep(2)` between requests, since you're a guest on someone
#    else's server, not the owner.

# %%
# OPTIONAL -- live network. Skip freely; everything later uses the cache.
import time
import requests

HEADERS = {"User-Agent": "icpsr-pipeline-lab (graduate course; your_email@here)"}
LISTING_URL = "https://reliefweb.int/updates?advanced-search=%28PC241%29&page={n}"

live_records = []
for page in range(2):                      # just 2 pages; it's a demo
    resp = requests.get(LISTING_URL.format(n=page), headers=HEADERS, timeout=30)
    resp.raise_for_status()
    page_articles = BeautifulSoup(resp.text).select("article.rw-river-article")
    live_records.extend(parse_listing_item(a) for a in page_articles)
    print(f"Live page {page}: {len(page_articles)} articles")
    time.sleep(2)

# %% [markdown]
# ## Step 4: Make emptiness fail loudly
#
# The failure mode that should worry you more than any crash: ReliefWeb
# redesigns their site, `article.rw-river-article` matches nothing, and
# your scraper returns zero records per page, forever, with no error. A
# crash gets noticed right away; a selector that quietly stops matching can
# run for weeks before anyone looks at the counts and asks why they
# dropped. So we make emptiness a crash:

# %%
def parse_listing_page(html, min_expected=1):
    """Parse a listing page, refusing to fail silently."""
    page_articles = BeautifulSoup(html).select("article.rw-river-article")
    if len(page_articles) < min_expected:
        raise ValueError(
            f"Parsed only {len(page_articles)} articles -- selector broken "
            "or page structure changed. Refusing to continue silently."
        )
    records = [parse_listing_item(a) for a in page_articles]
    n_missing_titles = sum(r["title"] is None for r in records)
    if n_missing_titles > len(records) * 0.1:
        raise ValueError(f"{n_missing_titles}/{len(records)} records missing "
                         "titles -- field selectors likely broken.")
    return records

# Works on good input:
print(f"{len(parse_listing_page(listing_html))} records parsed")

# %%
# Exercise: confirm it FAILS on bad input. Feed parse_listing_page some
# HTML that isn't a ReliefWeb listing (e.g. "<html><body>hi</body></html>")
# and check you get the ValueError, not an empty list.

# try it here

# %% [markdown]
# ## Step 5: Article pages
#
# The listing gives us metadata; the report pages have the actual text. We
# cached 8 of them in `reliefweb_html/articles/`. Article pages have their
# own structure (dev tools again): title in `<h1 class="rw-article__title">`,
# body in `<div class="rw-article__content">`.

# %%
import glob

def parse_article(html):
    soup = BeautifulSoup(html)
    title_el = soup.select_one("h1.rw-article__title")
    body_el = soup.select_one("div.rw-article__content")
    source_el = soup.select_one("dd.rw-entity-meta__tag-value--source")
    time_el = soup.select_one("time")
    return {
        "title": title_el.get_text(strip=True) if title_el else None,
        "source": source_el.get_text(strip=True) if source_el else None,
        "date": time_el["datetime"] if time_el else None,
        "text": body_el.get_text(" ", strip=True) if body_el else None,
    }

article_files = sorted(glob.glob(os.path.join(CACHE, "articles", "*.html")))
articles_df = pd.DataFrame([parse_article(open(f).read()) for f in article_files])
articles_df["n_words"] = articles_df["text"].str.split().str.len()
articles_df[["title", "source", "date", "n_words"]]

# %% [markdown]
# ## Step 6: Save it properly
#
# Parquet, not CSV (Day 1 storage session: typed, compressed, fast), plus
# the provenance habit: where, when, how.

# %%
import datetime
import json

df.to_parquet(os.path.join(OUTPUTS_DIR, "reliefweb_listing.parquet"))
articles_df.to_parquet(os.path.join(OUTPUTS_DIR, "reliefweb_articles.parquet"))

provenance = {
    "source": "reliefweb.int Ukraine updates (country facet PC241)",
    "collected_via": "cached snapshot in data/cached/reliefweb_html/",
    "snapshot_script": "data/acquisition/get_reliefweb_pages.py",
    "parsed_at": datetime.datetime.now().isoformat(),
    "terms": "ReliefWeb terms permit download/copy for non-commercial use",
}
with open(os.path.join(OUTPUTS_DIR, "reliefweb_provenance.json"), "w") as f:
    json.dump(provenance, f, indent=2)

print("Saved listing, articles, and provenance to outputs/")

# %% [markdown]
# ## Step 7: Break it, then fix it -- with and without an LLM
#
# Scraper-writing is the course's first example of *LLM-delegable work*:
# tedious, pattern-heavy, and instantly verifiable. Time to practice the
# verification part.
#
# Below is a parser someone wrote for ReliefWeb. It looks plausible. It
# returns garbage. **First, fix it by hand**: use the cached HTML (print
# `articles[0].prettify()`) to find what the selectors should be. Time
# yourself.

# %%
def broken_parse(article):
    """This parser is broken in two ways. Find them both."""
    title_el = article.select_one("h2.river-article__title a")     # bug 1
    date_el = article.select_one("span.date")                       # bug 2
    return {
        "title": title_el.get_text(strip=True) if title_el else None,
        "date": date_el.get_text(strip=True) if date_el else None,
    }

# Run it -- notice it doesn't crash. It just returns Nones. (Silent failure
# again. If you wrapped this in parse_listing_page's assertions, it would
# have refused to run. That's the point of Step 4.)
broken_parse(articles[0])

# %%
# Fix by hand here:

# try it here

# %% [markdown]
# Now the LLM route. We'll use the course `chat()` helper (we open this box
# properly on Day 3; today it's just "send text, get text"). The recipe
# that works: give the model (1) a *real chunk of the HTML* and (2) the
# broken code, and ask what's wrong. Always include the actual HTML, not a
# description of it -- without it, models confidently invent selectors for
# pages they've never seen.

# %%
import sys
sys.path.insert(0, COURSE_DIR)
from course_utils import chat

html_sample = articles[0].prettify()[:3000]   # a real sample, not a description

prompt = f"""Here is a sample <article> element from a page I'm scraping:

{html_sample}

This parser returns None for every field:

def broken_parse(article):
    title_el = article.select_one("h2.river-article__title a")
    date_el = article.select_one("span.date")
    ...

What's wrong, and what should the selectors be? Answer briefly."""

print(chat(prompt))

# %%
# Did the model get it right? Verify its suggested selectors actually work
# on all 20 articles before believing it. It's a three-line check, and
# skipping it is how "LLM-assisted" quietly turns into "LLM-dependent."

# try it here

# %% [markdown]
# ## Capstone variant
#
# For *your* measurement target, find one web source worth collecting
# (government statements, NGO reports, a news section, an NGO casualty
# tracker...). Before writing any code:
#
# 1. Check its robots.txt and terms. Write one sentence: are you allowed?
# 2. Identify the listing structure and pagination mechanism in dev tools.
# 3. Scrape 2-3 listing pages politely (or snapshot them first, like our
#    acquisition script does), parse title/date/link, save parquet +
#    provenance JSON.
# 4. Bring to stand-up tomorrow: your source, your one-sentence legality
#    call, and one thing that surprised you in the HTML.
#
# If your capstone source is hostile to scraping (JS-rendered, login-walled,
# ToS-prohibited), bring that to stand-up as-is: what you tried, why it
# didn't work, and we'll talk about Plan B (APIs? upstream sources? cached
# archives?).

# %% [markdown]
# ## If you finish early
#
# - Our backup target was Ukrinform's war section
#   (`ukrinform.net/rubric-ato?page=N`: classic query-string pagination,
#   permissive robots.txt). Write `parse_ukrinform_listing()` from scratch:
#   dev tools, selectors, assertions, the whole workflow. How much faster
#   was the second site?
# - Write `check_robots(url)`: fetch robots.txt from any domain and report
#   whether a given path is allowed for `User-agent: *`. (The standard
#   library has `urllib.robotparser` -- compare your reading of the raw
#   file against its verdict.)
# - Look at `data/acquisition/get_reliefweb_pages.py`. It has a subtle
#   politeness feature this lab's live loop lacks. What is it?
