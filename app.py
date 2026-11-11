import os
import pickle

import pandas as pd
import requests
import streamlit as st

# Ensure streamlit is installed and available in the environment

# =============================
# CONFIG
# =============================
API_BASE = "http://127.0.0.1:8000"  # Use local server for development
TMDB_IMG = "https://image.tmdb.org/t/p/w500"
PLACEHOLDER_POSTER = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA1MDAgNzUwIj4KICA8ZGVmcz4KICAgIDxsaW5lYXJHcmFkaWVudCBpZD0iZyIgeDE9IjAiIHgyPSIxIiB5MT0iMCIgeTI9IjEiPgogICAgICA8c3RvcCBvZmZzZXQ9IjAlIiBzdG9wLWNvbG9yPSIjMGYxNzJhIi8+CiAgICAgIDxzdG9wIG9mZnNldD0iMTAwJSIgc3RvcC1jb2xvcj0iIzMzNDE1NSIvPgogICAgPC9saW5lYXJHcmFkaWVudD4KICA8L2RlZnM+CiAgPHJlY3Qgd2lkdGg9IjUwMCIgaGVpZ2h0PSI3NTAiIGZpbGw9InVybCgjZykiLz4KICA8cmVjdCB4PSIyMCIgeT0iMjAiIHdpZHRoPSI0NjAiIGhlaWdodD0iNzEwIiByeD0iMTYiIGZpbGw9Im5vbmUiIHN0cm9rZT0iI2NiZDVlMSIgc3Ryb2tlLXdpZHRoPSI0Ii8+CiAgPHRleHQgeD0iMjUwIiB5PSIzMzAiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtc2l6ZT0iNDIiIGZvbnQtZmFtaWx5PSJBcmlhbCwgYW5zdGFsIiwgZmlsbD0iI2UyZThmMCIgZm9udC13ZWlnaHQ9ImJvbGQiPk1PVklFPC90ZXh0PgogIDx0ZXh0IHg9IjI1MCIgeT0iMzkwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LXNpemU9IjQyIiBmb250LWZhbWlseT0iQXJpYWwsIGFuc3RhbCIsIGZpbGw9IiNlMmU4ZjAiIGZvbnQtd2VpZ2h0PSJib2xkIj5QT1NURVI8L3RleHQ+CiAgPHRleHQgeD0iMjUwIiB5PSI2NjAiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtc2l6ZT0iMjIiIGZvbnQtZmFtaWx5PSJBcmlhbCwgYW5zdGFsIiBmaWxsPSIjY2JkNWUxIj5UT1BTRU9MSU1FPC90ZXh0PgogPC9zdmc+"

LOCAL_DF_PATH = os.path.join(os.path.dirname(__file__), "df.pkl")
LOCAL_CSV_PATH = os.path.join(os.path.dirname(__file__), "movies_metadata.csv")
TMDB_ATTRIBUTION = "This product uses the TMDb API but is not endorsed or certified by TMDb."


def _clean_text(value):
    if value is None or pd.isna(value):
        return None
    cleaned = str(value).strip()
    if not cleaned or cleaned.lower() in {"nan", "none", "null"}:
        return None
    return cleaned


def _normalize_image_url(value):
    value = _clean_text(value)
    if not value:
        return None
    if value.startswith(("http://", "https://", "data:")):
        return value
    if value.startswith("/"):
        return f"{TMDB_IMG}{value}"
    return value


def _safe_int(value, default=None):
    try:
        if value is None or pd.isna(value):
            return default
        if isinstance(value, str):
            value = value.strip()
            if not value or value.lower() in {"nan", "none", "null"}:
                return default
        return int(value)
    except (TypeError, ValueError):
        return default


@st.cache_data(ttl=3600)
def validate_image_url(url):
    value = _normalize_image_url(url)
    if not value:
        return False
    if value.startswith("data:"):
        return True
    try:
        head = requests.head(value, timeout=15, allow_redirects=True)
        if head.status_code in {200, 204}:
            return True
        if head.status_code in {403, 405}:
            get_resp = requests.get(value, timeout=15, stream=True)
            return get_resp.status_code in {200, 204}
        return False
    except requests.RequestException:
        try:
            resp = requests.get(value, timeout=15, stream=True)
            return resp.status_code in {200, 204}
        except requests.RequestException:
            return False


def poster_or_placeholder(url):
    value = _normalize_image_url(url)
    if value and validate_image_url(value):
        return value
    return PLACEHOLDER_POSTER


def load_local_movie_df():
    csv_path = LOCAL_CSV_PATH
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, low_memory=False)
            if not df.empty:
                if "title" in df.columns:
                    df["title"] = df["title"].fillna(df.get("original_title", pd.Series([None] * len(df))))
                    df["title"] = df["title"].map(_clean_text)
                if "original_title" in df.columns:
                    df["original_title"] = df["original_title"].map(_clean_text)
                if "poster_path" in df.columns:
                    df["poster_path"] = df["poster_path"].map(_clean_text)
                if "backdrop_path" in df.columns:
                    df["backdrop_path"] = df["backdrop_path"].map(_clean_text)
                return df
        except Exception:
            pass
    try:
        with open(LOCAL_DF_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _movie_title_from_row(row):
    for key in ("title", "original_title", "name"):
        value = _clean_text(row.get(key))
        if value:
            return value
    return "Untitled"


def _as_movie_detail(row, fallback_index):
    title = _movie_title_from_row(row)
    if not title or title == "Untitled":
        return None

    tmdb_id = _safe_int(row.get("tmdb_id"), _safe_int(row.get("id"), fallback_index))
    if tmdb_id is None:
        tmdb_id = fallback_index

    poster_url = _normalize_image_url(row.get("poster_url") or row.get("poster_path"))
    backdrop_url = _normalize_image_url(
        row.get("backdrop_url")
        or row.get("backdrop_path")
        or row.get("poster_url")
        or row.get("poster_path")
    ) or poster_url

    return {
        "tmdb_id": int(tmdb_id),
        "title": title,
        "overview": row.get("overview") or "",
        "release_date": row.get("release_date") or "",
        "poster_url": poster_or_placeholder(poster_url),
        "backdrop_url": poster_or_placeholder(backdrop_url),
        "genres": [],
        "vote_average": row.get("vote_average"),
    }


def local_home_cards(category: str = "popular", limit: int = 24):
    df = load_local_movie_df()
    if df is None:
        return []

    subset = df.copy()
    if "title" in subset.columns:
        subset["title"] = subset["title"].fillna(subset.get("original_title", pd.Series([None] * len(subset))))
        subset = subset[subset["title"].astype(str).str.strip().ne("")]
    if "popularity" in subset.columns:
        subset["popularity"] = pd.to_numeric(subset["popularity"], errors="coerce")
        subset = subset.sort_values("popularity", ascending=False)
    subset = subset.head(limit)

    cards = []
    for idx, row in subset.iterrows():
        movie = _as_movie_detail(row, idx)
        if movie:
            cards.append({
                "tmdb_id": movie["tmdb_id"],
                "title": movie["title"],
                "poster_url": movie["poster_url"],
                "backdrop_url": movie["backdrop_url"],
                "release_date": movie["release_date"],
                "overview": movie["overview"],
                "vote_average": movie["vote_average"],
            })
    return cards


def local_search_cards(query: str, limit: int = 24):
    df = load_local_movie_df()
    if df is None:
        return []

    q = (query or "").strip().lower()
    subset = df.copy()
    if "title" in subset.columns:
        subset["title"] = subset["title"].fillna(subset.get("original_title", pd.Series([None] * len(subset))))
        subset = subset[subset["title"].astype(str).str.strip().ne("")]
    if q:
        subset = subset[subset["title"].astype(str).str.contains(q, case=False, na=False)]
    subset = subset.head(limit)

    cards = []
    for idx, row in subset.iterrows():
        movie = _as_movie_detail(row, idx)
        if movie:
            cards.append({
                "tmdb_id": movie["tmdb_id"],
                "title": movie["title"],
                "poster_url": movie["poster_url"],
                "backdrop_url": movie["backdrop_url"],
                "release_date": movie["release_date"],
                "overview": movie["overview"],
                "vote_average": movie["vote_average"],
            })
    return cards


def local_details_by_id(tmdb_id: int):
    df = load_local_movie_df()
    if df is None:
        return None

    try:
        id_col = None
        for candidate in ("tmdb_id", "id"):
            if candidate in df.columns:
                id_col = candidate
                break

        if id_col:
            match = df[df[id_col].astype(str).str.strip().str.lower().eq(str(tmdb_id).strip().lower())]
            if not match.empty:
                row = match.iloc[0]
            else:
                row = df.iloc[int(tmdb_id)] if 0 <= int(tmdb_id) < len(df) else None
        else:
            row = df.iloc[int(tmdb_id)] if 0 <= int(tmdb_id) < len(df) else None
    except (IndexError, TypeError, ValueError):
        return None

    if row is None:
        return None

    movie = _as_movie_detail(row, tmdb_id)
    if movie is None:
        return None
    return movie


# =============================
# STATE + ROUTING (single-file pages)
# =============================
if "view" not in st.session_state:
    st.session_state.view = "home"  # home | details
if "selected_tmdb_id" not in st.session_state:
    st.session_state.selected_tmdb_id = None

qp_view = st.query_params.get("view")
qp_id = st.query_params.get("id")
if qp_view in ("home", "details"):
    st.session_state.view = qp_view
if qp_id:
    try:
        st.session_state.selected_tmdb_id = int(qp_id)
        st.session_state.view = "details"
    except (TypeError, ValueError):
        pass


def goto_home():
    st.session_state.view = "home"
    st.query_params["view"] = "home"
    if "id" in st.query_params:
        del st.query_params["id"]
    st.rerun()


def goto_details(tmdb_id: int):
    st.session_state.view = "details"
    st.session_state.selected_tmdb_id = int(tmdb_id)
    st.query_params["view"] = "details"
    st.query_params["id"] = str(int(tmdb_id))
    st.rerun()


# =============================
# API HELPERS
# =============================
@st.cache_data(ttl=30)  # short cache for autocomplete
def api_get_json(path: str, params: dict | None = None):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=25)
        if r.status_code >= 400:
            return None, f"HTTP {r.status_code}: {r.text[:300]}"
        return r.json(), None
    except requests.RequestException as exc:
        if path == "/home":
            return local_home_cards(params.get("category", "popular") if params else "popular", int(params.get("limit", 24)) if params else 24), None
        if path == "/tmdb/search":
            q = (params or {}).get("query", "")
            return local_search_cards(q, limit=24), None
        if path.startswith("/movie/id/"):
            tmdb_id = int(path.rsplit("/", 1)[-1])
            return local_details_by_id(tmdb_id), None
        return None, f"Request failed: {exc}"


def render_poster(movie):
    poster = poster_or_placeholder(movie.get("poster_url"))
    st.image(poster, width="stretch")


def render_movie_card(movie, key_prefix, row, col):
    tmdb_id = movie.get("tmdb_id")
    title = movie.get("title", "Untitled")
    movie_key = f"{key_prefix}_{row}_{col}_{tmdb_id}"

    render_poster(movie)
    if st.button("Open", key=movie_key) and tmdb_id:
        goto_details(tmdb_id)
    st.markdown(f"<div class='movie-title'>{title}</div>", unsafe_allow_html=True)


def poster_grid(cards, cols=6, key_prefix="grid"):
    if not cards:
        st.info("No movies to show.")
        return

    rows = (len(cards) + cols - 1) // cols
    idx = 0
    for row in range(rows):
        colset = st.columns(cols)
        for col in range(cols):
            if idx >= len(cards):
                break
            with colset[col]:
                render_movie_card(cards[idx], key_prefix, row, col)
            idx += 1


def to_cards_from_tfidf_items(tfidf_items):
    cards = []
    for x in tfidf_items or []:
        tmdb = x.get("tmdb") or {}
        if tmdb.get("tmdb_id"):
            cards.append(
                {
                    "tmdb_id": tmdb["tmdb_id"],
                    "title": tmdb.get("title") or x.get("title") or "Untitled",
                    "poster_url": tmdb.get("poster_url"),
                }
            )
    return cards


# =============================
# IMPORTANT: Robust TMDB search parsing
# Supports BOTH API shapes:
# 1) raw TMDB: {"results":[{id,title,poster_path,...}]}
# 2) list cards: [{tmdb_id,title,poster_url,...}]
# =============================
def parse_tmdb_search_to_cards(data, keyword: str, limit: int = 24):
    """
    Returns:
      suggestions: list[(label, tmdb_id)]
      cards: list[{tmdb_id,title,poster_url}]
    """
    keyword_l = keyword.strip().lower()

    # A) If API returns dict with 'results'
    if isinstance(data, dict) and "results" in data:
        raw = data.get("results") or []
        raw_items = parse_raw_results(raw)

    # B) If API returns already as list
    elif isinstance(data, list):
        raw_items = parse_list_results(data)
    else:
        return [], []

    # Word-match filtering (contains)
    matched = [x for x in raw_items if keyword_l in x["title"].lower()]

    # If nothing matched, fallback to raw list (so never blank)
    final_list = matched if matched else raw_items

    # Suggestions = top 10 labels
    suggestions = []
    for x in final_list[:10]:
        year = (x.get("release_date") or "")[:4]
        label = f"{x['title']} ({year})" if year else x["title"]
        suggestions.append((label, x["tmdb_id"]))

    # Cards = top N
    cards = [
        {"tmdb_id": x["tmdb_id"], "title": x["title"], "poster_url": x["poster_url"]}
        for x in final_list[:limit]
    ]
    return suggestions, cards


def parse_raw_results(raw):
    raw_items = []
    for m in raw:
        title = (m.get("title") or "").strip()
        tmdb_id = m.get("id")
        poster_path = m.get("poster_path")
        if not title or not tmdb_id:
            continue
        raw_items.append(
            {
                "tmdb_id": int(tmdb_id),
                "title": title,
                "poster_url": f"{TMDB_IMG}{poster_path}" if poster_path else None,
                "release_date": m.get("release_date", ""),
            }
        )
    return raw_items


def parse_list_results(data):
    raw_items = []
    for m in data:
        # might be {tmdb_id,title,poster_url}
        tmdb_id = m.get("tmdb_id") or m.get("id")
        title = (m.get("title") or "").strip()
        poster_url = m.get("poster_url")
        if not title or not tmdb_id:
            continue
        raw_items.append(
            {
                "tmdb_id": int(tmdb_id),
                "title": title,
                "poster_url": poster_url,
                "release_date": m.get("release_date", ""),
            }
        )
    return raw_items


# =============================
# SIDEBAR (clean)
# =============================
with st.sidebar:
    st.markdown("## 🎬 Menu")
    if st.button("🏠 Home"):
        goto_home()

    st.markdown("---")
    st.markdown("### 🏠 Home Feed (only home)")
    home_category = st.selectbox(
        "Category",
        ["trending", "popular", "top_rated", "now_playing", "upcoming"],
        index=0,
    )
    grid_cols = st.slider("Grid columns", 4, 8, 6)

# =============================
# HEADER
# =============================
st.title("🎬 Movie Recommender")
st.caption(TMDB_ATTRIBUTION)
st.markdown(
    "<div class='small-muted'>Type keyword → dropdown suggestions + matching results → open → details + recommendations</div>",
    unsafe_allow_html=True,
)
st.divider()

# ==========================================================
# VIEW: HOME
# ==========================================================
if st.session_state.get("view") == "home":
    typed = st.text_input(
        "Search by movie title (keyword)", placeholder="Type: avenger, batman, love..."
    )

    st.divider()

    # SEARCH MODE (Autocomplete + word-match results)
    if typed.strip():
        if len(typed.strip()) < 2:
            st.caption("Type at least 2 characters for suggestions.")
        else:
            data, err = api_get_json("/tmdb/search", params={"query": typed.strip()})

            if err or data is None:
                st.error(f"Search failed: {err}")
            else:
                suggestions, cards = parse_tmdb_search_to_cards(
                    data, typed.strip(), limit=24
                )

                # Dropdown
                if suggestions:
                    labels = ["-- Select a movie --"] + [s[0] for s in suggestions]
                    selected = st.selectbox("Suggestions", labels, index=0)

                    if selected != "-- Select a movie --":
                        # map label -> id
                        label_to_id = {s[0]: s[1] for s in suggestions}
                        goto_details(label_to_id[selected])
                else:
                    st.info("No suggestions found. Try another keyword.")

                st.markdown("### Results")
                poster_grid(cards, cols=grid_cols, key_prefix="search_results")

        st.stop()

    # HOME FEED MODE
    st.markdown(f"### 🏠 Home — {home_category.replace('_',' ').title()}")

    home_cards, err = api_get_json(
        "/home", params={"category": home_category, "limit": 24}
    )
    if err or not home_cards:
        st.error(f"Home feed failed: {err or 'Unknown error'}")
        st.stop()

    poster_grid(home_cards, cols=grid_cols, key_prefix="home_feed")

# ==========================================================
# VIEW: DETAILS
# ==========================================================
elif st.session_state.get("view") == "details":
    tmdb_id = st.session_state.get("selected_tmdb_id")
    if not tmdb_id:
        st.warning("No movie selected.")
        if st.button("← Back to Home"):
            goto_home()
        st.stop()

    # Top bar
    a, b = st.columns([3, 1])
    with a:
        st.markdown("### 📄 Movie Details")
    with b:
        if st.button("← Back to Home"):
            goto_home()

    # Details (your FastAPI safe route)
    data, err = api_get_json(f"/movie/id/{tmdb_id}")
    if err or not data:
        st.error(f"Could not load details: {err or 'Unknown error'}")
        st.stop()

    # Layout: Poster LEFT, Details RIGHT
    left, right = st.columns([1, 2.4], gap="large")

    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.image(poster_or_placeholder(data.get("poster_url")), width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"## {data.get('title','')}")
        release = data.get("release_date") or "-"
        genres = ", ".join([g["name"] for g in data.get("genres", [])]) or "-"
        st.markdown(
            f"<div class='small-muted'>Release: {release}</div>", unsafe_allow_html=True
        )
        st.markdown(
            f"<div class='small-muted'>Genres: {genres}</div>", unsafe_allow_html=True
        )
        st.markdown("---")
        st.markdown("### Overview")
        st.write(data.get("overview") or "No overview available.")
        st.markdown("</div>", unsafe_allow_html=True)

    if data.get("backdrop_url"):
        st.markdown("#### Backdrop")
        st.image(data["backdrop_url"], width="stretch")

    st.divider()
    st.markdown("### ✅ Recommendations")

    # Recommendations (TF-IDF + Genre) via your bundle endpoint
    title = (data.get("title") or "").strip()
    if title:
        bundle, err2 = api_get_json(
            "/movie/search",
            params={"query": title, "tfidf_top_n": 12, "genre_limit": 12},
        )

        if not err2 and bundle:
            st.markdown("#### 🔎 Similar Movies (TF-IDF)")
            poster_grid(
                to_cards_from_tfidf_items(bundle.get("tfidf_recommendations")),
                cols=grid_cols,
                key_prefix="details_tfidf",
            )

            st.markdown("#### 🎭 More Like This (Genre)")
            poster_grid(
                bundle.get("genre_recommendations", []),
                cols=grid_cols,
                key_prefix="details_genre",
            )
        else:
            st.info("Showing Genre recommendations (fallback).")
            genre_only, err3 = api_get_json(
                "/recommend/genre", params={"tmdb_id": tmdb_id, "limit": 18}
            )
            if not err3 and genre_only:
                poster_grid(
                    genre_only, cols=grid_cols, key_prefix="details_genre_fallback"
                )
            else:
                st.warning("No recommendations available right now.")
    else:
        st.warning("No title available to compute recommendations.")