import base64
import os
import pickle
from html import escape
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# ENV
# =========================
load_dotenv()
TMDB_API_KEY = (os.getenv("TMDB_API_KEY") or "").strip()

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"


class TMDBMovieCard(BaseModel):
    tmdb_id: int
    title: str
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None
    overview: Optional[str] = None


class TMDBMovieDetails(BaseModel):
    tmdb_id: int
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[Dict[str, Any]] = []
    vote_average: Optional[float] = None


class TFIDFRecItem(BaseModel):
    title: str
    score: float
    tmdb: Optional[TMDBMovieCard] = None


class SearchBundleResponse(BaseModel):
    query: str
    movie_details: TMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecItem] = []
    genre_recommendations: List[TMDBMovieCard] = []


TITLE_TO_IDX: Optional[Dict[str, int]] = None


def _norm_title(title: str) -> str:
    return (title or "").strip().lower()


def make_img_url(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    if isinstance(path, (float, int)) and pd.isna(path):
        return None
    value = str(path).strip()
    if not value or value.lower() in {"nan", "none", "null"}:
        return None
    if value.startswith(("http://", "https://", "data:")):
        return value
    if value.startswith("/"):
        return f"{TMDB_IMG_500}{value}"
    return value


def tmdb_is_configured() -> bool:
    return bool(TMDB_API_KEY) and TMDB_API_KEY.lower() not in {"placeholder", "dummy", "changeme"}


# =========================
# FASTAPI APP
# =========================
app = FastAPI(title="Movie Recommender API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local streamlit
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# PICKLE GLOBALS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DF_PATH = os.path.join(BASE_DIR, "df.pkl")
CSV_PATH = os.path.join(BASE_DIR, "movies_metadata.csv")
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")

df: Optional[pd.DataFrame] = None
indices_obj: Any = None
tfidf_matrix: Any = None
tfidf_obj: Any = None
LOCAL_MOVIE_DF: Optional[pd.DataFrame] = None


def _local_movie_df() -> Optional[pd.DataFrame]:
    if LOCAL_MOVIE_DF is not None and not LOCAL_MOVIE_DF.empty:
        return LOCAL_MOVIE_DF
    return df


def _row_to_movie_id(row: Dict[str, Any], fallback: int) -> int:
    raw = row.get("id")
    if raw is None or pd.isna(raw):
        return int(fallback)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(fallback)


def _safe_media_path(row: Dict[str, Any], key: str) -> Optional[str]:
    value = row.get(key)
    if value is None or pd.isna(value):
        return None
    value = str(value).strip()
    if not value or value.lower() in {"nan", "none", "null"}:
        return None
    return value


def _local_search_results(query: str, limit: int = 20) -> List[dict]:
    local_df = _local_movie_df()
    if local_df is None:
        return []

    subset = local_df.copy()
    if query:
        subset = subset[subset["title"].astype(str).str.contains(query, case=False, na=False)]
    subset = subset.head(limit)

    results: List[dict] = []
    for idx, row in subset.iterrows():
        movie_id = _row_to_movie_id(row, int(idx))
        poster_path = _safe_media_path(row, "poster_path") or _safe_media_path(row, "poster_url")
        results.append(
            {
                "id": int(movie_id),
                "title": str(row.get("title") or ""),
                "poster_path": make_img_url(poster_path) or local_placeholder_poster(str(row.get("title") or "")),
                "release_date": row.get("release_date") or "",
                "vote_average": row.get("vote_average"),
            }
        )
    return results


def _local_movie_details(movie_id: int) -> Optional[TMDBMovieDetails]:
    local_df = _local_movie_df()
    if local_df is None:
        return None

    row = None
    id_field = "id" if "id" in local_df.columns else None
    if id_field:
        try:
            match = local_df[local_df[id_field].astype(str).str.strip().str.lower().eq(str(movie_id).strip().lower())]
            if not match.empty:
                row = match.iloc[0]
        except Exception:
            row = None
    if row is None:
        try:
            row = local_df.iloc[int(movie_id)]
        except (IndexError, TypeError, ValueError):
            return None

    title = str(row.get("title") or "").strip()
    if not title:
        return None

    poster_path = _safe_media_path(row, "poster_path") or _safe_media_path(row, "poster_url")
    backdrop_path = _safe_media_path(row, "backdrop_path") or _safe_media_path(row, "backdrop_url") or poster_path
    actual_id = _row_to_movie_id(row, movie_id)

    return TMDBMovieDetails(
        tmdb_id=int(actual_id),
        title=title,
        overview=row.get("overview") or "",
        release_date=row.get("release_date") or "",
        poster_url=make_img_url(poster_path) or _fallback_poster_url(title),
        backdrop_url=make_img_url(backdrop_path) or _fallback_poster_url(title),
        genres=[],
    )


async def tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    q = dict(params)
    if not tmdb_is_configured():
        raise HTTPException(
            status_code=503,
            detail="TMDB API key is not configured. Add a valid TMDB_API_KEY to your .env file.",
        )

    q["api_key"] = TMDB_API_KEY

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(f"{TMDB_BASE}{path}", params=q)
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502,
            detail=f"TMDB request error: {type(e).__name__} | {repr(e)}",
        )

    if r.status_code != 200:
        raise HTTPException(
            status_code=502, detail=f"TMDB error {r.status_code}: {r.text}"
        )

    return r.json()


def _local_home_cards(limit: int = 24) -> List[TMDBMovieCard]:
    local_df = _local_movie_df()
    if local_df is None:
        return []

    subset = local_df.copy()
    if "popularity" in subset.columns:
        subset["popularity"] = pd.to_numeric(subset["popularity"], errors="coerce")
        subset = subset.sort_values("popularity", ascending=False)
    subset = subset.head(limit)

    cards: List[TMDBMovieCard] = []
    for idx, row in subset.iterrows():
        title = str(row.get("title") or "").strip()
        if not title:
            continue
        movie_id = _row_to_movie_id(row, int(idx))
        poster_path = _safe_media_path(row, "poster_path") or _safe_media_path(row, "poster_url")
        cards.append(
            TMDBMovieCard(
                tmdb_id=int(movie_id),
                title=title,
                poster_url=make_img_url(poster_path) or _fallback_poster_url(title),
                release_date=row.get("release_date") or "",
                vote_average=row.get("vote_average"),
            )
        )
    return cards


def tmdb_cards_from_results(
    results: List[dict], limit: int = 20
) -> List[TMDBMovieCard]:
    out: List[TMDBMovieCard] = []
    for m in (results or [])[:limit]:
        out.append(
            TMDBMovieCard(
                tmdb_id=int(m["id"]),
                title=m.get("title") or m.get("name") or "",
                poster_url=make_img_url(m.get("poster_path")),
                release_date=m.get("release_date"),
                vote_average=m.get("vote_average"),
            )
        )
    return out


async def tmdb_movie_details(movie_id: int) -> TMDBMovieDetails:
    if not tmdb_is_configured():
        details = _local_movie_details(movie_id)
        if details:
            return details
        raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found in local dataset")

    data = await tmdb_get(f"/movie/{movie_id}", {"language": "en-US"})
    return TMDBMovieDetails(
        tmdb_id=int(data["id"]),
        title=data.get("title") or "",
        overview=data.get("overview"),
        release_date=data.get("release_date"),
        poster_url=make_img_url(data.get("poster_path")),
        backdrop_url=make_img_url(data.get("backdrop_path")),
        genres=data.get("genres", []) or [],
    )


async def tmdb_search_movies(query: str, page: int = 1) -> Dict[str, Any]:
    """
    Raw TMDB response for keyword search (MULTIPLE results).
    Streamlit will use this for suggestions and grid.
    """
    if not tmdb_is_configured():
        return {"results": _local_search_results(query, limit=20)}

    return await tmdb_get(
        "/search/movie",
        {
            "query": query,
            "include_adult": "false",
            "language": "en-US",
            "page": page,
        },
    )


async def tmdb_search_first(query: str) -> Optional[dict]:
    if not tmdb_is_configured():
        results = _local_search_results(query, limit=1)
        return results[0] if results else None

    data = await tmdb_search_movies(query=query, page=1)
    results = data.get("results", [])
    return results[0] if results else None


# =========================
# TF-IDF Helpers
# =========================
def build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    """
    indices.pkl can be:
    - dict(title -> index)
    - pandas Series (index=title, value=index)
    We normalize into TITLE_TO_IDX.
    """
    title_to_idx: Dict[str, int] = {}

    if isinstance(indices, dict):
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx

    # pandas Series or similar mapping
    try:
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx
    except Exception:
        # last resort: if it's a list-like etc.
        raise RuntimeError(
            "indices.pkl must be dict or pandas Series-like (with .items())"
        )


def get_local_idx_by_title(title: str) -> int:
    global TITLE_TO_IDX
    if TITLE_TO_IDX is None:
        raise HTTPException(status_code=500, detail="TF-IDF index map not initialized")
    key = _norm_title(title)
    if key in TITLE_TO_IDX:
        return int(TITLE_TO_IDX[key])
    raise HTTPException(
        status_code=404, detail=f"Title not found in local dataset: '{title}'"
    )


def tfidf_recommend_titles(
    query_title: str, top_n: int = 10
) -> List[Tuple[str, float]]:
    """
    Returns list of (title, score) from local df using cosine similarity on TF-IDF matrix.
    Safe against missing columns/rows.
    """
    global df, tfidf_matrix
    if df is None or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="TF-IDF resources not loaded")

    idx = get_local_idx_by_title(query_title)

    # query vector
    qv = tfidf_matrix[idx]
    scores = (tfidf_matrix @ qv.T).toarray().ravel()

    # sort descending
    order = np.argsort(-scores)

    out: List[Tuple[str, float]] = []
    for i in order:
        if int(i) == int(idx):
            continue
        try:
            title_i = str(df.iloc[int(i)]["title"])
        except Exception:
            continue
        out.append((title_i, float(scores[int(i)])))
        if len(out) >= top_n:
            break
    return out


async def attach_tmdb_card_by_title(title: str) -> Optional[TMDBMovieCard]:
    """
    Uses TMDB search by title to fetch poster for a local title.
    If not found, returns None (never crashes the endpoint).
    """
    try:
        m = await tmdb_search_first(title)
        if not m:
            return None
        return TMDBMovieCard(
            tmdb_id=int(m["id"]),
            title=m.get("title") or title,
            poster_url=make_img_url(m.get("poster_path")),
            release_date=m.get("release_date"),
            vote_average=m.get("vote_average"),
        )
    except Exception:
        return None


# =========================
# STARTUP: LOAD PICKLES
# =========================
@app.on_event("startup")
def load_pickles():
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX

    # Load df
    with open(DF_PATH, "rb") as f:
        df = pickle.load(f)

    if CSV_PATH and os.path.exists(CSV_PATH):
        try:
            csv_df = pd.read_csv(CSV_PATH, low_memory=False)
            if not csv_df.empty and "title" in csv_df.columns:
                needed = [c for c in ["id", "title", "overview", "release_date", "vote_average", "poster_path", "backdrop_path"] if c in csv_df.columns]
                if needed:
                    csv_df = csv_df[needed].drop_duplicates(subset=["title"]).copy()
                    # Prefer CSV metadata as the fallback source because df.pkl is incomplete.
                    if "poster_path" in csv_df.columns and "title" in csv_df.columns:
                        df = csv_df
                    elif "title" in df.columns:
                        df = df.merge(csv_df, on="title", how="left")
        except Exception:
            pass

    # Load indices
    with open(INDICES_PATH, "rb") as f:
        indices_obj = pickle.load(f)

    # Load TF-IDF matrix (usually scipy sparse)
    with open(TFIDF_MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)

    # Load tfidf vectorizer (optional, not used directly here)
    with open(TFIDF_PATH, "rb") as f:
        tfidf_obj = pickle.load(f)

    # Build normalized map
    TITLE_TO_IDX = build_title_to_idx_map(indices_obj)

    # sanity
    if df is None or "title" not in df.columns:
        raise RuntimeError("df.pkl must contain a DataFrame with a 'title' column")


# =========================
# ROUTES
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}


# ---------- HOME FEED (TMDB) ----------
@app.get("/home", response_model=List[TMDBMovieCard], responses={
    400: {"description": "Invalid category"},
    500: {"description": "Home route failed"},
})
async def home(
    category: str = Query("popular"),
    limit: int = Query(24, ge=1, le=50),
):
    """
    Home feed for Streamlit (posters).
    category:
      - trending (trending/movie/day)
      - popular, top_rated, upcoming, now_playing  (movie/{category})
    """
    try:
        if not tmdb_is_configured():
            return _local_home_cards(limit=limit)

        if category == "trending":
            data = await tmdb_get("/trending/movie/day", {"language": "en-US"})
            return await tmdb_cards_from_results(data.get("results", []), limit=limit)

        if category not in {"popular", "top_rated", "upcoming", "now_playing"}:
            raise HTTPException(status_code=400, detail="Invalid category")

        data = await tmdb_get(f"/movie/{category}", {"language": "en-US", "page": 1})
        return await tmdb_cards_from_results(data.get("results", []), limit=limit)

    except HTTPException as exc:
        if not tmdb_is_configured() and exc.status_code in {401, 403, 404, 502, 503}:
            return _local_home_cards(limit=limit)
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Home route failed: {e}")


# ---------- TMDB KEYWORD SEARCH (MULTIPLE RESULTS) ----------
@app.get("/tmdb/search", responses={
    404: {"description": "No results found"},
})
async def tmdb_search(
    query: str = Query(..., min_length=1),
    page: int = Query(1, ge=1, le=10),
):
    """
    Returns RAW TMDB shape with 'results' list.
    Streamlit will use it for:
      - dropdown suggestions
      - grid results
    """
    try:
        return await tmdb_search_movies(query=query, page=page)
    except HTTPException as exc:
        if not tmdb_is_configured() and exc.status_code in {401, 403, 404, 502, 503}:
            return {"results": _local_search_results(query, limit=20)}
        raise


# ---------- MOVIE DETAILS (SAFE ROUTE) ----------
@app.get("/movie/id/{tmdb_id}", response_model=TMDBMovieDetails, responses={
    404: {"description": "Movie not found in local or remote dataset"},
})
async def movie_details_route(tmdb_id: int):
    try:
        return await tmdb_movie_details(tmdb_id)
    except HTTPException as exc:
        if not tmdb_is_configured() and exc.status_code in {401, 403, 404, 502, 503}:
            local = _local_movie_details(tmdb_id)
            if local:
                return local
        raise

# ---------- GENRE RECOMMENDATIONS ----------
@app.get("/recommend/genre", response_model=List[TMDBMovieCard], responses={
    400: {"description": "Invalid request"},
    404: {"description": "Movie not found"},
})
async def recommend_genre(
    tmdb_id: int = Query(...),
    limit: int = Query(18, ge=1, le=50),
):
    """
    Given a TMDB movie ID:
    - fetch details
    - pick first genre
    - discover movies in that genre (popular)
    """
    details = await tmdb_movie_details(tmdb_id)
    if not details.genres:
        return []

    genre_id = details.genres[0]["id"]
    discover = await tmdb_get(
        "/discover/movie",
        {
            "with_genres": genre_id,
            "language": "en-US",
            "sort_by": "popularity.desc",
            "page": 1,
        },
    )
    cards = await tmdb_cards_from_results(discover.get("results", []), limit=limit)
    return [c for c in cards if c.tmdb_id != tmdb_id]


# ---------- TF-IDF ONLY (debug/useful) ----------
@app.get("/recommend/tfidf", responses={
    404: {"description": "Title not found in local dataset"},
    500: {"description": "TF-IDF resources not loaded or no index"},
})
async def recommend_tfidf(
    title: str = Query(..., min_length=1),
    top_n: int = Query(10, ge=1, le=50),
):
    recs = tfidf_recommend_titles(title, top_n=top_n)
    return [{"title": t, "score": s} for t, s in recs]


# ---------- BUNDLE: Details + TF-IDF recs + Genre recs ----------
@app.get("/movie/search", response_model=SearchBundleResponse, responses={
    404: {"description": "No movie found for query"},
    500: {"description": "A recommendation step failed"},
})
async def search_bundle(
    query: str = Query(..., min_length=1),
    tfidf_top_n: int = Query(12, ge=1, le=30),
    genre_limit: int = Query(12, ge=1, le=30),
):
    """
    This endpoint is for when you have a selected movie and want:
      - movie details
      - TF-IDF recommendations (local) + posters
      - Genre recommendations (TMDB) + posters

    NOTE:
    - It selects the BEST match from TMDB for the given query.
    - If you want MULTIPLE matches, use /tmdb/search
    """
    best = await tmdb_search_first(query)
    if not best:
        raise HTTPException(
            status_code=404, detail=f"No movie found for query: {query}"
        )

    tmdb_id = int(best["id"])
    details = await tmdb_movie_details(tmdb_id)

    # 1) TF-IDF recommendations (never crash endpoint)
    tfidf_items: List[TFIDFRecItem] = []

    recs: List[Tuple[str, float]] = []
    try:
        # try local dataset by TMDB title
        recs = tfidf_recommend_titles(details.title, top_n=tfidf_top_n)
    except Exception:
        # fallback to user query
        try:
            recs = tfidf_recommend_titles(query, top_n=tfidf_top_n)
        except Exception:
            recs = []

    for title, score in recs:
        card = await attach_tmdb_card_by_title(title)
        tfidf_items.append(TFIDFRecItem(title=title, score=score, tmdb=card))

    # 2) Genre recommendations (TMDB discover by first genre)
    genre_recs: List[TMDBMovieCard] = []
    if details.genres:
        genre_id = details.genres[0]["id"]
        discover = await tmdb_get(
            "/discover/movie",
            {
                "with_genres": genre_id,
                "language": "en-US",
                "sort_by": "popularity.desc",
                "page": 1,
            },
        )
        cards = await tmdb_cards_from_results(
            discover.get("results", []), limit=genre_limit
        )
        genre_recs = [c for c in cards if c.tmdb_id != details.tmdb_id]

    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=genre_recs,
    )


def local_placeholder_poster(title: str) -> str:
    svg = f"""
    <svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 500 750'>
      <defs>
        <linearGradient id='g' x1='0' x2='1' y1='0' y2='1'>
          <stop offset='0%' stop-color='#0f172a'/>
          <stop offset='100%' stop-color='#334155'/>
        </linearGradient>
      </defs>
      <rect width='500' height='750' fill='url(#g)'/>
      <rect x='20' y='20' width='460' height='710' rx='16' fill='none' stroke='#cbd5e1' stroke-width='4'/>
      <text x='250' y='330' text-anchor='middle' font-size='42' font-family='Arial, sans-serif' fill='#e2e8f0' font-weight='bold'>MOVIE</text>
      <text x='250' y='390' text-anchor='middle' font-size='42' font-family='Arial, sans-serif' fill='#e2e8f0' font-weight='bold'>POSTER</text>
      <text x='250' y='660' text-anchor='middle' font-size='22' font-family='Arial, sans-serif' fill='#cbd5e1'>{escape(title[:32])}</text>
    </svg>
    """
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def _fallback_poster_url(title: Optional[str] = None) -> Optional[str]:
    return local_placeholder_poster(title or "Movie")