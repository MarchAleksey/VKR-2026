# Модуль сбора библиометрических данных через OpenAlex API.

import json
import time
import os
from typing import Iterator
from dotenv import load_dotenv

import requests
from tqdm import tqdm

load_dotenv()

BASE_URL = "https://api.openalex.org/works"

EMAIL = os.getenv("OPENALEX_EMAIL", "")

REQUEST_DELAY = 0.15

PER_PAGE = 200


def build_params(query: str, year_from: int, year_to: int, cursor: str = "*") -> dict:
    """Формирует параметры запроса к OpenAlex."""
    return {
        "search": query,
        "filter": (
            f"from_publication_date:{year_from}-01-01,"
            f"to_publication_date:{year_to}-12-31,"
            f"type:article"
        ),
        "per-page": PER_PAGE,
        "cursor": cursor,
        "mailto": EMAIL,
    }


def fetch_page(params: dict, max_retries: int = 5) -> dict:
    for attempt in range(max_retries):
        try:
            response = requests.get(BASE_URL, params=params, timeout=30)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                wait = 5 * (attempt + 1)
                print(f"[429] Rate limit. Ждём {wait} сек...")
                time.sleep(wait)
            else:
                print(f"[{response.status_code}] Ошибка запроса. Попытка {attempt + 1}/{max_retries}")
                time.sleep(2)
        except requests.RequestException as e:
            print(f"Сетевая ошибка: {e}. Попытка {attempt + 1}/{max_retries}")
            time.sleep(2)

    raise RuntimeError("Не удалось получить данные после всех попыток")


def iter_papers(query: str, year_from: int, year_to: int, max_papers: int = None) -> Iterator[dict]:
    cursor = "*"
    collected = 0

    while True:
        params = build_params(query, year_from, year_to, cursor)
        data = fetch_page(params)

        results = data.get("results", [])
        if not results:
            break

        for paper in results:
            yield paper
            collected += 1
            if max_papers and collected >= max_papers:
                return

        # Переход к следующей странице
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break

        time.sleep(REQUEST_DELAY)


def simplify_paper(paper: dict) -> dict:
    return {
        "id": paper.get("id"),
        "doi": paper.get("doi"),
        "title": paper.get("title"),
        "abstract": paper.get("abstract_inverted_index"),
        "year": paper.get("publication_year"),
        "cited_by_count": paper.get("cited_by_count"),
        "venue": (paper.get("primary_location") or {}).get("source", {}).get("display_name"),
        "venue_id": (paper.get("primary_location") or {}).get("source", {}).get("id"),
        "authors": [
            {
                "id": a.get("author", {}).get("id"),
                "name": a.get("author", {}).get("display_name"),
                "orcid": a.get("author", {}).get("orcid"),
            }
            for a in paper.get("authorships", [])
        ],
        "keywords": [
            k.get("display_name") for k in paper.get("keywords", [])
        ],
        "concepts": [
            c.get("display_name") for c in paper.get("concepts", [])
        ],
        "references": paper.get("referenced_works", []),
    }


def collect_and_save(
    query: str,
    year_from: int,
    year_to: int,
    output_path: str,
    max_papers: int = None,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    papers = []
    print(f"Сбор статей: '{query}' ({year_from}–{year_to})")

    for paper in tqdm(iter_papers(query, year_from, year_to, max_papers)):
        papers.append(simplify_paper(paper))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)

    print(f"Сохранено {len(papers)} статей в {output_path}")


if __name__ == "__main__":
    collect_and_save(
        query="machine learning",
        year_from=2015,
        year_to=2020,
        output_path="data/raw/papers_ml_2015_2020.json",
        max_papers=1000,  # колличество статей, можно потом убрать
    )