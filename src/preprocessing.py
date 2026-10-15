# Модуль предобработки библиометрических данных.

import json
import os
import re
from collections import defaultdict
from typing import List, Dict

import pandas as pd
from tqdm import tqdm


def restore_abstract(inverted_index: dict) -> str:
    # Восстанавливает текст аннотации из inverted index OpenAlex
    if not inverted_index:
        return ""
    positions = []
    for word, idxs in inverted_index.items():
        for idx in idxs:
            positions.append((idx, word))
    positions.sort()
    return " ".join(word for _, word in positions)


def clean_text(text: str) -> str:
    #Убирает HTML-теги, лишние пробелы и служебные символы 
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_raw(path: str) -> List[dict]:
    # Загружает сырой JSON, собранный модулем data_collection
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_papers(papers: List[dict]) -> pd.DataFrame:
    # Преобразует список статей в DataFrame с нужными полями.
    rows = []
    for p in tqdm(papers, desc="Нормализация статей"):
        abstract = restore_abstract(p.get("abstract"))
        rows.append({
            "paper_id": p.get("id"),
            "doi": p.get("doi"),
            "title": clean_text(p.get("title")),
            "abstract": clean_text(abstract),
            "year": p.get("year"),
            "cited_by_count": p.get("cited_by_count", 0),
            "venue": p.get("venue"),
            "venue_id": p.get("venue_id"),
            "authors": p.get("authors", []),
            "keywords": [clean_text(k).lower() for k in p.get("keywords", []) if k],
            "concepts": [c.lower() for c in p.get("concepts", []) if c],
            "references": p.get("references", []),
        })
    return pd.DataFrame(rows)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    # Удаляет дубликаты статей по DOI и по названию+году.
    before = len(df)
    df = df.drop_duplicates(subset=["paper_id"])
    df = df.drop_duplicates(subset=["doi"], keep="first")
    df["title_norm"] = df["title"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True)
    df = df.drop_duplicates(subset=["title_norm", "year"], keep="first")
    df = df.drop(columns=["title_norm"])
    print(f"Дедупликация: {before} → {len(df)} статей")
    return df.reset_index(drop=True)


def lemmatize_keywords(df: pd.DataFrame) -> pd.DataFrame:
    # Лемматизирует ключевые слова через spaCy.
    # Требует: python -m spacy download en_core_web_sm

    try:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except (ImportError, OSError):
        print("spaCy не установлен — пропускаем лемматизацию")
        return df

    def lemmatize_list(words):
        if not words:
            return []
        doc = nlp(" ".join(words))
        return [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]

    df["keywords"] = df["keywords"].apply(lemmatize_list)
    return df


def resolve_authors(df: pd.DataFrame) -> pd.DataFrame:
    # Приводит авторов к единому формату с ключом по ORCID или имени.
    author_records = defaultdict(lambda: {"name": None, "papers": [], "orcid": None})

    for _, row in df.iterrows():
        for a in row["authors"]:
            key = a.get("orcid") or a.get("id") or a.get("name")
            if not key:
                continue
            author_records[key]["name"] = a.get("name")
            author_records[key]["orcid"] = a.get("orcid")
            author_records[key]["papers"].append(row["paper_id"])

    authors_df = pd.DataFrame([
        {"author_key": k, "name": v["name"], "orcid": v["orcid"],
         "n_papers": len(v["papers"]), "paper_ids": v["papers"]}
        for k, v in author_records.items()
    ])

    print(f"Разрешено уникальных авторов: {len(authors_df)}")
    return authors_df


def build_citation_series(df: pd.DataFrame, current_year: int) -> pd.DataFrame:
    # Строит для каждой статьи накопленное число цитирований по годам.
    # Оценка: линейное распределение общего числа цитирований от года публикации.
    def series_for_row(row):
        if pd.isna(row["year"]):
            return {}
        years_since = max(current_year - int(row["year"]), 0)
        total = row["cited_by_count"]
        # Простая модель: к текущему году накоплено total,
        # распределяем равномерно по годам жизни статьи.
        series = {}
        for offset in range(1, years_since + 1):
            year = int(row["year"]) + offset
            series[year] = int(total * offset / max(years_since, 1))
        return series

    df["citation_series"] = df.apply(series_for_row, axis=1)
    return df


def build_time_slices(df: pd.DataFrame, years: List[int]) -> Dict[int, pd.DataFrame]:
    # Разбивает данные на срезы: для каждого года — только статьи,
    # опубликованные до этого года.
    slices = {}
    for year in years:
        slices[year] = df[df["year"] < year].copy()
        print(f"Срез {year}: {len(slices[year])} статей")
    return slices


def run_preprocessing(raw_path: str, output_dir: str, current_year: int = 2024):
    # Полный цикл предобработки: загрузка → нормализация → очистка → сохранение.
    os.makedirs(output_dir, exist_ok=True)

    raw = load_raw(raw_path)
    print(f"Загружено {len(raw)} сырых записей")

    df = normalize_papers(raw)
    df = deduplicate(df)
    df = lemmatize_keywords(df)
    df = build_citation_series(df, current_year)

    # Сохранение основного датасета
    papers_path = os.path.join(output_dir, "papers_clean.json")
    df.drop(columns=["citation_series"]).to_json(
        papers_path, orient="records", force_ascii=False, indent=2
    )

    # Сохранение временных рядов отдельно
    series_df = df[["paper_id", "year", "citation_series"]].copy()
    series_path = os.path.join(output_dir, "citation_series.json")
    series_df.to_json(series_path, orient="records", force_ascii=False, indent=2)

    # Сохранение авторов
    authors_df = resolve_authors(df)
    authors_path = os.path.join(output_dir, "authors.json")
    authors_df.to_json(authors_path, orient="records", force_ascii=False, indent=2)

    print(f"Сохранено: {papers_path}, {series_path}, {authors_path}")
    return df, authors_df


if __name__ == "__main__":
    run_preprocessing(
        raw_path="data/raw/papers_ml_2015_2020.json",
        output_dir="data/processed",
        current_year=2024,
    )