
# Модуль построения гетерогенного графа научного ландшафта.

import json
import os
import math
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData



def load_processed(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Загружает три файла, созданных модулем предобработки
    papers = pd.read_json(os.path.join(data_dir, "papers_clean.json"))
    authors = pd.read_json(os.path.join(data_dir, "authors.json"))
    series = pd.read_json(os.path.join(data_dir, "citation_series.json"))
    return papers, authors, series



def build_index_maps(papers: pd.DataFrame, authors: pd.DataFrame):
    # Строит отображения: исходный ID → индекс узла в графе.
    # Узлы нумеруются с 0 внутри каждого типа.

    paper_ids = papers["paper_id"].tolist()
    paper_to_idx = {pid: i for i, pid in enumerate(paper_ids)}

    author_keys = authors["author_key"].tolist()
    author_to_idx = {aid: i for i, aid in enumerate(author_keys)}

    # Журналы: уникальные venue_id, встречающиеся в papers
    venue_ids = papers["venue_id"].dropna().unique().tolist()
    venue_to_idx = {vid: i for i, vid in enumerate(venue_ids)}

    # Ключевые слова: уникальные леммы, встречающиеся в papers
    keyword_set = set()
    for kws in papers["keywords"]:
        for kw in kws:
            keyword_set.add(kw)
    keyword_list = sorted(keyword_set)
    keyword_to_idx = {kw: i for i, kw in enumerate(keyword_list)}

    maps = {
        "paper": paper_to_idx,
        "author": author_to_idx,
        "venue": venue_to_idx,
        "keyword": keyword_to_idx,
    }
    print(f"Узлы: paper={len(paper_to_idx)}, author={len(author_to_idx)}, "
          f"venue={len(venue_to_idx)}, keyword={len(keyword_to_idx)}")
    return maps


def temporal_weight(year_edge: int, current_year: int, half_life: float = 5.0) -> float:
    # Экспоненциальное затухание: чем старше связь, тем меньше её вес.
    # half_life — период полураспада в годах.
    if year_edge is None or pd.isna(year_edge):
        return 1.0
    delta = max(current_year - int(year_edge), 0)
    return math.exp(-math.log(2) * delta / half_life)


def build_paper_paper_edges(papers: pd.DataFrame, maps: dict, current_year: int):
    # Рёбра 'цитирует': paper → paper, вес по году цитируемой статьи
    src, dst, weights = [], [], []
    p2i = maps["paper"]
    paper_year = dict(zip(papers["paper_id"], papers["year"]))

    for _, row in papers.iterrows():
        if row["paper_id"] not in p2i:
            continue
        src_idx = p2i[row["paper_id"]]
        for ref_id in row["references"]:
            if ref_id in p2i:
                dst_idx = p2i[ref_id]
                src.append(src_idx)
                dst.append(dst_idx)
                weights.append(temporal_weight(paper_year.get(ref_id), current_year))
    return src, dst, weights


def build_paper_author_edges(papers: pd.DataFrame, maps: dict, current_year: int):
    # Рёбра 'написал': author → paper
    src, dst, weights = [], [], []
    a2i, p2i = maps["author"], maps["paper"]

    # Обратный словарь: paper_id → author_key
    author_lookup = {}
    for _, row in papers.iterrows():
        for a in row["authors"]:
            key = a.get("orcid") or a.get("id") or a.get("name")
            if key:
                author_lookup.setdefault(row["paper_id"], []).append(key)

    for _, row in papers.iterrows():
        if row["paper_id"] not in p2i:
            continue
        p_idx = p2i[row["paper_id"]]
        year = row["year"]
        for key in author_lookup.get(row["paper_id"], []):
            if key in a2i:
                src.append(a2i[key])
                dst.append(p_idx)
                weights.append(temporal_weight(year, current_year))
    return src, dst, weights


def build_paper_venue_edges(papers: pd.DataFrame, maps: dict, current_year: int):
    # Рёбра 'опубликована в': paper → venue
    src, dst, weights = [], [], []
    p2i, v2i = maps["paper"], maps["venue"]
    for _, row in papers.iterrows():
        vid = row["venue_id"]
        if row["paper_id"] in p2i and vid in v2i:
            src.append(p2i[row["paper_id"]])
            dst.append(v2i[vid])
            weights.append(temporal_weight(row["year"], current_year))
    return src, dst, weights


def build_paper_keyword_edges(papers: pd.DataFrame, maps: dict, current_year: int):
    # Рёбра 'содержит термин': paper → keyword
    src, dst, weights = [], [], []
    p2i, k2i = maps["paper"], maps["keyword"]
    for _, row in papers.iterrows():
        if row["paper_id"] not in p2i:
            continue
        p_idx = p2i[row["paper_id"]]
        for kw in row["keywords"]:
            if kw in k2i:
                src.append(p_idx)
                dst.append(k2i[kw])
                weights.append(temporal_weight(row["year"], current_year))
    return src, dst, weights


def build_paper_features(papers: pd.DataFrame, maps: dict, emb_dim: int = 64) -> torch.Tensor:
    # Признаки статей: год (нормализованный), log(cited_by_count), случайный эмбеддинг.
    # Заменяется на реальные эмбеддинги (SciBERT) на следующем шаге пайплайна.
    n = len(maps["paper"])
    feats = torch.zeros((n, emb_dim + 2), dtype=torch.float)
    p2i = maps["paper"]

    cited = papers["cited_by_count"].fillna(0).values
    max_cited = max(cited.max(), 1)
    years = papers["year"].fillna(papers["year"].median()).values
    min_year, max_year = years.min(), max(years.max(), min_year + 1)

    rng = np.random.default_rng(42)
    for _, row in papers.iterrows():
        idx = p2i.get(row["paper_id"])
        if idx is None:
            continue
        year_norm = (row["year"] - min_year) / (max_year - min_year)
        cited_norm = math.log1p(row["cited_by_count"]) / math.log1p(max_cited)
        feats[idx, 0] = year_norm
        feats[idx, 1] = cited_norm
        feats[idx, 2:] = torch.tensor(rng.normal(0, 0.1, emb_dim), dtype=torch.float)
    return feats


def build_author_features(authors: pd.DataFrame, maps: dict, emb_dim: int = 64) -> torch.Tensor:
    # Признаки авторов: log(число статей) + случайный эмбеддинг
    n = len(maps["author"])
    feats = torch.zeros((n, emb_dim + 1), dtype=torch.float)
    a2i = maps["author"]
    rng = np.random.default_rng(42)

    max_papers = max(authors["n_papers"].max(), 1)
    for _, row in authors.iterrows():
        idx = a2i.get(row["author_key"])
        if idx is None:
            continue
        feats[idx, 0] = math.log1p(row["n_papers"]) / math.log1p(max_papers)
        feats[idx, 1:] = torch.tensor(rng.normal(0, 0.1, emb_dim), dtype=torch.float)
    return feats


def build_venue_features(papers: pd.DataFrame, maps: dict, emb_dim: int = 64) -> torch.Tensor:
    # Признаки журналов: средний log(cited_by_count) статей журнала + случайный эмбеддинг
    n = len(maps["venue"])
    feats = torch.zeros((n, emb_dim + 1), dtype=torch.float)
    v2i = maps["venue"]
    rng = np.random.default_rng(42)

    agg = papers.groupby("venue_id")["cited_by_count"].mean().to_dict()
    max_avg = max(agg.values()) if agg else 1

    for vid, avg in agg.items():
        idx = v2i.get(vid)
        if idx is None:
            continue
        feats[idx, 0] = math.log1p(avg) / math.log1p(max(max_avg, 1))
        feats[idx, 1:] = torch.tensor(rng.normal(0, 0.1, emb_dim), dtype=torch.float)
    return feats


def build_keyword_features(maps: dict, emb_dim: int = 64) -> torch.Tensor:
    # Признаки ключевых слов: случайный эмбеддинг (заменяется на SPECTER позже)
    n = len(maps["keyword"])
    rng = np.random.default_rng(42)
    return torch.tensor(rng.normal(0, 0.1, (n, emb_dim)), dtype=torch.float)


def build_targets(papers: pd.DataFrame, series: pd.DataFrame, maps: dict,
                  horizon: int = 2, current_year: int = 2024) -> torch.Tensor:
    # Целевая переменная: log1p накопленных цитирований на горизонте `horizon` лет
    # после публикации.
    n = len(maps["paper"])
    targets = torch.zeros(n, dtype=torch.float)
    p2i = maps["paper"]

    series_map = dict(zip(series["paper_id"], series["citation_series"]))
    paper_year = dict(zip(papers["paper_id"], papers["year"]))

    for pid, idx in p2i.items():
        year = paper_year.get(pid)
        if year is None or pd.isna(year):
            continue
        target_year = int(year) + horizon
        s = series_map.get(pid, {})
        if isinstance(s, dict) and target_year in s:
            targets[idx] = math.log1p(s[target_year])
    return targets


def build_hetero_graph(data_dir: str, output_path: str,
                       current_year: int = 2024, horizon: int = 2):
    # Полный пайплайн: загрузка → индексы → рёбра → признаки → HeteroData.
    papers, authors, series = load_processed(data_dir)
    maps = build_index_maps(papers, authors)

    graph = HeteroData()

    # Узлы 
    graph["paper"].x = build_paper_features(papers, maps)
    graph["author"].x = build_author_features(authors, maps)
    graph["venue"].x = build_venue_features(papers, maps)
    graph["keyword"].x = build_keyword_features(maps)

    # Рёбра
    src, dst, w = build_paper_paper_edges(papers, maps, current_year)
    graph["paper", "cites", "paper"].edge_index = torch.tensor([src, dst], dtype=torch.long)
    graph["paper", "cites", "paper"].edge_weight = torch.tensor(w, dtype=torch.float)

    src, dst, w = build_paper_author_edges(papers, maps, current_year)
    graph["author", "writes", "paper"].edge_index = torch.tensor([src, dst], dtype=torch.long)
    graph["author", "writes", "paper"].edge_weight = torch.tensor(w, dtype=torch.float)

    src, dst, w = build_paper_venue_edges(papers, maps, current_year)
    graph["paper", "published_in", "venue"].edge_index = torch.tensor([src, dst], dtype=torch.long)
    graph["paper", "published_in", "venue"].edge_weight = torch.tensor(w, dtype=torch.float)

    src, dst, w = build_paper_keyword_edges(papers, maps, current_year)
    graph["paper", "has_keyword", "keyword"].edge_index = torch.tensor([src, dst], dtype=torch.long)
    graph["paper", "has_keyword", "keyword"].edge_weight = torch.tensor(w, dtype=torch.float)

    # Целевая переменная
    graph["paper"].y = build_targets(papers, series, maps, horizon, current_year)

    # Сохранение
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(graph, output_path)
    print(f"Граф сохранён: {output_path}")
    print(graph)

    return graph


if __name__ == "__main__":
    build_hetero_graph(
        data_dir="data/processed",
        output_path="data/graphs/hetero_graph.pt",
        current_year=2024,
        horizon=2,
    )