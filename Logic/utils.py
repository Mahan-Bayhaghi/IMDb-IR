from typing import Dict, List
from Logic.core.search import SearchEngine
from Logic.core.spell_correction import SpellCorrection
from Logic.core.snippet import Snippet
from Logic.core.indexer.indexes_enum import Indexes, Index_types
from Logic.core.indexer.index_reader import Index_reader
import json

movies_dataset = None  # TODO
search_engine = SearchEngine()


def import_dataset():
    with open("D:/Sharif/Daneshgah stuff/term 6/mir/project/IMDb-IR/Logic/IMDB_crawled.json", 'r') as file:
        data = json.load(file)
    return data


def correct_text(text: str, all_documents: List) -> str:
    """
    Correct the give query text, if it is misspelled using Jacard similarity

    Paramters
    ---------
    text: str
        The query text
    all_documents : list of json objects
        The input documents.

    Returns
    str
        The corrected form of the given text
    """
    # TODO: You can add any preprocessing steps here, if needed!

    spell_correction_obj = SpellCorrection(all_documents[:])
    text = spell_correction_obj.spell_check(text)
    return text


def search(
        query: str,
        max_result_count: int,
        method: str = "ltn-lnn",
        weights_list=None,
        should_print=False,
        preferred_genre: str = None,
):
    """
    Finds relevant documents to query

    Parameters
    ---------------------------------------------------------------------------------------------------
    max_result_count: Return top 'max_result_count' docs which have the highest scores.
                      notice that if max_result_count = -1, then you have to return all docs

    mode: 'detailed' for searching in title and text separately.
          'overall' for all words, and weighted by where the word appears on.

    where: when mode ='detailed', when we want search query
            in title or text not both of them at the same time.

    method: 'ltn.lnn' or 'ltc.lnc' or 'OkapiBM25'

    preferred_genre: A list containing preference rates for each genre. If None, the preference rates are equal.

    Returns
    ----------------------------------------------------------------------------------------------------
    list
    Retrieved documents with snippet
    """
    if weights_list is None:
        weights_list = [0.3, 0.3, 0.4]
    weights = {
        Indexes.STARS: weights_list[0],
        Indexes.GENRES: weights_list[1],
        Indexes.SUMMARIES: weights_list[2]
    }

    return search_engine.search(
        query, method, weights, max_results=max_result_count, safe_ranking=True
    )


def get_movie_by_id(id: str, movies_dataset: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Get movie by its id

    Parameters
    ---------------------------------------------------------------------------------------------------
    id: str
        The id of the movie

    movies_dataset: List[Dict[str, str]]
        The dataset of movies

    Returns
    ----------------------------------------------------------------------------------------------------
    dict
        The movie with the given id
    """
    result = {}
    for movie in movies_dataset:
        if movie["id"] == id:
            result = movie
            break

    result["Image_URL"] = (
        "https://m.media-amazon.com/images/M/MV5BNDE3ODcxYzMtY2YzZC00NmNlLWJiNDMtZDViZWM2MzIxZDYwXkEyXkFqcGdeQXVyNjAwNDUxODI@._V1_.jpg"
        # a default picture for selected movies
    )
    result["URL"] = (
        f"https://www.imdb.com/title/{result['id']}"  # The url pattern of IMDb movies
    )
    return result
