﻿import numpy as np
from copy import copy


def loubar_thres(arr, is_sorted=False):
    """
    :param arr: arr to compute loubar
    :param is_sorted: if the arr is not sorted, copy it and sorted it ascendingly first
    :return:
        loubar: the percentage of array above which is considered as hotspots
        arr_thres: the threshold to pass to be considered as hotspots, i.e., arr[arr>arr_thres] are the hotspots
    """
    if not is_sorted:
        arr = copy(arr)
        arr = np.array(sorted(arr))

    lonrenz_y = arr.cumsum() / arr.sum()
    lonrenz_y = np.insert(lonrenz_y, 0, 0)
    x_axis = np.arange(lonrenz_y.size)/(lonrenz_y.size-1)
    slope = (lonrenz_y[-1] - lonrenz_y[-2]) / (x_axis[-1] - x_axis[-2])
    loubar = (slope - 1) / slope
    arr_thres = arr[int(np.ceil((lonrenz_y.size - 1) * loubar) - 1)]

    return loubar, arr_thres
    

# https://gist.github.com/xijo/d4bad3953f7b9979dd91
_REPLACEMENTS = {
    "â‚¬": "€", "â€š": "‚", "â€ž": "„", "â€¦": "…", "Ë†" : "ˆ",
    "â€¹": "‹", "â€˜": "‘", "â€™": "’", "â€œ": "“", "â€" : "”",
    "â€¢": "•", "â€“": "–", "â€”": "—", "Ëœ" : "˜", "â„¢": "™",
    "â€º": "›", "Å“" : "œ", "Å’" : "Œ", "Å¾" : "ž", "Å¸" : "Ÿ",
    "Å¡" : "š", "Å½" : "Ž", "Â¡" : "¡", "Â¢" : "¢", "Â£" : "£",
    "Â¤" : "¤", "Â¥" : "¥", "Â¦" : "¦", "Â§" : "§", "Â¨" : "¨",
    "Â©" : "©", "Âª" : "ª", "Â«" : "«", "Â¬" : "¬", "Â®" : "®",
    "Â¯" : "¯", "Â°" : "°", "Â±" : "±", "Â²" : "²", "Â³" : "³",
    "Â´" : "´", "Âµ" : "µ", "Â¶" : "¶", "Â·" : "·", "Â¸" : "¸",
    "Â¹" : "¹", "Âº" : "º", "Â»" : "»", "Â¼" : "¼", "Â½" : "½",
    "Â¾" : "¾", "Â¿" : "¿", "Ã€" : "À", "Ã‚" : "Â", "Ãƒ" : "Ã",
    "Ã„" : "Ä", "Ã…" : "Å", "Ã†" : "Æ", "Ã‡" : "Ç", "Ãˆ" : "È",
    "Ã‰" : "É", "ÃŠ" : "Ê", "Ã‹" : "Ë", "ÃŒ" : "Ì", "ÃŽ" : "Î",
    "Ã‘" : "Ñ", "Ã’" : "Ò", "Ã“" : "Ó", "Ã”" : "Ô", "Ã•" : "Õ",
    "Ã–" : "Ö", "Ã—" : "×", "Ã˜" : "Ø", "Ã™" : "Ù", "Ãš" : "Ú",
    "Ã›" : "Û", "Ãœ" : "Ü", "Ãž" : "Þ", "ÃŸ" : "ß", "Ã¡" : "á",
    "Ã¢" : "â", "Ã£" : "ã", "Ã¤" : "ä", "Ã¥" : "å", "Ã¦" : "æ",
    "Ã§" : "ç", "Ã¨" : "è", "Ã©" : "é", "Ãª" : "ê", "Ã«" : "ë",
    "Ã¬" : "ì", "Ã­"  : "í", "Ã®" : "î", "Ã¯" : "ï", "Ã°" : "ð",
    "Ã±" : "ñ", "Ã²" : "ò", "Ã³" : "ó", "Ã´" : "ô", "Ãµ" : "õ",
    "Ã¶" : "ö", "Ã·" : "÷", "Ã¸" : "ø", "Ã¹" : "ù", "Ãº" : "ú",
    "Ã»" : "û", "Ã¼" : "ü", "Ã½" : "ý", "Ã¾" : "þ", "Ã¿" : "ÿ"
  }
  

def fix_spanish_encoding(s):
    for w, r in _REPLACEMENTS.items():
        s = s.replace(w, r)
    return s