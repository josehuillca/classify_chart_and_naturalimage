from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict


class Bbox(TypedDict):
    x:int
    y:int
    w:int
    h:int
    conf:float


class Paper(TypedDict):
    prism_doi: str
    pii: str
    openaccess: bool
    xml: str

class Figure(TypedDict):
    refid: str
    caption: str
    figure_type: str
    chart_type: str
    contain_subfigure: bool
    bbox_subfigure: List[Bbox]

class Image(TypedDict):
    ref:str
    category:str
    type:str
    mimetype:str
    width:int
    height:int
    url:str
    file:str

class Paragraph(TypedDict):
    id:str
    text:str
    #text-processed