from pydantic import BaseModel
from typing import Optional

class ExtractionResult(BaseModel):
    from_: str = ""  # from_ prevents python keyword conflict
    documentNo: str = ""
    documentDate: str = ""
    summaryDescription: str = ""
    finalPayableAmount: float = 0.0

    class Config:
        populate_by_name = True
        alias_generator = lambda string: "from" if string == "from_" else string

class APIResponse(BaseModel):
    errCode: Optional[str] = None
    error: Optional[str] = None
    result: Optional[dict] = None