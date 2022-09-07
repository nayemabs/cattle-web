from typing import List, Optional

from pydantic import BaseModel




class Cattle(BaseModel):
    weight: Optional[float] = None
    status: Optional[str] = None
    # cattle: List[str] = []

