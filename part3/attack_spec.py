"""Part 3: a bounded attack proposal. The injector applies it to a copy.

Two families (build plan S5):
    freeze   repeat the first matching payload for a CAN ID over an
             interval -- stops a value that would normally change.
    offset   add a bounded signed delta to one payload byte of a CAN ID
             over an interval, clamped back into 0-255 -- spoofs a value
             pushed toward an extreme (ROAD's max_speedometer_attack is
             this shape). byte_index/offset only apply to this family;
             see defender/stage2.py's per-byte "fields" for why a single
             payload byte is the natural unit here.
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

AttackFamily = Literal["freeze", "offset"]


class AttackSpec(BaseModel):
    """A proposal. The injector applies it to a copy of a window."""

    model_config = ConfigDict(extra="forbid")

    attack_id: str = Field(min_length=1)
    family: AttackFamily
    target_can_id: str = Field(min_length=1)
    start_offset_ms: float = Field(ge=0)
    duration_ms: float = Field(gt=0)
    reasoning: str = ""
    # "offset" only: which payload byte to perturb (0 = first byte) and the
    # signed delta added to it before clamping back into 0-255.
    byte_index: Optional[int] = Field(default=None, ge=0, le=7)
    offset: Optional[int] = Field(default=None, ge=-128, le=127)

    @model_validator(mode="after")
    def check_family_fields(self):
        is_offset = self.family == "offset"
        has_offset_fields = self.byte_index is not None or self.offset is not None
        if is_offset and (self.byte_index is None or self.offset is None):
            raise ValueError("family 'offset' needs both byte_index and offset")
        if not is_offset and has_offset_fields:
            raise ValueError(f"family {self.family!r} must not set byte_index or offset")
        if is_offset and self.offset == 0:
            raise ValueError("offset must not be 0; that would be a no-op attack")
        return self
