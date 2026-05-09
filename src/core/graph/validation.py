from pydantic import BaseModel, field_validator, ValidationError

class ValidTriple(BaseModel):
    """
    A Pydantic model to represent a valid knowledge graph triple.

    This model enforces that the subject, predicate, and object are all non-empty,
    meaningful strings, preventing the storage of "garbage" data extracted by the LLM.
    """
    subject: str
    predicate: str
    object: str

    @field_validator('subject', 'predicate', 'object')
    def must_not_be_empty(cls, v: str) -> str:
        """
        Validates that a field is not None, empty, or just whitespace.
        """
        if not v or not v.strip():
            raise ValueError("Field cannot be empty or just whitespace")
        return v.strip() # Always return the stripped version