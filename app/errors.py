class PipelineError(Exception):
    """Base class for pipeline errors."""


class ModelCallError(PipelineError):
    """A provider call failed (network, HTTP error, timeout)."""


class BudgetExceeded(ModelCallError):
    """Not enough time left in the request budget to start a call."""


class ModelOutputError(PipelineError):
    """The model answered, but not with JSON matching the role's schema (after one repair)."""


class PIILeakError(PipelineError):
    """Outgoing text still contains a value the masker removed. The call is blocked."""


class StoreError(PipelineError):
    """Supabase call failed."""
