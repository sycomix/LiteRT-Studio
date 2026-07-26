class StudioError(Exception):
    """Base error for a user-correctable Studio operation."""


class ConfigurationError(StudioError):
    """Raised when a plan configuration is invalid."""


class InspectionError(StudioError):
    """Raised when source assets cannot be inspected safely."""
