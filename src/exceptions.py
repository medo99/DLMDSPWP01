"""Project-specific exception types."""


class ProjectError(Exception):
    """Base exception for expected project-level failures."""


class DataFileNotFoundError(ProjectError):
    """Raised when a required dataset file cannot be found."""


class DataLoadingError(ProjectError):
    """Raised when a dataset exists but cannot be parsed or read."""


class DataValidationError(ProjectError):
    """Raised when a dataset violates the expected assignment schema."""


class DatabaseOperationError(ProjectError):
    """Raised when SQLite persistence fails."""
