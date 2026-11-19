"""Exception hierarchy.

Every error the application raises deliberately derives from :class:`SalesAgentError`, so the CLI
can print a readable message for expected failures while genuine programming errors still surface
with a traceback.

Tool errors are split into two kinds that the orchestrator reports back to the model:

* :class:`ToolInputError` - the model sent invalid arguments (it can fix them and retry);
* :class:`ToolExecutionError` - valid arguments, but execution failed (rejected SQL, timeout,
  missing index, embedding API failure).
"""


class SalesAgentError(Exception):
    """Base class for all expected application errors."""


class ConfigurationError(SalesAgentError):
    """Invalid or missing configuration (environment variables, paths, API keys)."""


class ToolInputError(SalesAgentError, ValueError):
    """A tool received arguments that do not satisfy its contract."""


class ToolExecutionError(SalesAgentError):
    """A tool received valid arguments but could not complete."""


class SqlExecutionError(ToolExecutionError):
    """A SQL query was rejected, failed, or exceeded a resource limit."""


class KnowledgeIndexError(ToolExecutionError):
    """The vector index is missing, stale, or could not be queried."""


class EmbeddingError(ToolExecutionError):
    """The embedding provider failed to return vectors."""


class ModelError(SalesAgentError):
    """The chat model provider failed (network, auth, rate limit, timeout)."""
