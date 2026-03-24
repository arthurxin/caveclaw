from __future__ import annotations

import contextlib
import io
from copy import deepcopy
from typing import Any, Dict, Optional

from .types import PythonProgramBackend, PythonProgramExecutionRequest, PythonProgramExecutionResult


class IPythonProgramRuntime:
    def __init__(
        self,
        initial_namespace: Optional[Dict[str, Any]] = None,
        *,
        persist_state: bool = True,
    ):
        self.persist_state = persist_state
        self._initial_namespace = dict(initial_namespace or {})
        self._session_namespace = dict(initial_namespace or {})

    def snapshot_namespace(self) -> Dict[str, Any]:
        return dict(self._session_namespace)

    def update_namespace(self, values: Dict[str, Any]) -> None:
        self._session_namespace.update(values)

    def reset(self, namespace: Optional[Dict[str, Any]] = None) -> None:
        base = self._initial_namespace if namespace is None else namespace
        self._session_namespace = dict(base)

    def execute(self, request: PythonProgramExecutionRequest) -> PythonProgramExecutionResult:
        try:
            from IPython.core.interactiveshell import InteractiveShell
            from IPython.utils.capture import capture_output
            from traitlets.config import Config
        except ImportError as exc:
            return PythonProgramExecutionResult(
                success=False,
                error=f"ImportError: {exc}. Install IPython to enable the ipython backend.",
                backend="ipython",
                namespace=self.snapshot_namespace(),
                metadata=deepcopy(request.metadata),
            )

        config = Config()
        config.InteractiveShell.cache_size = 0
        config.InteractiveShell.history_length = 0
        config.InteractiveShell.automagic = False
        config.InteractiveShell.colors = "nocolor"
        config.InteractiveShell.autoindent = False
        shell = InteractiveShell.instance(config=config)

        previous_namespace = dict(shell.user_ns)
        effective_namespace = self.snapshot_namespace()
        effective_namespace.update(request.namespace)
        shell.user_ns.clear()
        shell.user_ns.update(effective_namespace)

        try:
            with capture_output() as captured:
                transformed = shell.transform_cell(request.block.code)
                result = shell.run_cell(transformed)

            final_namespace = dict(shell.user_ns)
            if self.persist_state:
                self._session_namespace = dict(final_namespace)

            success = not bool(getattr(result, "error_before_exec", None) or getattr(result, "error_in_exec", None))
            error_obj = getattr(result, "error_before_exec", None) or getattr(result, "error_in_exec", None)
            error = None if error_obj is None else f"{type(error_obj).__name__}: {error_obj}"

            return PythonProgramExecutionResult(
                success=success,
                stdout=captured.stdout,
                stderr=captured.stderr,
                error=error,
                backend="ipython",
                namespace=final_namespace,
                metadata=deepcopy(request.metadata),
            )
        finally:
            shell.user_ns.clear()
            shell.user_ns.update(previous_namespace)


class PythonProgramExecutor:
    def __init__(self, *, ipython_runtime: Optional[IPythonProgramRuntime] = None):
        self.ipython_runtime = ipython_runtime or IPythonProgramRuntime()

    def execute(self, request: PythonProgramExecutionRequest) -> PythonProgramExecutionResult:
        if request.backend == "ipython":
            return self._execute_ipython(request)
        return self._execute_python(request)

    def _execute_python(self, request: PythonProgramExecutionRequest) -> PythonProgramExecutionResult:
        namespace = dict(request.namespace)
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                exec(request.block.code, namespace, namespace)
            return PythonProgramExecutionResult(
                success=True,
                stdout=stdout_buffer.getvalue(),
                stderr=stderr_buffer.getvalue(),
                backend="python",
                namespace=namespace,
                metadata=deepcopy(request.metadata),
            )
        except Exception as exc:  # pragma: no cover - exercised in tests via message only
            return PythonProgramExecutionResult(
                success=False,
                stdout=stdout_buffer.getvalue(),
                stderr=stderr_buffer.getvalue(),
                error=f"{type(exc).__name__}: {exc}",
                backend="python",
                namespace=namespace,
                metadata=deepcopy(request.metadata),
            )

    def _execute_ipython(self, request: PythonProgramExecutionRequest) -> PythonProgramExecutionResult:
        return self.ipython_runtime.execute(request)


__all__ = ["IPythonProgramRuntime", "PythonProgramExecutor"]
