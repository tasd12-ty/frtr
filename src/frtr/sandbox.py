"""E2B code sandbox for safe formula verification.

Provides an isolated execution environment for verifying Excel formula
results by executing equivalent Python/pandas code.
"""

from __future__ import annotations

import json
from typing import Optional


class CodeSandbox:
    """E2B-based code execution sandbox.

    Used to verify formula answers by running equivalent Python code
    in an isolated environment. If E2B is not available, falls back
    to a local (unsafe) eval.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key
        self._sandbox = None

    def _get_sandbox(self):
        """Lazy-initialize E2B sandbox."""
        if self._sandbox is None:
            try:
                from e2b_code_interpreter import Sandbox

                self._sandbox = Sandbox(api_key=self._api_key)
            except Exception as e:
                print(f"Warning: E2B sandbox not available: {e}")
                return None
        return self._sandbox

    def execute_python(self, code: str) -> dict:
        """Execute Python code in the sandbox.

        Returns:
            {"success": bool, "output": str, "error": str | None}
        """
        sandbox = self._get_sandbox()
        if sandbox is None:
            return self._local_eval(code)

        try:
            execution = sandbox.run_code(code)
            output = ""
            if execution.results:
                output = "\n".join(str(r.text) for r in execution.results if r.text)
            error = None
            if execution.error:
                error = f"{execution.error.name}: {execution.error.value}"
            return {
                "success": error is None,
                "output": output,
                "error": error,
            }
        except Exception as e:
            return {"success": False, "output": "", "error": str(e)}

    def verify_formula(
        self,
        workbook_path: str,
        formula_or_ref: str,
        expected: Optional[str] = None,
    ) -> dict:
        """Verify an Excel formula/cell reference against a workbook.

        Loads the workbook in the sandbox and evaluates the reference.
        """
        code = f"""
import openpyxl
wb = openpyxl.load_workbook("{workbook_path}", data_only=True)

ref = "{formula_or_ref}"
# Parse sheet!cell reference
if "!" in ref:
    sheet_name, cell_ref = ref.split("!", 1)
    ws = wb[sheet_name]
else:
    ws = wb.active
    cell_ref = ref

try:
    val = ws[cell_ref].value
    print(f"VALUE: {{val}}")
except:
    print(f"FORMULA: {{ref}}")
"""
        return self.execute_python(code)

    def _local_eval(self, code: str) -> dict:
        """Fallback: execute locally (use with caution)."""
        try:
            import io
            import contextlib

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exec(code, {"__builtins__": __builtins__})
            return {"success": True, "output": output.getvalue(), "error": None}
        except Exception as e:
            return {"success": False, "output": "", "error": str(e)}

    def close(self) -> None:
        """Clean up the sandbox."""
        if self._sandbox is not None:
            try:
                self._sandbox.kill()
            except Exception:
                pass
            self._sandbox = None
