import asyncio
import unittest
from importlib.util import find_spec

from agent_core.core.python_program_execution import (
    DefaultPythonRuntimeBridge,
    IPythonProgramRuntime,
    PythonProgramBlock,
    PythonProgramExecutionRequest,
    PythonProgramExecutionResult,
    PythonProgramExecutor,
    extract_first_python_program_block,
    extract_python_program_blocks,
)
from agent_core.assistant_messages import AgentContext, RuntimeState


class PythonProgramExecutionTests(unittest.TestCase):
    def test_extracts_multiple_python_fenced_blocks(self):
        text = """
before
```python
print("one")
```
middle
```python
print("two")
```
"""
        blocks = extract_python_program_blocks(text)
        self.assertEqual(len(blocks), 2)
        self.assertIn('print("one")', blocks[0].code)
        self.assertIn('print("two")', blocks[1].code)

    def test_extract_first_python_program_block_returns_none_when_missing(self):
        self.assertIsNone(extract_first_python_program_block("no fenced code here"))

    def test_extracts_triple_quote_python_blocks(self):
        text = '''
before
"""python
print("triple")
"""
after
'''
        block = extract_first_python_program_block(text)
        self.assertIsNotNone(block)
        self.assertIn('print("triple")', block.code)

    def test_python_backend_executes_code_with_namespace(self):
        executor = PythonProgramExecutor()
        request = PythonProgramExecutionRequest(
            block=PythonProgramBlock(code='print(greeting)\nvalue = 3'),
            backend="python",
            namespace={"greeting": "hello"},
        )

        result = executor.execute(request)

        self.assertTrue(result.success)
        self.assertEqual(result.stdout.strip(), "hello")
        self.assertEqual(result.namespace["value"], 3)

    def test_python_backend_captures_errors(self):
        executor = PythonProgramExecutor()
        request = PythonProgramExecutionRequest(
            block=PythonProgramBlock(code='raise ValueError("bad")'),
            backend="python",
        )

        result = executor.execute(request)

        self.assertFalse(result.success)
        self.assertIn("ValueError: bad", result.error)

    def test_ipython_runtime_import_error_is_reported(self):
        runtime = IPythonProgramRuntime(initial_namespace={"greeting": "hello"})
        executor = PythonProgramExecutor(ipython_runtime=runtime)

        result = executor.execute(
            PythonProgramExecutionRequest(
                block=PythonProgramBlock(code='print(greeting)'),
                backend="ipython",
            )
        )

        if find_spec("IPython") is None:
            self.assertFalse(result.success)
            self.assertIn("Install IPython", result.error)
        else:
            self.assertTrue(result.success)
            self.assertEqual(result.stdout.strip(), "hello")

    def test_default_runtime_bridge_syncs_supported_namespace_values(self):
        bridge = DefaultPythonRuntimeBridge()
        request = PythonProgramExecutionRequest(
            block=PythonProgramBlock(code="value = 3\nsummary = {'total': 3}"),
            backend="python",
            namespace={"messages": [], "runtime": RuntimeState()},
        )
        result = PythonProgramExecutionResult(
            success=True,
            backend="python",
            namespace={
                "messages": [],
                "runtime": RuntimeState(),
                "value": 3,
                "summary": {"total": 3},
                "helper": lambda: None,
            },
        )
        agent_context = AgentContext(messages=[])

        bridge_result = asyncio.run(
            bridge.build_runtime_bridge_result(
                execution_request=request,
                execution_result=result,
                agent_context=agent_context,
                python_block=request.block,
            )
        )

        self.assertEqual(bridge_result.synced_variables, ["value", "summary"])
        self.assertEqual(len(bridge_result.runtime_ops), 2)
        self.assertEqual(bridge_result.skipped_variables["helper"], "runtime_internal")


if __name__ == "__main__":
    unittest.main()
