"""Tools that really execute: a safe calculator and a per-session scratchpad.

Agents call a tool with one line:  `CALL calculator: <expression>`  or
`CALL scratchpad: write <text>` / `CALL scratchpad: read`.
"""

import ast
import operator
import re
from dataclasses import dataclass, field

_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
           ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_CALL_RE = re.compile(r"^\s*\**\s*CALL\s+([A-Za-z_]+)\s*:?\s*(.*)$", re.IGNORECASE | re.MULTILINE)


class ToolError(ValueError):
    pass


def calculate(expression: str) -> str:
    """Evaluate arithmetic only (numbers, + - * / // % **, parentheses). No names, no calls."""
    expr = expression.strip().strip("`").replace("×", "*").replace("÷", "/").replace("^", "**")
    expr = re.sub(r"(?<=\d),(?=\d{3}\b)", "", expr).replace("$", "")
    if not expr:
        raise ToolError("empty expression")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ToolError(f"not an arithmetic expression: {expression!r}") from e

    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 12:
                raise ToolError("exponent too large")
            return _BINOPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](ev(node.operand))
        raise ToolError(f"unsupported element in {expression!r}")

    try:
        value = ev(tree)
    except ZeroDivisionError as e:
        raise ToolError("division by zero") from e
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, float):
        value = round(value, 6)
    return str(value)


@dataclass
class Scratchpad:
    notes: list[str] = field(default_factory=list)

    def run(self, arg: str) -> str:
        arg = arg.strip()
        low = arg.lower()
        if low.startswith("write"):
            text = arg[5:].strip(" :")
            if not text:
                raise ToolError("nothing to write")
            self.notes.append(text[:300])
            return f"saved note {len(self.notes)}"
        if low.startswith("read"):
            return " | ".join(f"{i + 1}. {n}" for i, n in enumerate(self.notes)) or "(empty)"
        raise ToolError("scratchpad usage: 'write <text>' or 'read'")


@dataclass(frozen=True)
class ToolCall:
    tool: str
    argument: str


def parse_tool_call(text: str) -> ToolCall | None:
    """First `CALL <tool>: <arg>` line in an agent message, or None."""
    m = _CALL_RE.search(text)
    if not m:
        return None
    return ToolCall(m.group(1).lower(), m.group(2).strip())


def execute(call: ToolCall, allowed: tuple[str, ...], scratchpad: Scratchpad) -> tuple[str, bool]:
    """Run a tool call. Returns (result text, ok). Errors are returned to the agent, not raised."""
    if call.tool not in allowed:
        return f"error: tool '{call.tool}' is not available to you (you have: "\
               f"{', '.join(allowed) or 'none'})", False
    try:
        if call.tool == "calculator":
            return calculate(call.argument), True
        if call.tool == "scratchpad":
            return scratchpad.run(call.argument), True
    except ToolError as e:
        return f"error: {e}", False
    return f"error: unknown tool '{call.tool}'", False
