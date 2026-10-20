"""Frozen v1 protocol: exact rational arithmetic; never execute model output."""
import ast
import re
from fractions import Fraction

PROTOCOL = "math-answer-v1"
NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d+)?)(?:\s*/\s*[+-]?\d+)?")


def rational(text):
    text = text.strip().replace("−", "-")
    if len(text) > 80 or NUMBER.fullmatch(text) is None:
        raise ValueError("not a single finite rational number")
    parts = text.split("/")
    try:
        result = Fraction(parts[0].strip())
        if len(parts) == 2:
            result /= Fraction(parts[1].strip())
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("invalid rational number") from exc
    if abs(result.numerator) > 10**18 or result.denominator > 10**18:
        raise ValueError("number exceeds protocol bounds")
    return result


def parse_answer(output):
    # An unfinished thought is never a final answer. Completed thought is ignored.
    if "<think>" in output and "</think>" not in output:
        raise ValueError("unfinished thinking")
    output = output.rsplit("</think>", 1)[-1].strip()
    if output.count("答案：") + output.count("答案:") != 1:
        raise ValueError("expected exactly one answer marker")
    match = re.search(r"(?:^|\n)\s*答案[：:]\s*([^\n]+)\s*$", output)
    if match is None:
        raise ValueError("answer must be the final line")
    return rational(match.group(1))


def evaluate_expression(expression):
    """Only integer literals, + - * / and parentheses, with bounded work."""
    if len(expression) > 160:
        raise ValueError("expression too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, RecursionError) as exc:
        raise ValueError("invalid expression") from exc
    if len(list(ast.walk(tree))) > 64:
        raise ValueError("expression too complex")

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            result = Fraction(node.value)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            result = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                result = a + b
            elif isinstance(node.op, ast.Sub):
                result = a - b
            elif isinstance(node.op, ast.Mult):
                result = a * b
            else:
                if not b:
                    raise ValueError("division by zero")
                result = a / b
        else:
            raise ValueError("operation is not allowed")
        if abs(result.numerator) > 10**12 or result.denominator > 10**12:
            raise ValueError("expression exceeds bounds")
        return result

    return visit(tree.body)


def problem_key(expression):
    # Normalize associative/commutative + and *, retaining subtraction/division order.
    evaluate_expression(expression)
    tree = ast.parse(expression, mode="eval").body

    def canonical(node):
        if isinstance(node, ast.Constant):
            return str(node.value)
        if isinstance(node, ast.UnaryOp):
            return f"neg({canonical(node.operand)})" if isinstance(node.op, ast.USub) else canonical(node.operand)
        op = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}[type(node.op)]
        if op in ("+", "*"):
            def flatten(part):
                if isinstance(part, ast.BinOp) and type(part.op) is type(node.op):
                    return flatten(part.left) + flatten(part.right)
                return [canonical(part)]
            return op + "(" + ",".join(sorted(flatten(node))) + ")"
        return f"{op}({canonical(node.left)},{canonical(node.right)})"

    return canonical(tree)
