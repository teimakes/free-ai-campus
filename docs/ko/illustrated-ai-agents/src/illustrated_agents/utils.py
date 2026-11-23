import json
import html
import inspect
import difflib
import textwrap
import uuid

try:
    from pygments import highlight
    from pygments.lexers import DiffLexer, JsonLexer, PythonLexer
    from pygments.formatters import HtmlFormatter
    from rich.tree import Tree
    from rich.panel import Panel
    from rich.console import Console

    HAS_NOTEBOOK_DEPS = True
except ImportError:
    HAS_NOTEBOOK_DEPS = False


class Renderable:
    """Jupyter나 Marimo에서 쓰는 출력 클래스의 기본 클래스이다.

    인스턴스를 생성할 때는 별도 패키지가 필요 없고 화면에 표시할 때 rich와 pygments를 사용한다.
    하위 클래스에서 _render_html을 구현한다.
    """

    def _repr_html_(self):
        if not HAS_NOTEBOOK_DEPS:
            return (
                f"<i>{type(self).__name__}: 이 출력을 보려면 rich와 pygments를 "
                f"설치하세요.</i>"
            )
        return self._render_html()

    def _render_html(self) -> str:
        raise NotImplementedError


def _get_source(source):
    """소스 코드를 가져온다. 특정 메서드나 속성을 지정할 수 있다."""
    if type(source) is property:
        src = inspect.getsource(source.fget)
    else:
        src = inspect.getsource(source)
    return textwrap.dedent(src)


class DiffViewer(Renderable):
    """
    가져온 두 클래스의 코드 차이를 구문 강조와 함께
    Jupyter 노트북 셀에 표시한다.
    """

    def __init__(self, old_cls, new_cls, old_name="Old", new_name="New", style="github-dark"):
        self.old_cls = old_cls
        self.new_cls = new_cls
        self.old_name = old_name
        self.new_name = new_name
        self.style = style

    def _render_html(self):
        # 1. 소스 코드 추출 및 정리
        old_src = _get_source(self.old_cls)
        new_src = _get_source(self.new_cls)

        # 2. 두 버전의 차이 계산
        diff = "\n".join(
            difflib.unified_diff(
                old_src.splitlines(),
                new_src.splitlines(),
                fromfile=self.old_name,
                tofile=self.new_name,
                lineterm="",
            )
        )

        # 3. Pygments로 변경 내용과 파이썬 구문 서식 지정
        uid = f"dv-{uuid.uuid4().hex[:8]}"
        formatter = HtmlFormatter(style=self.style, cssclass=uid)
        bg = formatter.style.background_color

        # 4. 구문 강조
        html = highlight(diff, DiffLexer(), formatter)

        # 5. 노트북에 표시할 CSS 추가
        return f"""<style>
{formatter.get_style_defs(f".{uid}")}
.{uid}{{background:{bg}!important;color:#e6edf3!important;font-size:14px;line-height:1.4;border-radius:8px;padding:12px;overflow-x:auto}}
.{uid} pre{{background:{bg}!important;color:#e6edf3!important;margin:0}}
.{uid} code{{background:{bg}!important;color:#e6edf3!important}}
</style>
{html}"""


class CodeAnnotator(Renderable):
    """
    파이썬 코드 옆에 줄별 설명을 표시한다.
    """

    def __init__(
        self,
        source,
        annotations: dict[tuple | str, str] = None,
        style="github-dark",
        property: str = None,
    ):
        self.source = source
        self.property = property
        self.style = style
        self.annotations = {
            (k, k) if isinstance(k, int) else tuple(k): v for k, v in (annotations or {}).items()
        }

    def _render_html(self):
        src = _get_source(self.source)
        hl_lines = [ln for start, end in self.annotations for ln in range(start, end + 1)]

        uid = f"ca-{uuid.uuid4().hex[:8]}"
        fmt = HtmlFormatter(style=self.style, linenos="inline", cssclass=uid, hl_lines=hl_lines)
        code = highlight(src, PythonLexer(), fmt)
        bg = fmt.style.background_color

        # Pygments가 pre 시작 부분에 추가하는 빈 <span></span> 제거
        # 일부 노트북 환경에서 줄 정렬이 어긋날 수 있음
        code = code.replace("<pre><span></span>", "<pre>")

        # 설명과 코드 줄을 연결하는 div 요소 구성
        notes = "".join(
            f'<div style="position:absolute;top:{(start - 1) * 1.5}em;height:{(end - start + 1) * 1.5}em;'
            f'display:flex;align-items:center;font-size:14px;padding-left:16px;padding-right:16px">'
            f'<span style="position:absolute;left:0;top:0;bottom:0;width:3px;background:#f1c40f;border-radius:2px"></span>'
            f'<span style="color:#ddd;line-height:1.4">{text}</span></div>'
            for (start, end), text in sorted(self.annotations.items())
        )

        # Pygments가 생성하는 pre 태그의 기본 줄 간격 재지정
        # 줄 간격 설정이 우선 적용되도록 !important 사용
        return f"""<style>
pre{{line-height:1.5em!important}}
{fmt.get_style_defs(f".{uid}")}
.{uid}{{background:{bg}!important;color:#e6edf3!important;font-size:14px}}
.{uid} pre{{margin:0;padding:0;line-height:1.5em!important;background:{bg}!important;color:#e6edf3!important}}
.{uid} code{{background:{bg}!important;color:#e6edf3!important}}
.{uid} .hll{{background:#3d4626!important;display:block;width:100%}}
.{uid} .linenos{{background:{bg}!important;color:#e6edf3!important}}
</style>
<div style="display:inline-flex;align-items:flex-start;background:{bg};border-radius:8px;border:1px solid #444;margin:10px 0;font-family:monospace;font-size:14px;line-height:1.5em">
<div style="flex:3;overflow-x:auto;background:{bg}">{code}</div>
<div style="flex:1;min-width:500px;position:relative;background:{bg};border-left:1px solid #555;color:#ccc;font-family:system-ui,sans-serif;font-size:14px">{notes}</div>
</div>"""

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(self._repr_html_())


class ChapterOverview:
    """Jupyter에서 장별 구현 내용을 Rich 패널과 트리로 표시한다."""

    def __init__(self, modules):
        self.modules = modules

    def __repr__(self):
        if not HAS_NOTEBOOK_DEPS:
            return f"{type(self).__name__}: 표시하려면 rich를 설치하세요."
        tree = Tree("[bold]TinyAgent[/]")
        max_len = max(len(m[0]) for m in self.modules)

        for name, status, desc in self.modules:
            padded = name.ljust(max_len)
            if status == "new":
                tree.add(f"[bold green]{padded}[/] [dim]← [/][bold]추가 [dim]({desc})[/]")
            elif status == "updated":
                tree.add(f"[bold orange]{padded}[/] [dim]← [/][bold]수정 [dim]({desc})[/]")
            else:
                tree.add(f"[dim]{padded}[/]")

        panel = Panel(tree, title="이번 장에서 구현한 내용", border_style="dim")
        Console(force_terminal=True).print(panel)
        return ""


class TrajectoryViewer(Renderable):
    STYLE = """
    .trajectory details { margin: 6px 0; border: 1px solid #e4e4e7; border-radius: 6px; }
    .trajectory summary { cursor: pointer; padding: 8px 12px; font-weight: 500; background: #fafafa; }
    .trajectory summary .kind { color: #71717a; font-weight: 400; margin-left: 6px; }
    .trajectory .run > summary { background: #eef2ff; }
    .trajectory .steps { margin-left: 20px; }
    .trajectory .field { padding: 8px 12px; }
    .trajectory .label { color: #52525b; font-size: 0.8em; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px; }
    .trajectory pre { margin: 0; white-space: pre-wrap; font-family: ui-monospace, monospace; font-size: 0.9em; }
    .trajectory .highlight { background: transparent; font-size: 0.9em; }
    """

    def __init__(self, trajectory):
        self.trajectory = trajectory

    def _format(self, value):
        if isinstance(value, (dict, list)):
            code = json.dumps(value, indent=2, default=str)
            highlighted = highlight(code, JsonLexer(), HtmlFormatter(nowrap=True))
            return f"<div class='highlight'>{highlighted}</div>"
        return f"<pre>{html.escape(str(value))}</pre>"

    def _render_step(self, step_index, step):
        if step.answer:
            kind = "최종 답변"
        elif step.action:
            kind = step.action.get("name", "action")
        else:
            kind = "추론"
        fields = "\n".join(
            f"<div class='field'><div class='label'>{label}</div>{self._format(value)}</div>"
            for label, value in vars(step).items()
            if value
        )
        return f"<details><summary>{step_index}단계 <span class='kind'>· {kind}</span></summary>{fields}</details>"

    def _render_run(self, run_index, run):
        query = f"<details class='run'><summary>{run_index}번째 실행 <span class='kind'>· 사용자 질문</span></summary><div class='field'>{self._format(run['query'])}</div></details>"
        steps = "\n".join(
            self._render_step(step_index, step) for step_index, step in enumerate(run["steps"], start=1)
        )
        return f"{query}<div class='steps'>{steps}</div>"

    def _render_html(self):
        pygments_css = HtmlFormatter().get_style_defs(".highlight")
        runs = "\n".join(
            self._render_run(run_index, run) for run_index, run in enumerate(self.trajectory.runs, start=1)
        )
        return f"<style>{pygments_css}{self.STYLE}</style><div class='trajectory'>{runs}</div>"


class Exercise(Renderable):
    """힌트와 풀이를 펼쳐 볼 수 있는 Jupyter 실습이다.

    사용 예:
        Exercise(
            task="max_steps를 1로 설정하고 실행하면 어떤 결과가 나오는가?",
            hint="도구 결과를 확인하려면 두 단계 이상 필요하다.",
            solution="max_steps가 1이면 모델이 관찰 결과를 받을 다음 단계가 없다."
        )
    """

    def __init__(self, task, hint=None, solution=None):
        self.task = task
        self.hint = hint
        self.solution = solution

    def _render_html(self):
        uid = f"ex-{uuid.uuid4().hex[:8]}"
        bg = "#0d1117"

        sections = ""
        if self.hint:
            sections += (
                f'<details class="ex-section"><summary>힌트</summary>'
                f'<div class="ex-body">{html.escape(self.hint)}</div></details>'
            )
        if self.solution:
            sections += (
                f'<details class="ex-section"><summary>풀이</summary>'
                f'<div class="ex-body">{html.escape(self.solution)}</div></details>'
            )

        return f"""<style>
.{uid}{{background:{bg};border:1px solid #f1c40f44;border-left:3px solid #f1c40f;border-radius:6px;padding:16px 20px;margin:10px 0;font-family:system-ui,-apple-system,sans-serif;color:#e6edf3;font-size:14px;line-height:1.6}}
.{uid} .ex-title{{color:#f1c40f;font-weight:600;font-size:13px;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:10px}}
.{uid} .ex-task{{margin-bottom:14px;white-space:pre-line}}
.{uid} .ex-section{{margin:4px 0}}
.{uid} .ex-section>summary{{cursor:pointer;color:#8b949e;font-size:13px;padding:4px 0}}
.{uid} .ex-section>summary:hover{{color:#c9d1d9}}
.{uid} .ex-body{{color:#c9d1d9;font-size:13px;padding:8px 0 4px 20px;line-height:1.6;white-space:pre-line}}
</style>
<div class="{uid}">
<div class="ex-title">실습</div>
<div class="ex-task">{html.escape(self.task)}</div>
{sections}
</div>"""


class ComparisonViewer(Renderable):
    """여러 구현을 나란히 비교한다.

    사용 예:
        ComparisonViewer([
            ("Prompt-based", ch5.TinyAgent._execute_action),
            ("Native",       ch5_native.TinyAgent._execute_action),
        ])
    """

    def __init__(self, sources, style="github-dark"):
        self.sources = sources  # [(label, source), ...]
        self.style = style

    def _render_html(self):
        uid = f"cv-{uuid.uuid4().hex[:8]}"
        fmt = HtmlFormatter(style=self.style, linenos="inline", cssclass=uid)
        bg = fmt.style.background_color

        columns = ""
        for label, source in self.sources:
            src = _get_source(source)
            code = highlight(src, PythonLexer(), fmt)
            code = code.replace("<pre><span></span>", "<pre>")
            columns += (
                f'<div class="{uid}-col">'
                f'<div class="{uid}-label">{html.escape(label)}</div>'
                f'<div class="{uid}-code">{code}</div></div>'
            )

        return f"""<style>
{fmt.get_style_defs(f".{uid}")}
.{uid}-wrap{{display:flex;gap:0;background:{bg};border-radius:8px;border:1px solid #444;margin:10px 0;overflow-x:auto}}
.{uid}-col{{flex:1;min-width:0;border-right:1px solid #30363d}}
.{uid}-col:last-child{{border-right:none}}
.{uid}-label{{color:#8b949e;font-size:13px;font-family:system-ui,-apple-system,sans-serif;padding:8px 12px;border-bottom:1px solid #30363d;font-weight:600}}
.{uid}-code{{overflow-x:auto}}
.{uid}{{background:{bg}!important;color:#e6edf3!important;font-size:13px}}
.{uid} pre{{margin:0;padding:8px;line-height:1.5em!important;background:{bg}!important;color:#e6edf3!important}}
.{uid} .linenos{{background:{bg}!important;color:#e6edf3!important}}
</style>
<div class="{uid}-wrap">{columns}</div>"""


class EvolutionViewer(Renderable):
    """함수나 메서드가 장마다 어떻게 바뀌는지 표시한다.

    새로 추가되거나 수정된 줄을 단계마다 강조한다.

    사용 예:
        EvolutionViewer([
            ("Chapter 1", ch1.TinyAgent._step),
            ("Chapter 4", ch4.TinyAgent._step),
            ("Chapter 5", ch5.TinyAgent._step),
        ])
    """

    def __init__(self, stages, style="github-dark"):
        self.stages = stages  # [(label, source), ...]
        self.style = style

    def _render_html(self):
        uid = f"ev-{uuid.uuid4().hex[:8]}"
        bg = HtmlFormatter(style=self.style).style.background_color
        prev_lines = None
        blocks = []

        for label, source in self.stages:
            src = _get_source(source)
            curr_lines = src.splitlines()

            # 이전 버전과 비교해 추가하거나 변경한 줄 찾기
            hl_lines = []
            if prev_lines is not None:
                matcher = difflib.SequenceMatcher(None, prev_lines, curr_lines)
                for tag, _, _, j1, j2 in matcher.get_opcodes():
                    if tag in ("insert", "replace"):
                        hl_lines.extend(range(j1 + 1, j2 + 1))

            fmt = HtmlFormatter(style=self.style, linenos="inline", cssclass=uid, hl_lines=hl_lines)
            code = highlight(src, PythonLexer(), fmt)
            code = code.replace("<pre><span></span>", "<pre>")

            blocks.append(
                f'<div class="{uid}-stage">'
                f'<div class="{uid}-label">{html.escape(label)}</div>'
                f'<div class="{uid}-code">{code}</div></div>'
            )
            prev_lines = curr_lines

        connector = f'<div class="{uid}-conn">&#9660;</div>'
        body = connector.join(blocks)

        return f"""<style>
{fmt.get_style_defs(f".{uid}")}
.{uid}-wrap{{margin:10px 0}}
.{uid}-stage{{background:{bg};border:1px solid #444;border-radius:8px;overflow:hidden}}
.{uid}-label{{color:#8b949e;font-size:13px;font-family:system-ui,-apple-system,sans-serif;padding:8px 12px;border-bottom:1px solid #30363d;font-weight:600}}
.{uid}-code{{overflow-x:auto}}
.{uid}-conn{{text-align:center;color:#30363d;font-size:18px;padding:4px 0}}
.{uid}{{background:{bg}!important;color:#e6edf3!important;font-size:13px}}
.{uid} pre{{margin:0;padding:8px;line-height:1.5em!important;background:{bg}!important;color:#e6edf3!important}}
.{uid} .hll{{background:#3d4626!important;display:block;width:100%}}
.{uid} .linenos{{background:{bg}!important;color:#e6edf3!important}}
</style>
<div class="{uid}-wrap">{body}</div>"""
