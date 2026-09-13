"""Render the project report HTML into a portable PDF with ReportLab."""

from __future__ import annotations

import html
import re
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    KeepTogether,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).parent
SOURCE = ROOT / "informe_proyecto.html"
TARGET = ROOT / "informe_proyecto.pdf"


def clean_text(node: Tag | NavigableString) -> str:
    text = node.get_text(" ", strip=True) if isinstance(node, Tag) else str(node)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def inline_markup(node: Tag) -> str:
    """Convert simple inline HTML into ReportLab paragraph markup."""
    if isinstance(node, NavigableString):
        return html.escape(str(node))
    if node.name in {"math", "code", "span"}:
        return html.escape(clean_text(node))
    content = "".join(inline_markup(child) for child in node.children)
    if node.name in {"strong", "b"}:
        return f"<b>{content}</b>"
    if node.name in {"em", "i"}:
        return f"<i>{content}</i>"
    if node.name == "br":
        return "<br/>"
    return content


def paragraph(node: Tag, style: ParagraphStyle) -> Paragraph:
    return Paragraph(inline_markup(node), style)


def render_list(node: Tag, styles: dict[str, ParagraphStyle], ordered: bool = False):
    items = []
    for index, item in enumerate(node.find_all("li", recursive=False), start=1):
        prefix = f"{index}. " if ordered else "• "
        items.append(Paragraph(prefix + inline_markup(item), styles["BodyText"]))
        items.append(Spacer(1, 0.07 * cm))
    return items


def render_table(node: Tag, styles: dict[str, ParagraphStyle]):
    rows = []
    for tr in node.find_all("tr"):
        cells = []
        for cell in tr.find_all(["th", "td"], recursive=False):
            cells.append(Paragraph(inline_markup(cell), styles["TableText"]))
        if cells:
            rows.append(cells)
    if not rows:
        return []
    widths = [17.0 * cm / max(len(rows[0]), 1)] * len(rows[0])
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f3d4c")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5dc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f8f7")]),
    ]))
    return [table, Spacer(1, 0.22 * cm)]


def build_story(soup: BeautifulSoup):
    base = getSampleStyleSheet()
    styles = {
        "Title": ParagraphStyle("Title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=22, leading=26, textColor=colors.HexColor("#0f3d4c"), alignment=TA_CENTER, spaceAfter=10),
        "Subtitle": ParagraphStyle("Subtitle", parent=base["Normal"], fontName="Helvetica", fontSize=13, leading=16, textColor=colors.HexColor("#64748b"), alignment=TA_CENTER, spaceAfter=18),
        "H1": ParagraphStyle("H1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=16, leading=19, textColor=colors.HexColor("#0f3d4c"), spaceBefore=12, spaceAfter=7, keepWithNext=True),
        "H2": ParagraphStyle("H2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=15, textColor=colors.HexColor("#23636a"), spaceBefore=8, spaceAfter=5, keepWithNext=True),
        "H3": ParagraphStyle("H3", parent=base["Heading3"], fontName="Helvetica-Bold", fontSize=10.5, leading=13, textColor=colors.HexColor("#23636a"), spaceBefore=6, spaceAfter=4, keepWithNext=True),
        "BodyText": ParagraphStyle("BodyText", parent=base["BodyText"], fontName="Helvetica", fontSize=9.2, leading=12.2, textColor=colors.HexColor("#172033"), spaceAfter=5),
        "TableText": ParagraphStyle("TableText", parent=base["BodyText"], fontName="Helvetica", fontSize=7.5, leading=9, textColor=colors.HexColor("#172033")),
        "Quote": ParagraphStyle("Quote", parent=base["BodyText"], fontName="Helvetica-Oblique", fontSize=9.2, leading=12.2, leftIndent=10, borderColor=colors.HexColor("#d58a4a"), borderWidth=2, borderPadding=6, backColor=colors.HexColor("#fff7ed"), spaceBefore=5, spaceAfter=7),
        "Code": ParagraphStyle("Code", parent=base["Code"], fontName="Courier", fontSize=7.8, leading=10, backColor=colors.HexColor("#eef6f5"), borderColor=colors.HexColor("#2d7a78"), borderWidth=2, borderPadding=6, spaceBefore=4, spaceAfter=7),
    }
    story = []
    body = soup.body or soup
    for node in body.children:
        if not isinstance(node, Tag):
            continue
        name = node.name
        if name == "header":
            continue
        if name == "h1":
            story.append(Paragraph(clean_text(node), styles["Title"]))
            story.append(HRFlowable(width="70%", thickness=1, color=colors.HexColor("#d58a4a"), hAlign="CENTER"))
        elif name == "h2":
            story.append(Paragraph(clean_text(node), styles["H1"]))
        elif name == "h3":
            story.append(Paragraph(clean_text(node), styles["H2"]))
        elif name == "p":
            style = styles["Quote"] if node.find_parent("blockquote") else styles["BodyText"]
            story.append(paragraph(node, style))
        elif name == "blockquote":
            story.append(Paragraph(inline_markup(node), styles["Quote"]))
        elif name in {"ul", "ol"}:
            story.extend(render_list(node, styles, ordered=name == "ol"))
        elif name == "pre":
            story.append(Preformatted(clean_text(node), styles["Code"], maxLineLength=110))
        elif name == "table":
            story.extend(render_table(node, styles))
    return story


def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawRightString(A4[0] - 2.1 * cm, 1.1 * cm, f"Página {doc.page}")
    canvas.restoreState()


def main() -> None:
    soup = BeautifulSoup(SOURCE.read_text(encoding="utf-8"), "html.parser")
    doc = BaseDocTemplate(
        str(TARGET), pagesize=A4, leftMargin=2.1 * cm, rightMargin=2.1 * cm,
        topMargin=1.8 * cm, bottomMargin=1.7 * cm, title="Informe del proyecto ITA",
        author="Proyecto ITA",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates([PageTemplate(id="report", frames=frame, onPage=add_page_number)])
    doc.build(build_story(soup))
    print(TARGET)


if __name__ == "__main__":
    main()