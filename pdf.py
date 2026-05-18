import base64
import io
import re
import time

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable, Image as RLImage, PageBreak,
    Paragraph, SimpleDocTemplate, Spacer,
)

from config import MODEL

def generate_pdf(report_md: str, plots_b64: list, dataset_name: str) -> bytes:
    """
    Convert a markdown report string + list of base64 chart images into a PDF.
    Returns raw PDF bytes.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm,  bottomMargin=2.5*cm,
        title="Agentic Analyser Report",
    )

    grey1   = colors.HexColor("#ededeb")
    grey2   = colors.HexColor("#8f8f9e")
    grey3   = colors.HexColor("#48485a")
    accent  = colors.HexColor("#c8ff3e")
    dark_bg = colors.HexColor("#18181c")

    sTitle  = ParagraphStyle("sTitle",  fontName="Helvetica-Bold",   fontSize=26, textColor=grey1,  spaceAfter=4,   leading=30)
    sMeta   = ParagraphStyle("sMeta",   fontName="Helvetica",         fontSize=9,  textColor=grey3,  spaceAfter=20)
    sH2     = ParagraphStyle("sH2",     fontName="Helvetica-Bold",   fontSize=15, textColor=grey1,  spaceBefore=16, spaceAfter=5)
    sH3     = ParagraphStyle("sH3",     fontName="Helvetica-Bold",   fontSize=11, textColor=grey1,  spaceBefore=10, spaceAfter=3)
    sBody   = ParagraphStyle("sBody",   fontName="Helvetica",         fontSize=10, textColor=grey2,  leading=15,     spaceAfter=7)
    sBullet = ParagraphStyle("sBullet", fontName="Helvetica",         fontSize=10, textColor=grey2,  leading=15,     spaceAfter=3, leftIndent=14)
    sCode   = ParagraphStyle("sCode",   fontName="Courier",           fontSize=8,
                              textColor=colors.HexColor("#ffb077"), backColor=dark_bg,
                              leading=12, leftIndent=8, rightIndent=8, spaceBefore=4, spaceAfter=4)
    sCapt   = ParagraphStyle("sCapt",   fontName="Helvetica-Oblique", fontSize=8,  textColor=grey3,
                              alignment=TA_CENTER, spaceAfter=10)

    story = [
        Spacer(1, 0.3*cm),
        HRFlowable(width="100%", thickness=3, color=accent, spaceAfter=14),
        Paragraph("Agentic Analyser", sTitle),
        Paragraph(
            f"Dataset: <b>{dataset_name}</b> &nbsp;·&nbsp; {time.strftime('%d %b %Y, %H:%M')}",
            sMeta,
        ),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1c1c22"), spaceAfter=20),
    ]

    def inline(t: str) -> str:
        t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        t = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', t)
        t = re.sub(r'\*(.+?)\*',     r'<i>\1</i>', t)
        t = re.sub(r'`([^`]+)`',     r'<font face="Courier" color="#ffb077">\1</font>', t)
        return t

    lines, i = report_md.split("\n"), 0
    while i < len(lines):
        ln = lines[i]
        if ln.strip().startswith("```"):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(
                    lines[i].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                )
                i += 1
            story.append(Paragraph("<br/>".join(code_lines), sCode))
        elif ln.startswith("## "):
            story += [
                HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1c1c22"),
                           spaceBefore=6, spaceAfter=0),
                Paragraph(inline(ln[3:].strip()), sH2),
            ]
        elif ln.startswith("### "):
            story.append(Paragraph(inline(ln[4:].strip()), sH3))
        elif ln.strip() in ("---", "***"):
            story.append(HRFlowable(width="100%", thickness=1,
                                    color=colors.HexColor("#1c1c22"),
                                    spaceBefore=6, spaceAfter=6))
        elif ln.startswith(("- ", "* ")):
            story.append(Paragraph("• " + inline(ln[2:].strip()), sBullet))
        elif re.match(r"^\d+\. ", ln):
            m = re.match(r"^(\d+)\. (.+)", ln)
            story.append(Paragraph(f"{m.group(1)}. {inline(m.group(2))}", sBullet))
        elif not ln.strip():
            story.append(Spacer(1, 3))
        else:
            story.append(Paragraph(inline(ln), sBody))
        i += 1

    if plots_b64:
        story += [
            PageBreak(),
            HRFlowable(width="100%", thickness=3, color=accent, spaceAfter=14),
            Paragraph("Charts", sH2),
            HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1c1c22"), spaceAfter=12),
        ]
        for idx, b64 in enumerate(plots_b64):
            try:
                img = RLImage(
                    io.BytesIO(base64.b64decode(b64)),
                    width=15*cm, height=10*cm, kind="proportional",
                )
                story += [img, Paragraph(f"Figure {idx + 1}", sCapt), Spacer(1, 0.4*cm)]
            except Exception:
                pass

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(grey3)
        canvas.drawString(2*cm, 1.1*cm, f"Agentic Analyser · {MODEL}")
        canvas.drawRightString(A4[0] - 2*cm, 1.1*cm, f"Page {doc.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    buf.seek(0)
    return buf.read()
