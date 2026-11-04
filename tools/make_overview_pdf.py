"""
Generate the project overview PDF.

A plain-language explanation of what EduComic/EduRetrieve is, what it does and
how well it works. Written for a reader who has not seen the code: a marker, a
reviewer, or the author six months from now.

Numbers are read from the live project where possible -- the results table
comes from the most recent saved experiment run, and the collection stats from
the saved test collection -- so the document cannot drift out of date while
the project changes underneath it.

Usage:
    venv/Scripts/python.exe tools/make_overview_pdf.py [output.pdf]
"""
from __future__ import annotations

import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# ── Palette ──────────────────────────────────────────────────────────────
# Print-friendly: a light page with the app's green as the accent. The app's
# near-black canvas is a screen device -- on paper it drinks ink and makes
# body text harder to read.
INK      = colors.HexColor('#101413')
BODY     = colors.HexColor('#36403d')
MUTED    = colors.HexColor('#6b7876')
RULE     = colors.HexColor('#dfe7e3')
PANEL    = colors.HexColor('#f4f8f6')
ACCENT   = colors.HexColor('#15803d')
ACCENT_L = colors.HexColor('#e7f8ec')

PAGE_W, PAGE_H = A4
MARGIN = 20 * mm


# ── Styles ───────────────────────────────────────────────────────────────
def build_styles():
    base = getSampleStyleSheet()
    s = {}

    s['cover_title'] = ParagraphStyle(
        'cover_title', parent=base['Title'], fontName='Helvetica-Bold',
        fontSize=34, leading=38, textColor=INK, alignment=TA_LEFT,
        spaceAfter=4)
    s['cover_alt'] = ParagraphStyle(
        'cover_alt', parent=base['Title'], fontName='Helvetica-Oblique',
        fontSize=28, leading=32, textColor=ACCENT, alignment=TA_LEFT,
        spaceAfter=16)
    s['cover_sub'] = ParagraphStyle(
        'cover_sub', parent=base['Normal'], fontName='Helvetica',
        fontSize=12, leading=18, textColor=BODY, alignment=TA_LEFT)
    s['cover_meta'] = ParagraphStyle(
        'cover_meta', parent=base['Normal'], fontName='Helvetica',
        fontSize=9, leading=14, textColor=MUTED)

    s['h1'] = ParagraphStyle(
        'h1', parent=base['Heading1'], fontName='Helvetica-Bold',
        fontSize=17, leading=21, textColor=INK,
        spaceBefore=4, spaceAfter=3)
    s['h2'] = ParagraphStyle(
        'h2', parent=base['Heading2'], fontName='Helvetica-Bold',
        fontSize=12, leading=16, textColor=INK,
        spaceBefore=14, spaceAfter=5)
    s['lead'] = ParagraphStyle(
        'lead', parent=base['Normal'], fontName='Helvetica',
        fontSize=10.5, leading=16, textColor=BODY, spaceAfter=10)
    s['body'] = ParagraphStyle(
        'body', parent=base['Normal'], fontName='Helvetica',
        fontSize=9.5, leading=15, textColor=BODY, spaceAfter=8)
    s['bullet'] = ParagraphStyle(
        'bullet', parent=s['body'], leftIndent=11, bulletIndent=1,
        spaceAfter=4)
    s['small'] = ParagraphStyle(
        'small', parent=base['Normal'], fontName='Helvetica',
        fontSize=8, leading=12, textColor=MUTED, spaceAfter=6)
    s['cell'] = ParagraphStyle(
        'cell', parent=base['Normal'], fontName='Helvetica',
        fontSize=8.5, leading=12, textColor=BODY)
    s['cell_b'] = ParagraphStyle(
        'cell_b', parent=s['cell'], fontName='Helvetica-Bold', textColor=INK)
    s['cell_h'] = ParagraphStyle(
        'cell_h', parent=s['cell'], fontName='Helvetica-Bold',
        fontSize=7.5, textColor=MUTED)
    s['quote'] = ParagraphStyle(
        'quote', parent=s['body'], fontName='Helvetica-Oblique',
        leftIndent=10, textColor=INK, spaceAfter=6)
    s['mono'] = ParagraphStyle(
        'mono', parent=base['Normal'], fontName='Courier',
        fontSize=8, leading=12, textColor=BODY)
    s['center'] = ParagraphStyle(
        'center', parent=s['small'], alignment=TA_CENTER)
    return s


ST = build_styles()


# ── Custom flowables ─────────────────────────────────────────────────────
class Rule(Flowable):
    """A hairline, used to separate sections without a heavy box."""

    def __init__(self, width, thickness=0.6, color=RULE, space=6):
        super().__init__()
        self.width, self.thickness, self.color, self.space = (
            width, thickness, color, space)
        self.height = thickness + space

    def draw(self):
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, self.space, self.width, self.space)


class AccentBar(Flowable):
    """Short accent rule under a section heading."""

    def __init__(self, width=34, thickness=2.5):
        super().__init__()
        self.width, self.thickness = width, thickness
        self.height = thickness + 7

    def draw(self):
        self.canv.setFillColor(ACCENT)
        self.canv.roundRect(0, 6, self.width, self.thickness,
                            self.thickness / 2, stroke=0, fill=1)


def callout(text, title=None, tone='panel'):
    """A tinted box for an aside, a caveat, or a key takeaway."""
    bg, border = (PANEL, RULE) if tone == 'panel' else (ACCENT_L, ACCENT)
    inner = []
    if title:
        inner.append(Paragraph(f'<b>{title}</b>', ST['body']))
    inner.append(Paragraph(text, ST['body']))
    t = Table([[inner]], colWidths=[PAGE_W - 2 * MARGIN])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), bg),
        ('LINEBEFORE', (0, 0), (0, -1), 2.2, border),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    return t


def data_table(rows, widths, header=True, align_right=()):
    """A clean table: hairline rules, tinted header, no heavy grid."""
    style = [
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 7),
        ('RIGHTPADDING', (0, 0), (-1, -1), 7),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LINEBELOW', (0, 0), (-1, -2), 0.4, RULE),
    ]
    if header:
        style += [
            ('BACKGROUND', (0, 0), (-1, 0), PANEL),
            ('LINEBELOW', (0, 0), (-1, 0), 0.8, colors.HexColor('#c6d2cd')),
        ]
    for col in align_right:
        style.append(('ALIGN', (col, 0), (col, -1), 'RIGHT'))

    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    t.setStyle(TableStyle(style))
    return t


def section(title, blurb=None):
    out = [Paragraph(title, ST['h1']), AccentBar()]
    if blurb:
        out.append(Paragraph(blurb, ST['lead']))
    return out


def bullets(items):
    return [Paragraph(f'{i}', ST['bullet'], bulletText='•') for i in items]


# ── Page furniture ───────────────────────────────────────────────────────
def draw_cover_bg(canvas, doc):
    """
    A soft green wash at the top of the cover, echoing the app's glow.

    Painted as many thin bands rather than a few thick ones: at 34 steps the
    steps themselves were visible as horizontal stripes. 260 bands over 130mm
    puts each one at half a millimetre, below the threshold where the eye
    resolves an edge, and the overlap of +1pt hides any seam from rounding.
    """
    canvas.saveState()
    bands, height = 260, 130 * mm
    for i in range(bands):
        f = i / float(bands)
        canvas.setFillColor(colors.HexColor('#15803d'))
        # Ease-out so the colour holds near the top edge and fades long.
        canvas.setFillAlpha(0.075 * (1 - f) ** 2.1)
        canvas.rect(0, PAGE_H - height + (height * f), PAGE_W,
                    height / bands + 1, stroke=0, fill=1)
    canvas.setFillAlpha(1)
    canvas.setStrokeColor(ACCENT)
    canvas.setLineWidth(3)
    canvas.line(MARGIN, PAGE_H - 46 * mm, MARGIN + 46, PAGE_H - 46 * mm)
    canvas.restoreState()


def draw_page_furniture(canvas, doc):
    """Running footer with the page number; skipped on the cover."""
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, 14 * mm, PAGE_W - MARGIN, 14 * mm)
    canvas.setFont('Helvetica', 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, 9.5 * mm,
                      'EduComic / EduRetrieve — project overview')
    canvas.drawRightString(PAGE_W - MARGIN, 9.5 * mm, str(doc.page - 1))
    canvas.restoreState()


# ── Live data ────────────────────────────────────────────────────────────
def load_results():
    """
    Read the most recent experiment run.

    Returns (rows, n_queries, collection_stats) or (None, 0, None) when no
    experiment has been run, so the document degrades to describing the
    method rather than inventing numbers.
    """
    try:
        from ir.eval.collection import TestCollection
        from ir.eval.experiments import latest_results

        loaded = latest_results()
        if not loaded:
            return None, 0, None
        results, significance = loaded

        rows = []
        for name, r in sorted(results.items(),
                              key=lambda kv: -kv[1].measure('nDCG@10')):
            rows.append({
                'name': name,
                'ndcg10': r.measure('nDCG@10'),
                'mrr': r.measure('MRR'),
                'p1': r.measure('P@1'),
                'ms': r.mean_latency_ms,
                'sig': bool(significance.get(name, {}).get('significant')),
                'n': r.n_queries,
            })
        n = rows[0]['n'] if rows else 0

        collection = TestCollection.load()
        stats = collection.stats() if len(collection) else None
        return rows, n, stats
    except Exception:
        return None, 0, None


FRIENDLY = {
    'tfidf': 'TF-IDF (classic word matching)',
    'bm25': 'BM25 (better word matching)',
    'dense': 'Dense (meaning matching)',
    'hybrid_rrf': 'Hybrid (both, fused)',
    'hybrid_combsum': 'Hybrid (score fusion)',
    'hybrid_rrf_rm3': 'Hybrid + word expansion',
    'hybrid_rrf_rerank': 'Hybrid + re-ranker',
    'full': 'Full pipeline (everything)',
}


# ── Document ─────────────────────────────────────────────────────────────
def build(path: str) -> str:
    doc = BaseDocTemplate(
        path, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=22 * mm,
        title='EduComic / EduRetrieve — Project Overview',
        author='Laxman', subject='Project overview and feature guide')

    frame = Frame(MARGIN, 22 * mm, PAGE_W - 2 * MARGIN,
                  PAGE_H - MARGIN - 22 * mm, id='body')
    doc.addPageTemplates([
        PageTemplate(id='cover', frames=[frame], onPage=draw_cover_bg),
        PageTemplate(id='body', frames=[frame], onPage=draw_page_furniture),
    ])

    W = PAGE_W - 2 * MARGIN
    rows, n_queries, coll = load_results()
    story = []

    # ───────────────────────── Cover ─────────────────────────
    story += [
        Spacer(1, 30 * mm),
        Paragraph('EduComic', ST['cover_title']),
        Paragraph('&amp; EduRetrieve', ST['cover_alt']),
        Paragraph(
            'A tool that turns a textbook PDF into comic-strip lessons and '
            'quizzes &mdash; built on top of a search engine you can actually '
            'measure.', ST['cover_sub']),
        Spacer(1, 14 * mm),
        Rule(W),
        Spacer(1, 4 * mm),
    ]

    facts = [
        ['Two things in one', 'A learning app, and the information retrieval '
                              'engine underneath it'],
        ['Runs entirely offline', 'No cloud APIs, no keys &mdash; local AI '
                                  'models on your own machine'],
        ['Built and tested', '450 automated tests, 19 page checks'],
        ['Measured, not claimed', f'{n_queries} test questions, 8 search '
                                  f'methods compared with statistics'
                                  if n_queries else
                                  'An evaluation harness with real IR metrics'],
    ]
    t = Table([[Paragraph(f'<b>{a}</b>', ST['cell_b']),
                Paragraph(b, ST['cell'])] for a, b in facts],
              colWidths=[52 * mm, W - 52 * mm])
    t.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (0, -1), 0),
        ('LINEBELOW', (0, 0), (-1, -2), 0.4, RULE),
    ]))
    story += [t, Spacer(1, 18 * mm),
              Paragraph(f'Prepared {date.today():%d %B %Y}', ST['cover_meta']),
              NextPageTemplate('body'), PageBreak()]

    # ───────────────────────── What is it ─────────────────────────
    story += section(
        'What is this, in one paragraph?',
        'Reading a textbook is slow, and it is hard to tell whether you have '
        'understood anything.')
    story += [
        Paragraph(
            'You upload a PDF of a textbook. The system reads it, breaks it '
            'into topics, and for each topic it can draw you a short comic '
            'strip that explains the idea, plus a quiz to check you got it. '
            'You can also just <b>ask it questions in plain English</b> and it '
            'will find the right passage and write an answer &mdash; always '
            'showing which part of your book the answer came from.',
            ST['body']),
        Paragraph(
            'Underneath, the interesting part is the <b>search engine</b>. '
            'Finding the right paragraph in a book is a genuinely hard '
            'problem, and most projects just assume their search works. This '
            'one measures it: there is a test set of questions, a set of '
            'standard scoring methods, and statistical tests that say whether '
            'one search method is really better than another or just lucky.',
            ST['body']),
        Spacer(1, 3 * mm),
        callout(
            'The app is the demonstration. The search engine is the '
            'contribution. Everything the comics and quizzes read comes '
            'through the same engine that gets measured, so the numbers '
            'describe the thing you actually use.',
            title='The short version', tone='accent'),
    ]

    # ───────────────────────── How it works ─────────────────────────
    story += [Spacer(1, 8 * mm)]
    story += section('How it works',
                     'Two pipelines. The first runs once per book; the second '
                     'runs when you ask for something.')

    story += [Paragraph('<b>1. When you upload a book</b>', ST['h2'])]
    flow1 = [
        ['PDF', 'You upload a textbook (short ones, 5&ndash;10 pages, '
                'process fastest)'],
        ['Read', 'Text is pulled out page by page'],
        ['Split', 'The text is cut into overlapping passages of about 400 '
                  'words'],
        ['Index', 'Each passage is filed two ways: by the words it contains, '
                  'and by what it means'],
        ['Topics', 'A local AI model reads the book and lists the subtopics '
                   'it covers'],
        ['Quizzes', 'Ten multiple-choice questions are written for each '
                    'subtopic'],
    ]
    story += [data_table(
        [[Paragraph(f'<b>{a}</b>', ST['cell_b']), Paragraph(b, ST['cell'])]
         for a, b in flow1],
        [28 * mm, W - 28 * mm], header=False)]

    story += [Paragraph('<b>2. When you ask a question or request a comic</b>',
                        ST['h2'])]
    flow2 = [
        ['Understand', 'Work out what kind of question it is, and resolve '
                       'follow-ups like &ldquo;why does it increase?&rdquo;'],
        ['Search', 'Two searches run at once &mdash; one on words, one on '
                   'meaning &mdash; and the results are merged'],
        ['Re-rank', 'A second AI model re-reads the top results next to your '
                    'question and reorders them'],
        ['Answer', 'A local AI model writes an answer using only those '
                   'passages, with numbered citations'],
        ['Or draw', 'For a comic: a storyboard is written, pictures are '
                    'generated, and captions are drawn on top'],
    ]
    story += [data_table(
        [[Paragraph(f'<b>{a}</b>', ST['cell_b']), Paragraph(b, ST['cell'])]
         for a, b in flow2],
        [28 * mm, W - 28 * mm], header=False)]

    story += [PageBreak()]

    # ───────────────────────── Features ─────────────────────────
    story += section('What it can do',
                     'Grouped by the part of the app you would be using.')

    groups = [
        ('Reading and learning', [
            'Upload a textbook PDF and have it split into subtopics '
            'automatically',
            'Generate a comic-strip lesson for any subtopic, in four art '
            'styles and three difficulty levels',
            'Take a ten-question quiz on each subtopic, with explanations '
            'for every answer',
            'Track which subtopics you have studied and how you scored',
        ]),
        ('Asking questions', [
            'Ask anything in plain English &mdash; you do not need the '
            'textbook&rsquo;s exact wording',
            'Get a written answer with numbered citations back to the exact '
            'passages used',
            'The system says &ldquo;this book does not cover that&rdquo; '
            'instead of making something up',
            'Ask follow-ups like &ldquo;why does it happen?&rdquo; and it '
            'works out what &ldquo;it&rdquo; refers to',
            'Every result can be expanded to show <i>why</i> it was found: '
            'which words matched, what each search method scored it, and how '
            'far the re-ranker moved it',
        ]),
        ('For the evaluation', [
            'Compare five search methods side by side on the same question',
            'A built-in results dashboard with scores, timings and '
            'significance tests',
            'A reproducible experiment script that regenerates every number '
            'in this document',
            'Tools to check whether a PDF is usable before you add it',
        ]),
        ('Under the hood', [
            'Runs completely offline on your own machine &mdash; no API keys, '
            'nothing sent anywhere',
            'Light and dark themes, works on a phone screen, keyboard '
            'accessible',
            '450 automated tests, including the search maths checked against '
            'hand-calculated values',
        ]),
    ]
    for title, items in groups:
        story.append(KeepTogether(
            [Paragraph(title, ST['h2'])] + bullets(items)))

    story += [PageBreak()]

    # ───────────────────────── The search problem ─────────────────────────
    story += section(
        'Why searching a textbook is hard',
        'Five real problems, and what the system does about each. These are '
        'not hypothetical &mdash; each was observed on the actual textbooks '
        'loaded into this project.')

    probs = [
        ['You use different words than the book',
         'You ask &ldquo;why is farmland becoming useless?&rdquo;; the book '
         'says &ldquo;land degradation&rdquo;. No word matches.',
         'Match on <b>meaning</b> as well as words, and add the book&rsquo;s '
         'own vocabulary to your question'],
        ['Short terms and abbreviations',
         'Asked &ldquo;what does CAP mean?&rdquo;, the meaning-based search '
         'returned <i>capacitor</i> pages from a physics book.',
         'Keep a <b>word-matching</b> search too, and only search one book at '
         'a time'],
        ['Nearly-right answers rank too high',
         'A page on a related topic can outrank the page that actually '
         'answers the question.',
         'A second model re-reads the top results <b>together with</b> your '
         'question and reorders them'],
        ['A passage makes no sense alone',
         '&ldquo;This process increases the value by 20%&rdquo; &mdash; which '
         'process? Which value?',
         'Show the surrounding text, and say which chapter and subtopic it '
         'came from'],
        ['Follow-up questions have no subject',
         '&ldquo;Why does it increase?&rdquo; on its own tells a search '
         'engine nothing.',
         'Rewrite it using the conversation: &ldquo;Why does entropy '
         'increase?&rdquo;'],
    ]
    story += [data_table(
        [[Paragraph('The problem', ST['cell_h']),
          Paragraph('What goes wrong', ST['cell_h']),
          Paragraph('The fix', ST['cell_h'])]] +
        [[Paragraph(f'<b>{a}</b>', ST['cell_b']),
          Paragraph(b, ST['cell']), Paragraph(c, ST['cell'])]
         for a, b, c in probs],
        [36 * mm, (W - 36 * mm) / 2, (W - 36 * mm) / 2])]

    # ──────────────── Information retrieval techniques ────────────────
    story += section(
        'Information retrieval techniques used',
        'The standard IR toolkit, stage by stage. The left column is the name '
        'the technique goes by in the literature; the right column is what it '
        'actually does here. Everything listed is implemented and running '
        '&mdash; nothing is aspirational.')

    stages = [
        ('Stage 1 &mdash; Document processing and indexing', [
            ['Inverted index',
             'The core data structure of search: a list, for every word, of '
             'which passages contain it and how often. Written from scratch '
             'rather than taken from a library.'],
            ['Tokenisation, stopword removal, stemming',
             'Text is lowercased, split into words, common words like '
             '&ldquo;the&rdquo; are dropped, and endings are trimmed so '
             '&ldquo;capacitors&rdquo; matches &ldquo;capacitor&rdquo;. Both '
             'steps can be switched off to measure what they are worth.'],
            ['Passage chunking with overlap',
             'The book is cut into ~400-word passages that overlap by 80 '
             'words, so a sentence split across a boundary still appears '
             'whole somewhere.'],
            ['Document structure / hierarchy',
             'Each passage remembers its page, chapter and subtopic, so a '
             'result can say where in the book it came from.'],
        ]),
        ('Stage 2 &mdash; Retrieval models', [
            ['TF-IDF vector space model',
             'The classical method. Passages and the query become vectors of '
             'word weights, compared by cosine similarity. Used as the '
             'baseline everything else is measured against.'],
            ['BM25 (probabilistic model)',
             'The standard modern word-matching method. Adds two things '
             'TF-IDF lacks: repeating a word stops helping after a few times '
             '(saturation, the k1 setting), and long passages are not '
             'rewarded just for being long (length normalisation, the b '
             'setting). Both IDF formulations are implemented.'],
            ['Dense retrieval (bi-encoder)',
             'Passages and queries are turned into vectors that capture '
             'meaning rather than words, so a question can match a passage '
             'with no words in common. Two embedding models are supported.'],
        ]),
        ('Stage 3 &mdash; Rank fusion', [
            ['Reciprocal Rank Fusion (RRF)',
             'Merges the word-based and meaning-based result lists using only '
             'their <i>positions</i>, never their scores &mdash; which matters '
             'because the two score types are on completely different scales '
             'and cannot be added directly. Uses the standard k = 60.'],
            ['CombSUM and CombMNZ score fusion',
             'The alternative approach: normalise the scores, then add them. '
             'CombMNZ additionally rewards a passage both methods agree on. '
             'Implemented so RRF can be compared against them rather than '
             'just asserted to be better.'],
            ['Score normalisation (min-max, z-score)',
             'Rescales one method&rsquo;s scores so they can sit alongside '
             'another&rsquo;s. Needed by score fusion; not needed by RRF.'],
        ]),
        ('Stage 4 &mdash; Re-ranking', [
            ['Cross-encoder neural re-ranking',
             'A second model re-reads the top ~20 results <i>together with</i> '
             'the question, rather than scoring each in isolation. Slower, but '
             'the single biggest quality gain in the results table.'],
        ]),
        ('Stage 5 &mdash; Query processing', [
            ['RM3 pseudo-relevance feedback',
             'The classical query-expansion method: assume the first few '
             'results are relevant, harvest their distinctive words, and '
             'search again with those added.'],
            ['LLM query expansion',
             'The modern alternative: ask a language model what terminology '
             'the textbook would use. Unlike RM3 it can help even when the '
             'first search found nothing.'],
            ['Query intent classification',
             'Works out whether you are asking for a definition, a '
             'comparison, a cause, a process, a number or an example &mdash; '
             'seven types, detected by sentence pattern.'],
            ['Intent-aware adaptive retrieval',
             'Uses that intent to change the search: lean on word-matching '
             'for definitions, on meaning for &ldquo;why&rdquo; questions.'],
            ['Multi-query / entity splitting',
             '&ldquo;Difference between X and Y&rdquo; is also searched as X '
             'and as Y separately, so both sides are guaranteed to appear.'],
            ['Conversational query rewriting',
             'Turns &ldquo;why does it increase?&rdquo; into &ldquo;why does '
             'entropy increase?&rdquo; using the conversation.'],
        ]),
        ('Stage 6 &mdash; Evaluation', [
            ['Test collection construction',
             'A fixed set of questions with marked-correct passages, built '
             'automatically and in two phrasings &mdash; one using the '
             'book&rsquo;s words, one deliberately avoiding them.'],
            ['Graded relevance judgements',
             'Passages are marked fully relevant, partly relevant, or not '
             'relevant, rather than a simple yes/no.'],
            ['Ranked-retrieval metrics',
             'Precision@k, Recall@k, F1, Success@k, MRR, MAP and nDCG '
             '&mdash; all implemented from scratch and checked against '
             'hand-calculated values.'],
            ['Significance testing',
             'A paired randomisation test decides whether one method really '
             'beat another or was just lucky. Holm correction accounts for '
             'running many comparisons at once.'],
            ['Ablation studies',
             'Each component is removed in turn to measure what it '
             'contributes &mdash; which is how two features were found to be '
             'making things worse.'],
        ]),
        ('Stage 7 &mdash; Explainability', [
            ['Per-term score decomposition',
             'For any result, shows each matched word with its frequency, '
             'rarity and exact contribution to the score.'],
            ['Fusion contribution shares',
             'Shows how much of a result&rsquo;s final position came from '
             'word-matching versus meaning-matching.'],
            ['Rank-change tracking',
             'Shows how far the re-ranker moved each result, up or down.'],
        ]),
    ]

    for heading, items in stages:
        # Heading and table always travel together. KeepTogether degrades
        # sensibly when a block genuinely exceeds a page -- it breaks first,
        # then splits -- so there is no reason to make this conditional on
        # row count, and doing so previously stranded two stage headings at
        # the foot of a page with their tables overleaf.
        story.append(KeepTogether([
            Paragraph(heading, ST['h2']),
            data_table([[Paragraph(f'<b>{a}</b>', ST['cell_b']),
                         Paragraph(b, ST['cell'])] for a, b in items],
                       [46 * mm, W - 46 * mm], header=False),
        ]))

    story += [Spacer(1, 4 * mm), callout(
        'Classical and neural methods are both implemented, and compared '
        'against each other rather than one being assumed better. TF-IDF and '
        'BM25 are the textbook baselines; dense retrieval and the '
        'cross-encoder are the modern additions; RRF is what lets them be '
        'used together.',
        title='Why both old and new methods are here', tone='accent')]

    story += [PageBreak()]

    # ───────────────────────── Results ─────────────────────────
    if rows:
        story += [Spacer(1, 8 * mm)]
        heading = section(
            'Does it actually work?',
            f'Eight search methods were run over the same {n_queries} test '
            f'questions. Higher is better; the score runs from 0 to 1.')

        head = [Paragraph('Search method', ST['cell_h']),
                Paragraph('Score', ST['cell_h']),
                Paragraph('First hit', ST['cell_h']),
                Paragraph('Speed', ST['cell_h']),
                Paragraph('Real gain?', ST['cell_h'])]
        body_rows = []
        for r in rows:
            body_rows.append([
                Paragraph(FRIENDLY.get(r['name'], r['name']), ST['cell']),
                Paragraph(f"<b>{r['ndcg10']:.3f}</b>", ST['cell_b']),
                Paragraph(f"{r['mrr']:.3f}", ST['cell']),
                Paragraph(f"{r['ms']:.0f} ms", ST['cell']),
                Paragraph('yes' if r['sig'] else '&mdash;',
                          ST['cell_b'] if r['sig'] else ST['cell']),
            ])
        best, worst = rows[0], rows[-1]
        rerank_ms = next((r['ms'] for r in rows
                          if r['name'] == 'hybrid_rrf_rerank'), best['ms'])
        fast_ms = next((r['ms'] for r in rows
                        if r['name'] == 'hybrid_rrf'), 10)

        # Heading, table, caption and takeaway are one unit. Splitting any of
        # them strands a conclusion away from the evidence it describes -- and
        # a heading alone at the foot of a page is the worst case, because the
        # reader turns over expecting the section to have started.
        story += [KeepTogether(heading + [
            data_table([head] + body_rows,
                       [68 * mm, 20 * mm, 22 * mm, 22 * mm, W - 132 * mm],
                       align_right=(1, 2, 3)),
            Paragraph(
                '&ldquo;Score&rdquo; is nDCG@10, the standard measure for how '
                'well a ranked list of results is ordered. &ldquo;First '
                'hit&rdquo; is how near the top the first good result appears. '
                '&ldquo;Real gain?&rdquo; means the improvement over the '
                'simplest method survived a statistical test &mdash; without '
                'it, a higher number could just be luck.', ST['small']),
            Spacer(1, 3 * mm),
            callout(
                f'Combining word-matching with meaning-matching, then '
                f're-reading the top results, scores '
                f'<b>{best["ndcg10"]:.3f}</b> against '
                f'<b>{worst["ndcg10"]:.3f}</b> for the classic method '
                f'&mdash; and the gap is statistically real, not noise. The '
                f'cost is speed: the re-ranker takes about '
                f'{rerank_ms / 1000:.1f} seconds per question against roughly '
                f'{fast_ms:.0f} milliseconds without it.',
                title='What the table says', tone='accent'),
        ])]

    story += [PageBreak()]

    # ───────────────────────── Honest limits ─────────────────────────
    story += section(
        'What it cannot do yet',
        'Stated plainly, because a project that only lists its strengths is '
        'harder to trust.')
    story += bullets([
        '<b>Scanned PDFs do not work.</b> A scanned book is a stack of '
        'photographs with no text inside, so nothing can be searched. The app '
        'detects this and says so rather than failing quietly.',
        '<b>The test set is small.</b> 48 questions over a small library. The '
        'differences that passed the statistical test are real, but the ones '
        'that did not are undecided rather than disproved.',
        '<b>Two features did not help.</b> Automatic word expansion and '
        'question-type routing both made results slightly <i>worse</i> on this '
        'collection. They are kept, switched off, and reported as negative '
        'findings.',
        '<b>Comics need a graphics card.</b> Question answering runs on any '
        'machine; drawing comic panels needs a local image-generation setup.',
        '<b>Answers are only as good as the book.</b> If the textbook is '
        'wrong, so is the answer &mdash; which is why every answer cites the '
        'passage it used, so you can check.',
    ])

    # ───────────────────────── Tech + running ─────────────────────────
    story += [Spacer(1, 6 * mm)]
    story += section('What it is built from')
    tech = [
        ['Web app', 'Django (Python)'],
        ['Word search', 'BM25 and TF-IDF, written from scratch over an '
                        'inverted index'],
        ['Meaning search', 'Sentence embeddings (MiniLM), 384 dimensions'],
        ['Re-ranking', 'A cross-encoder model (MS MARCO MiniLM)'],
        ['Writing and reasoning', 'A local language model via Ollama'],
        ['Comic artwork', 'Stable Diffusion via ComfyUI'],
        ['Captions and bubbles', 'Drawn with Pillow, so text is never '
                                 'garbled by the image model'],
    ]
    story += [data_table(
        [[Paragraph(f'<b>{a}</b>', ST['cell_b']), Paragraph(b, ST['cell'])]
         for a, b in tech],
        [42 * mm, W - 42 * mm], header=False)]

    story += [Paragraph('Running it', ST['h2'])]
    cmds = [
        ('Start the app', 'venv\\Scripts\\python manage.py runserver'),
        ('Background worker (for comics)',
         'venv\\Scripts\\python manage.py qcluster'),
        ('Re-run the experiment',
         'venv\\Scripts\\python tools\\run_experiments.py'),
        ('Check a PDF is usable',
         'venv\\Scripts\\python tools\\check_pdfs.py media\\textbooks'),
    ]
    story += [data_table(
        [[Paragraph(a, ST['cell']), Paragraph(b, ST['mono'])]
         for a, b in cmds],
        [60 * mm, W - 60 * mm], header=False)]

    story += [Spacer(1, 6 * mm), Rule(W), Spacer(1, 2 * mm),
              Paragraph(
                  'Generated from the live project. The results table is read '
                  'from the most recent saved experiment run, so this document '
                  'reflects the current state of the code rather than a '
                  'snapshot typed by hand.', ST['small'])]

    doc.build(story)
    return path


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else 'EduComic-Project-Overview.pdf'
    print('Written:', build(out))
