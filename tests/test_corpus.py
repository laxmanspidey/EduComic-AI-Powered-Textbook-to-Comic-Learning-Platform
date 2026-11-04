"""
Tests for ir.corpus.

The most important test here is
`TestChapterAttachment.test_pages_outside_every_chapter_are_still_indexed`.
It is the regression test for the bug this module exists to fix: the original
pipeline chunked only inside detected chapter ranges, so pages no chapter
covered were never indexed at all.

No PDF is needed -- PageText objects are constructed directly, so these tests
exercise chunking and structure attachment without touching PyMuPDF.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ir.corpus import (  # noqa: E402
    Corpus,
    CorpusStats,
    PageText,
    Passage,
    assign_parents,
    attach_chapters,
    build_from_texts,
    chunk_continuous,
    chunk_page_bounded,
    clean_text,
)


def pages(*word_counts: int) -> list[PageText]:
    """
    Build pages whose words are globally unique and encode their page number,
    e.g. page 2 -> 'p2w0 p2w1 ...'. That makes page attribution checkable by
    reading the passage text.
    """
    return [
        PageText(page=i + 1, text=' '.join(f'p{i + 1}w{j}' for j in range(n)))
        for i, n in enumerate(word_counts)
    ]


class TestCleanText(unittest.TestCase):

    def test_dehyphenates_across_line_break(self):
        # A hyphen-split token is unmatchable: no query contains "capac".
        self.assertEqual(clean_text('capac-\nitor'), 'capacitor')

    def test_collapses_whitespace(self):
        self.assertEqual(clean_text('a   b\n\n\tc'), 'a b c')

    def test_strips_isolated_page_number_lines(self):
        self.assertNotIn('42', clean_text('text\n 42 \nmore text'))

    def test_keeps_numbers_inside_prose(self):
        self.assertIn('42', clean_text('the answer is 42 exactly'))


class TestChunkContinuous(unittest.TestCase):

    def test_crosses_page_boundaries(self):
        # Two 30-word pages, chunk of 50: the first passage must contain words
        # from both pages and report the full span.
        chunks = chunk_continuous(pages(30, 30), chunk_size=50, overlap=0)
        text, ps, pe = chunks[0]
        self.assertIn('p1w0', text)
        self.assertIn('p2w0', text)
        self.assertEqual((ps, pe), (1, 2))

    def test_page_span_is_min_and_max_of_contributing_pages(self):
        chunks = chunk_continuous(pages(10, 10, 10), chunk_size=30, overlap=0)
        self.assertEqual((chunks[0][1], chunks[0][2]), (1, 3))

    def test_overlap_shares_words_between_adjacent_chunks(self):
        chunks = chunk_continuous(pages(100), chunk_size=40, overlap=10)
        first = set(chunks[0][0].split())
        second = set(chunks[1][0].split())
        self.assertEqual(len(first & second), 10)

    def test_stride_is_size_minus_overlap(self):
        # 100 words, size 40, overlap 10 -> stride 30 -> windows start at
        # 0, 30, 60. The window at 60 reaches word 100, i.e. the end, so
        # iteration stops: a fourth window at 90 would cover words 90-99,
        # which the third window already contains, and emitting it would
        # duplicate content in the index for no gain.
        chunks = chunk_continuous(pages(100), chunk_size=40, overlap=10)
        self.assertEqual(len(chunks), 3)

    def test_overlapping_windows_still_cover_every_word(self):
        # No coverage gap despite stopping early: 0-39, 30-69, 60-99.
        chunks = chunk_continuous(pages(100), chunk_size=40, overlap=10)
        covered = {w for text, _, _ in chunks for w in text.split()}
        self.assertEqual(len(covered), 100)

    def test_no_overlap_partitions_exactly(self):
        chunks = chunk_continuous(pages(90), chunk_size=30, overlap=0)
        self.assertEqual(len(chunks), 3)
        words = [w for c in chunks for w in c[0].split()]
        self.assertEqual(len(words), 90)
        self.assertEqual(len(set(words)), 90)

    def test_skips_pages_without_usable_text(self):
        # Page 2 is empty (a scanned page), so no passage may cite it.
        docs = [PageText(1, ' '.join(f'a{i}' for i in range(50))),
                PageText(2, ''),
                PageText(3, ' '.join(f'c{i}' for i in range(50)))]
        chunks = chunk_continuous(docs, chunk_size=200, overlap=0)
        self.assertEqual(len(chunks), 1)
        self.assertEqual((chunks[0][1], chunks[0][2]), (1, 3))

    def test_empty_document_yields_nothing(self):
        self.assertEqual(chunk_continuous([PageText(1, '')], 100, 0), [])

    def test_rejects_invalid_window(self):
        with self.assertRaises(ValueError):
            chunk_continuous(pages(10), chunk_size=0, overlap=0)
        with self.assertRaises(ValueError):
            chunk_continuous(pages(10), chunk_size=10, overlap=10)


class TestChunkPageBounded(unittest.TestCase):

    def test_never_crosses_page_boundaries(self):
        chunks = chunk_page_bounded(pages(30, 30), chunk_size=50, overlap=0)
        for text, ps, pe in chunks:
            self.assertEqual(ps, pe)
            page_tags = {w.split('w')[0] for w in text.split()}
            self.assertEqual(len(page_tags), 1)

    def test_produces_more_chunks_than_continuous_for_short_pages(self):
        # Four 30-word pages at chunk_size 100: continuous packs them into
        # 2 passages, page-bounded is forced into 4 short ones.
        p = pages(30, 30, 30, 30)
        self.assertLess(len(chunk_continuous(p, 100, 0)),
                        len(chunk_page_bounded(p, 100, 0)))


class TestChapterAttachment(unittest.TestCase):

    def setUp(self):
        self.passages = [
            Passage(f'd::p{i}', 'text', 'd', 'Doc', page_start=pg,
                    page_end=pg, position=i, token_count=2)
            for i, pg in enumerate([1, 5, 12, 99])
        ]

    def test_pages_outside_every_chapter_are_still_indexed(self):
        """
        REGRESSION TEST for the original indexing bug.

        Chapters cover pages 1-6 only. Page 12 and page 99 belong to no
        chapter. Under the old pipeline those pages were never chunked and so
        vanished from the corpus. Here they must survive as passages, merely
        without a chapter label.
        """
        chapters = [{'number': 1, 'title': 'One', 'page_start': 1, 'page_end': 3},
                    {'number': 2, 'title': 'Two', 'page_start': 4, 'page_end': 6}]
        result = attach_chapters(self.passages, chapters)

        self.assertEqual(len(result), 4)
        unlabelled = [p for p in result if p.chapter_number is None]
        self.assertEqual([p.page_start for p in unlabelled], [12, 99])

    def test_labels_pages_inside_a_chapter(self):
        chapters = [{'number': 7, 'title': 'Seven', 'page_start': 1, 'page_end': 3}]
        attach_chapters(self.passages, chapters)
        self.assertEqual(self.passages[0].chapter_number, 7)
        self.assertEqual(self.passages[0].chapter_title, 'Seven')

    def test_overlapping_ranges_resolve_to_narrowest(self):
        # Your Social Science book really does produce ranges like this.
        chapters = [{'number': 1, 'title': 'Broad', 'page_start': 1, 'page_end': 20},
                    {'number': 2, 'title': 'Narrow', 'page_start': 5, 'page_end': 5}]
        attach_chapters(self.passages, chapters)
        page5 = next(p for p in self.passages if p.page_start == 5)
        self.assertEqual(page5.chapter_title, 'Narrow')

    def test_no_chapters_leaves_passages_untouched(self):
        attach_chapters(self.passages, [])
        self.assertTrue(all(p.chapter_number is None for p in self.passages))


class TestParentAssignment(unittest.TestCase):

    def test_groups_consecutive_passages(self):
        ps = [Passage(f'd::p{i}', 't', 'd', 'D', 1, 1, i, 1) for i in range(7)]
        assign_parents(ps, group_size=3)
        groups = [p.parent_id for p in ps]
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[1], groups[2])
        self.assertNotEqual(groups[2], groups[3])

    def test_every_passage_receives_a_parent(self):
        ps = [Passage(f'd::p{i}', 't', 'd', 'D', 1, 1, i, 1) for i in range(5)]
        assign_parents(ps, group_size=2)
        self.assertTrue(all(p.parent_id for p in ps))


class TestPassageMetadata(unittest.TestCase):

    def test_page_label_single_page(self):
        p = Passage('d::p0', 't', 'd', 'Doc', 4, 4, 0, 1)
        self.assertEqual(p.page_label, 'p4')

    def test_page_label_span(self):
        p = Passage('d::p0', 't', 'd', 'Doc', 4, 6, 0, 1)
        self.assertEqual(p.page_label, 'p4-6')

    def test_citation_includes_chapter_when_known(self):
        p = Passage('d::p0', 't', 'd', 'Physics', 4, 4, 0, 1,
                    chapter_number=2, chapter_title='Electrostatics')
        self.assertIn('Electrostatics', p.citation)
        self.assertIn('p4', p.citation)

    def test_citation_omits_chapter_when_unknown(self):
        p = Passage('d::p0', 't', 'd', 'Physics', 4, 4, 0, 1)
        self.assertEqual(p.citation, 'Physics | p4')


class TestCorpusContainer(unittest.TestCase):

    def setUp(self):
        self.corpus = build_from_texts(
            {'docA': ' '.join(f'alpha{i}' for i in range(900)),
             'docB': ' '.join(f'beta{i}' for i in range(500))},
            chunk_size=400, overlap=80)

    def test_documents_view_maps_id_to_text(self):
        docs = self.corpus.documents()
        self.assertEqual(len(docs), len(self.corpus))
        self.assertTrue(all(isinstance(v, str) for v in docs.values()))

    def test_get_by_id(self):
        pid = self.corpus.passages[0].passage_id
        self.assertIs(self.corpus.get(pid), self.corpus.passages[0])

    def test_get_unknown_returns_none(self):
        self.assertIsNone(self.corpus.get('nope'))

    def test_by_document_filters(self):
        a = self.corpus.by_document('docA')
        self.assertTrue(a)
        self.assertTrue(all(p.doc_id == 'docA' for p in a))

    def test_parent_text_includes_sibling_passages(self):
        first = self.corpus.by_document('docA')[0]
        siblings = [p for p in self.corpus
                    if p.parent_id == first.parent_id]
        parent = self.corpus.parent_text(first.passage_id)
        self.assertGreaterEqual(len(parent), len(first.text))
        if len(siblings) > 1:
            self.assertGreater(len(parent), len(first.text))

    def test_parent_text_unknown_id_is_empty(self):
        self.assertEqual(self.corpus.parent_text('nope'), '')

    def test_stats_counts_passages(self):
        self.assertEqual(self.corpus.stats.n_passages, len(self.corpus))
        self.assertEqual(self.corpus.stats.n_documents, 2)

    def test_save_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'corpus.jsonl'
            self.corpus.save(path)
            loaded = Corpus.load(path)
        self.assertEqual(len(loaded), len(self.corpus))
        self.assertEqual(loaded.documents(), self.corpus.documents())
        self.assertEqual(loaded.stats.n_passages, self.corpus.stats.n_passages)
        self.assertEqual(loaded.passages[0].citation,
                         self.corpus.passages[0].citation)


class TestCorpusStatsReport(unittest.TestCase):

    def test_report_warns_about_empty_documents(self):
        stats = CorpusStats(n_documents=1, empty_documents=['scan.pdf (400 pages)'])
        report = stats.report()
        self.assertIn('WARNING', report)
        self.assertIn('scan.pdf', report)
        self.assertIn('OCR', report)

    def test_report_lists_duplicates(self):
        stats = CorpusStats(duplicate_documents=['b.pdf (identical to a.pdf)'])
        self.assertIn('identical to a.pdf', stats.report())

    def test_mean_passage_tokens_handles_empty_corpus(self):
        self.assertEqual(CorpusStats().mean_passage_tokens, 0.0)

    def test_build_from_texts_flags_empty_input(self):
        corpus = build_from_texts({'empty': ''})
        self.assertEqual(len(corpus), 0)
        self.assertIn('empty', corpus.stats.empty_documents)


if __name__ == '__main__':
    unittest.main(verbosity=2)
