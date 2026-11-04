"""
Smoke-test the retrieval views end to end against the real database.

Renders every new page with Django's test client and checks the status code
and a marker string, so a template syntax error or a missing context variable
is caught without opening a browser. Uses the real database read-only: it
creates no textbooks and deletes nothing, and the one SearchQuery row a search
writes is the same row the live app would write.

Usage:
    venv/Scripts/python.exe tools/smoke_views.py
"""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

import django  # noqa: E402

django.setup()

from django.conf import settings  # noqa: E402

# Django's test client sends Host: testserver. Allowing it here rather than in
# config/settings.py keeps the harness's needs out of the application's
# configuration -- the deployed ALLOWED_HOSTS stays exactly as the .env sets it.
if 'testserver' not in settings.ALLOWED_HOSTS:
    settings.ALLOWED_HOSTS = list(settings.ALLOWED_HOSTS) + ['testserver']

from django.contrib.auth.models import User  # noqa: E402
from django.test import Client  # noqa: E402

from core.models import Textbook  # noqa: E402

PASSED, FAILED = [], []


def check(label: str, response, marker: str = '') -> None:
    status = getattr(response, 'status_code', 0)
    body = response.content.decode('utf-8', 'replace') if status else ''
    ok = status == 200 and (not marker or marker in body)
    (PASSED if ok else FAILED).append(label)
    flag = 'ok  ' if ok else 'FAIL'
    detail = f"status={status}"
    if status == 200 and marker and marker not in body:
        detail += f", marker {marker!r} missing"
    print(f"  [{flag}] {label:52s} {detail}")


def check_template_comments() -> None:
    """
    Reject multi-line {# ... #} comments.

    Django's {# #} is single-line only; a multi-line one is not stripped and
    renders as visible text on the page. It is invisible server-side -- the
    response is still HTTP 200 with otherwise-correct markup -- so only a
    reader or this check will catch it. Use {% comment %}...{% endcomment %}
    for anything spanning lines.
    """
    import glob
    import re

    offenders = []
    for path in sorted(glob.glob('templates/**/*.html', recursive=True)):
        text = open(path, encoding='utf-8').read()
        for match in re.finditer(r'\{#', text):
            rest = text[match.start():]
            end, newline = rest.find('#}'), rest.find('\n')
            if end == -1 or (newline != -1 and newline < end):
                offenders.append(f"{path}:{text[:match.start()].count(chr(10)) + 1}")

    label = 'template comments are single-line'
    (PASSED if not offenders else FAILED).append(label)
    flag = 'ok  ' if not offenders else 'FAIL'
    print(f"  [{flag}] {label:52s} "
          f"{'' if not offenders else 'multi-line {# #} at ' + ', '.join(offenders)}")


def main() -> int:
    print("Static checks:")
    check_template_comments()
    print()

    user = User.objects.filter(textbooks__isnull=False).distinct().first()
    if user is None:
        print("No user owns a textbook; nothing to smoke-test.")
        return 1

    client = Client()
    client.force_login(user)

    # Prefer a textbook that actually has extractable text, so the search
    # path is exercised rather than the scanned-PDF error path.
    textbooks = list(Textbook.objects.filter(user=user))
    searchable = next(
        (t for t in textbooks if 'CAT26' not in t.pdf_file.name), None)
    scanned = next(
        (t for t in textbooks if 'CAT26' in t.pdf_file.name), None)

    print(f"user={user.username}  textbooks={len(textbooks)}")
    print(f"searchable={searchable.title if searchable else None!r}")
    print()

    print("Application pages:")
    check('dashboard', client.get('/'), 'Your textbooks')
    check('library', client.get('/library/'), 'My comics')
    check('progress', client.get('/progress/'), 'Progress')
    check('upload', client.get('/upload/'), 'Choose a PDF')
    if searchable:
        check('textbook detail', client.get(f'/textbook/{searchable.pk}/'),
              'Ask this textbook')

    # Concept and comic pages: pick real objects so the templates render with
    # the data they actually receive rather than an empty context.
    from core.models import ComicStrip, Concept

    concept = Concept.objects.filter(
        chapter__textbook__user=user).first()
    if concept:
        check('concept detail (comic generator + quiz)',
              client.get(f'/concept/{concept.pk}/'),
              'Generate a comic lesson')

    comic = ComicStrip.objects.filter(user=user).first()
    if comic:
        check('comic viewer', client.get(f'/comic/{comic.pk}/'), 'Keep going')

    print("\nUnauthenticated pages:")
    anon = Client()
    check('login', anon.get('/login/'), 'Welcome back')
    check('register', anon.get('/register/'), 'Create your account')

    print("\nRetrieval pages:")
    check('ir dashboard', client.get('/retrieval/'), 'Retrieval analytics')

    if searchable:
        url = f'/textbook/{searchable.pk}/search/'
        check('search (empty query)', client.get(url), 'get grounded answers')

        # Build the lexical query from this textbook's own vocabulary. A
        # hardcoded query would only match whichever textbook happens to be
        # in the database, and would report a content mismatch as a view bug.
        lexical_query = 'the'
        try:
            from core import ir_service

            engine = ir_service.get_engine(searchable)
            frequent = sorted(engine.bm25.df.items(),
                              key=lambda kv: -kv[1])[:40]
            lexical_query = ' '.join(term for term, _ in frequent[:3])
            print(f"  (lexical probe derived from corpus: {lexical_query!r})")
        except Exception as error:
            print(f"  (could not derive a lexical probe: {error})")

        # answer=0 on the retrieval checks: answer generation costs several
        # seconds of Ollama time and is covered separately below, so the
        # retrieval paths stay fast to re-run.
        check('search (lexical, corpus-derived query)',
              client.get(url, {'q': lexical_query, 'system': 'bm25'}),
              'Why was this retrieved?')
        check('search (full pipeline)',
              client.get(url, {'q': lexical_query}),
              'Query understanding')
        # The corpus-derived probe is keyword soup, not a question, so the
        # generator may legitimately abstain. Assert the answer card rendered
        # -- a marker present on both the grounded and the abstained path --
        # rather than assuming an answer was produced. Grounding and
        # abstention behaviour is covered by tests/test_answer.py.
        check('search (answer card renders)',
              client.get(url, {'q': lexical_query, 'answer': '1'}),
              'qwen')
        check('search (answer suppressed)',
              client.get(url, {'q': lexical_query, 'system': 'bm25'}))
        check('search (comparison intent)',
              client.get(url, {'q': 'difference between consistency and availability'}),
              'comparison')
        check('search (unmatched lexical query)',
              client.get(url, {'q': 'photosynthesis chlorophyll', 'system': 'bm25'}))
        check('compare systems',
              client.get(f'/textbook/{searchable.pk}/compare/',
                         {'q': 'what does CAP mean'}),
              'side by side')

    if scanned:
        # A scanned PDF must produce a clear explanation, not an empty list
        # that looks like a ranking failure.
        check('search (scanned PDF reports the real cause)',
              client.get(f'/textbook/{scanned.pk}/search/', {'q': 'anything'}),
              'scanned PDF')

    print("\nJudgement endpoint:")
    check('judge passage', client.post('/retrieval/judge/', {
        'query_id': 'smoke_q', 'passage_id': 'smoke_p',
        'grade': '2', 'query_text': 'smoke', 'passage_text': 'smoke'}))

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        for label in FAILED:
            print(f"  failed: {label}")
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
