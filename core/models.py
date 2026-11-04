from django.db import models
from django.contrib.auth.models import User
import uuid


class Textbook(models.Model):
    """Represents an uploaded PDF textbook."""
    STATUS_CHOICES = [
        ('uploaded', 'Uploaded'),
        ('processing', 'Processing'),
        ('ready', 'Ready'),
        ('error', 'Error'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='textbooks')
    title = models.CharField(max_length=255)
    subject = models.CharField(max_length=100, blank=True)
    grade_level = models.CharField(max_length=50, blank=True)
    pdf_file = models.FileField(upload_to='textbooks/')
    thumbnail = models.ImageField(upload_to='textbook_thumbs/', null=True, blank=True)
    total_pages = models.IntegerField(default=0)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='uploaded')
    error_message = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return self.title

    @property
    def chapter_count(self):
        return self.chapters.count()

    @property
    def concept_count(self):
        return Concept.objects.filter(chapter__textbook=self).count()


class Chapter(models.Model):
    """A chapter within a textbook."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    textbook = models.ForeignKey(Textbook, on_delete=models.CASCADE, related_name='chapters')
    title = models.CharField(max_length=255)
    number = models.IntegerField(default=1)
    page_start = models.IntegerField(default=0)
    page_end = models.IntegerField(default=0)
    summary = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['number']

    def __str__(self):
        return f"Chapter {self.number}: {self.title}"


class Concept(models.Model):
    """A key concept within a chapter."""
    DIFFICULTY_CHOICES = [
        ('beginner', 'Beginner'),
        ('intermediate', 'Intermediate'),
        ('advanced', 'Advanced'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    chapter = models.ForeignKey(Chapter, on_delete=models.CASCADE, related_name='concepts')
    title = models.CharField(max_length=255)
    description = models.TextField()
    page_references = models.JSONField(default=list)   # list of page numbers
    difficulty = models.CharField(max_length=20, choices=DIFFICULTY_CHOICES, default='intermediate')
    order = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['order']

    def __str__(self):
        return self.title


class ComicStrip(models.Model):
    """A generated comic for a concept."""
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('generating', 'Generating'),
        ('ready', 'Ready'),
        ('error', 'Error'),
    ]

    ART_STYLE_CHOICES = [
        ('comic', 'Comic'),
        ('manga', 'Manga'),
        ('cartoon', 'Cartoon'),
        ('realistic', 'Realistic Illustration'),
    ]

    LEVEL_CHOICES = [
        ('beginner', 'Beginner'),
        ('intermediate', 'Intermediate'),
        ('revision', 'Exam Revision'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    concept = models.ForeignKey(Concept, on_delete=models.CASCADE, related_name='comic_strips')
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='comic_strips')
    title = models.CharField(max_length=255, blank=True)
    level = models.CharField(max_length=20, choices=LEVEL_CHOICES, default='intermediate')
    art_style = models.CharField(max_length=20, choices=ART_STYLE_CHOICES, default='comic')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    storyboard = models.JSONField(default=list)   # list of scene dicts
    layout_image = models.ImageField(upload_to='comics/', null=True, blank=True)
    error_message = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Comic: {self.concept.title} ({self.level})"

    @property
    def panel_count(self):
        return self.panels.count()


class ComicPanel(models.Model):
    """A single panel in a comic strip."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    comic_strip = models.ForeignKey(ComicStrip, on_delete=models.CASCADE, related_name='panels')
    panel_number = models.IntegerField()
    scene_description = models.TextField()
    dialogue = models.TextField(blank=True)
    caption = models.TextField(blank=True)
    concept_label = models.CharField(max_length=255, blank=True)
    image = models.ImageField(upload_to='panels/', null=True, blank=True)
    prompt_used = models.TextField(blank=True)
    comfyui_prompt_id = models.CharField(max_length=100, blank=True)
    generated_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['panel_number']

    def __str__(self):
        return f"Panel {self.panel_number} of {self.comic_strip}"


class Quiz(models.Model):
    """A quiz associated with a concept."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    concept = models.ForeignKey(Concept, on_delete=models.CASCADE, related_name='quizzes')
    question = models.TextField()
    options = models.JSONField(default=list)        # list of 4 strings
    correct_index = models.IntegerField(default=0)  # 0-indexed
    explanation = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"Quiz: {self.question[:60]}..."


class StudentProgress(models.Model):
    """Track student progress per concept."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='progress')
    concept = models.ForeignKey(Concept, on_delete=models.CASCADE, related_name='progress')
    viewed = models.BooleanField(default=False)
    quiz_attempts = models.IntegerField(default=0)
    quiz_correct = models.IntegerField(default=0)
    last_accessed = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ['user', 'concept']

    def __str__(self):
        return f"{self.user.username} - {self.concept.title}"

    @property
    def quiz_score_pct(self):
        if self.quiz_attempts == 0:
            return 0
        return int((self.quiz_correct / self.quiz_attempts) * 100)


# ─── Information Retrieval ────────────────────────────────────────────────────
#
# Only live state lives in the database. Evaluation artefacts -- the test
# collection, qrels and experiment runs -- are files under ir/data/, because
# they must be reproducible and version-controllable, and because the
# evaluation runs as a plain script without Django loaded.


class SearchSession(models.Model):
    """
    A conversation with one textbook.

    Exists so multi-turn query rewriting has history to resolve against: when
    a student asks "why does it increase?", the referent is in the previous
    turn, not in the query.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE,
                             related_name='search_sessions')
    textbook = models.ForeignKey(Textbook, on_delete=models.CASCADE,
                                 related_name='search_sessions')
    created_at = models.DateTimeField(auto_now_add=True)
    last_active = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-last_active']

    def __str__(self):
        return f"Search session: {self.textbook.title} ({self.user.username})"

    def history(self, limit: int = 4):
        """
        Recent turns, oldest first, in the form the rewriter expects.

        Returns `ir.query.rewrite.ConversationTurn` objects. Capped because
        referents almost always point at the immediately preceding turn, and a
        longer window invites the rewriter to drag in stale topics.
        """
        from ir.query.rewrite import ConversationTurn

        turns = list(self.queries.order_by('-created_at')[:limit])
        turns.reverse()
        return [ConversationTurn(query=t.query_text, answer=t.top_snippet)
                for t in turns]


class SearchQuery(models.Model):
    """
    One executed search, with its retrieval trace.

    The trace is stored whole as JSON rather than normalised into columns. It
    is write-once diagnostic data whose shape changes as pipeline stages are
    added, and it is only ever read back for the query it belongs to, so a
    schema migration per new stage would be pure cost.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(SearchSession, on_delete=models.CASCADE,
                                related_name='queries', null=True, blank=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE,
                             related_name='search_queries')
    textbook = models.ForeignKey(Textbook, on_delete=models.CASCADE,
                                 related_name='search_queries')

    query_text = models.TextField()
    rewritten_text = models.TextField(blank=True)
    intent = models.CharField(max_length=32, blank=True)
    system = models.CharField(max_length=32, default='full')

    expansion_terms = models.JSONField(default=list)
    trace = models.JSONField(default=dict)

    result_count = models.IntegerField(default=0)
    top_snippet = models.TextField(blank=True)
    total_ms = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = 'search queries'

    def __str__(self):
        return f"{self.query_text[:60]} ({self.result_count} results)"

    @property
    def was_rewritten(self) -> bool:
        return bool(self.rewritten_text
                    and self.rewritten_text.strip() != self.query_text.strip())

    @property
    def stage_timings(self) -> dict:
        return self.trace.get('timings_ms', {}) if self.trace else {}


class RelevanceJudgement(models.Model):
    """
    A human relevance judgement, for verifying the synthetic test collection.

    The generated collection assumes the passage a question was written from
    is the relevant one. That assumption needs checking on a subset, and the
    agreement rate between human and synthetic judgements is itself a
    reportable result. Exported to qrels.jsonl to be merged with the generated
    judgements, which keep the higher grade on conflict.
    """
    GRADE_CHOICES = [
        (0, 'Not relevant'),
        (1, 'Partially relevant'),
        (2, 'Highly relevant'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE,
                             related_name='judgements')
    query_id = models.CharField(max_length=64)
    query_text = models.TextField()
    passage_id = models.CharField(max_length=128)
    passage_text = models.TextField(blank=True)
    grade = models.IntegerField(choices=GRADE_CHOICES, default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['query_id', 'passage_id']
        unique_together = ['user', 'query_id', 'passage_id']

    def __str__(self):
        return f"{self.query_id} / {self.passage_id}: {self.grade}"
