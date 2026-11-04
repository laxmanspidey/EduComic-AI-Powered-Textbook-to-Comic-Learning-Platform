"""
Admin registrations for the retrieval models.

Kept separate from `core/admin.py` for the same reason as the views: the comic
application and the retrieval layer share models but nothing else. Imported
from `core/admin.py` so Django's autodiscovery picks it up.
"""
from __future__ import annotations

from django.contrib import admin

from .models import RelevanceJudgement, SearchQuery, SearchSession


@admin.register(SearchSession)
class SearchSessionAdmin(admin.ModelAdmin):
    list_display = ['textbook', 'user', 'query_count', 'last_active']
    list_filter = ['textbook']
    readonly_fields = ['created_at', 'last_active']

    @admin.display(description='queries')
    def query_count(self, obj):
        return obj.queries.count()


@admin.register(SearchQuery)
class SearchQueryAdmin(admin.ModelAdmin):
    list_display = ['query_text', 'intent', 'system', 'result_count',
                     'rewritten', 'ms', 'created_at']
    list_filter = ['intent', 'system', 'textbook']
    search_fields = ['query_text', 'rewritten_text']
    readonly_fields = ['created_at', 'trace']

    @admin.display(boolean=True, description='rewritten')
    def rewritten(self, obj):
        return obj.was_rewritten

    @admin.display(description='ms')
    def ms(self, obj):
        return f"{obj.total_ms:.0f}"


@admin.register(RelevanceJudgement)
class RelevanceJudgementAdmin(admin.ModelAdmin):
    list_display = ['query_id', 'passage_id', 'grade', 'user', 'created_at']
    list_filter = ['grade', 'user']
    search_fields = ['query_id', 'query_text', 'passage_id']
    readonly_fields = ['created_at']
    actions = ['export_as_qrels']

    @admin.action(description='Export selected judgements to qrels.jsonl')
    def export_as_qrels(self, request, queryset):
        """
        Append the selected human judgements to the test collection.

        Merged rather than overwritten, with the higher grade winning per
        (query, passage), so a human judgement can upgrade a synthetic one
        without discarding the generated set.
        """
        from ir.eval.collection import Judgement, TestCollection

        collection = TestCollection.load()
        if not len(collection):
            self.message_user(
                request,
                "No test collection found. Generate one first with "
                "tools/run_experiments.py --generate.",
                level='warning')
            return

        collection.merge_judgements([
            Judgement(query_id=j.query_id, passage_id=j.passage_id,
                      grade=j.grade, source='human')
            for j in queryset
        ])
        collection.save()
        self.message_user(
            request,
            f"Merged {queryset.count()} human judgement(s) into "
            f"ir/data/collection/qrels.jsonl. Re-run the experiment to "
            f"evaluate against them.")
