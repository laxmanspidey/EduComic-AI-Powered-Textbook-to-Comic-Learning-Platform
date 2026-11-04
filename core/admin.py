from django.contrib import admin
from .models import Textbook, Chapter, Concept, ComicStrip, ComicPanel, Quiz, StudentProgress


@admin.register(Textbook)
class TextbookAdmin(admin.ModelAdmin):
    list_display = ['title', 'user', 'status', 'total_pages', 'created_at']
    list_filter = ['status']
    search_fields = ['title', 'user__username']


@admin.register(Chapter)
class ChapterAdmin(admin.ModelAdmin):
    list_display = ['title', 'number', 'textbook', 'page_start', 'page_end']
    list_filter = ['textbook']


@admin.register(Concept)
class ConceptAdmin(admin.ModelAdmin):
    list_display = ['title', 'chapter', 'difficulty', 'order']
    list_filter = ['difficulty', 'chapter__textbook']
    search_fields = ['title']


@admin.register(ComicStrip)
class ComicStripAdmin(admin.ModelAdmin):
    list_display = ['concept', 'user', 'level', 'art_style', 'status', 'created_at']
    list_filter = ['status', 'level', 'art_style']


@admin.register(ComicPanel)
class ComicPanelAdmin(admin.ModelAdmin):
    list_display = ['panel_number', 'comic_strip', 'generated_at']


@admin.register(Quiz)
class QuizAdmin(admin.ModelAdmin):
    list_display = ['question', 'concept', 'correct_index']
    search_fields = ['question']


@admin.register(StudentProgress)
class StudentProgressAdmin(admin.ModelAdmin):
    list_display = ['user', 'concept', 'viewed', 'quiz_attempts', 'quiz_correct']

# Retrieval models live in their own module; import it so autodiscovery
# registers them alongside the comic application models.
from . import ir_admin  # noqa: F401,E402
