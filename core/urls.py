from django.urls import path
from . import ir_views, views

urlpatterns = [
    # Information Retrieval
    path('textbook/<uuid:pk>/search/', ir_views.textbook_search, name='textbook_search'),
    path('textbook/<uuid:pk>/compare/', ir_views.search_compare, name='search_compare'),
    path('retrieval/', ir_views.ir_dashboard, name='ir_dashboard'),
    path('retrieval/judge/', ir_views.judge_passage, name='judge_passage'),

    # Auth
    path('register/', views.register_view, name='register'),

    # Dashboard
    path('', views.dashboard, name='dashboard'),

    # Textbooks
    path('upload/', views.textbook_upload, name='textbook_upload'),
    path('textbook/<uuid:pk>/', views.textbook_detail, name='textbook_detail'),
    path('textbook/<uuid:pk>/status/', views.textbook_status_api, name='textbook_status'),
    path('textbook/<uuid:pk>/reprocess/', views.reprocess_textbook, name='textbook_reprocess'),

    # Concepts
    path('concept/<uuid:pk>/', views.concept_detail, name='concept_detail'),
    path('concept/<uuid:concept_pk>/generate/', views.generate_comic_view, name='generate_comic'),

    # Comics
    path('comic/<uuid:pk>/', views.comic_view, name='comic_view'),
    path('comic/<uuid:pk>/status/', views.comic_status_api, name='comic_status'),
    path('comic/<uuid:pk>/rerender/', views.rerender_comic, name='comic_rerender'),

    # Quiz
    path('quiz/<uuid:quiz_pk>/submit/', views.submit_quiz, name='submit_quiz'),

    # Library & Progress
    path('library/', views.my_library, name='library'),
    path('progress/', views.my_progress, name='progress'),
]
