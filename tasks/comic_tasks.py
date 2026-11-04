"""
Background tasks for the AI Comic Textbook.
Runs via Django Q2.

Task 1: process_textbook   – extract PDF, index RAG, create chapters & concepts
Task 2: generate_comic     – build storyboard, generate panels, compose comic
"""
import io
import os
import uuid
from pathlib import Path
from datetime import datetime

from django.conf import settings
from django.core.files.base import ContentFile


def process_textbook(textbook_id: str):
    """
    Full RAG pipeline for a textbook:
    1. Extract PDF text page-by-page
    2. Detect chapters
    3. Chunk & index into ChromaDB
    4. Run Curriculum Agent → create Chapter + Concept objects
    5. Generate quizzes for each concept

    Called by Django Q2 in a background worker.
    """
    import django
    django.setup()

    from core.models import Textbook, Chapter, Concept, Quiz
    from core import ir_service
    from rag.extractor import extract_text_by_pages, detect_chapters, get_chapter_text, get_total_pages
    from rag.chunker import chunk_pages
    from rag.embedder import index_chunks
    from agents.curriculum_agent import analyze_chapter
    from agents.quiz_agent import generate_quiz

    try:
        textbook = Textbook.objects.get(id=textbook_id)
        textbook.status = 'processing'
        textbook.save(update_fields=['status'])

        pdf_path = textbook.pdf_file.path

        # Step 1: Extract
        print(f"[Task] Extracting PDF: {pdf_path}")
        pages = extract_text_by_pages(pdf_path)
        total_pages = len(pages)
        textbook.total_pages = total_pages
        textbook.save(update_fields=['total_pages'])

        # Step 2: Detect chapters (3 strategies: regex → font-size → auto-split)
        chapters_data = detect_chapters(pages, pdf_path=pdf_path)
        print(f"[Task] Detected {len(chapters_data)} chapters")

        # Build the retrieval engine once and reuse it for every concept.
        # Calling ir_service.get_engine() per concept would be correct but
        # wasteful: each call re-fits BM25 and TF-IDF, and this task saves the
        # textbook several times, which invalidates the version-keyed cache.
        ir_engine = None
        try:
            ir_engine = ir_service.get_engine(textbook, rebuild=True)
            print(f"[Task] IR engine ready: {ir_engine.stats()}")
        except Exception as ir_error:
            # Quiz generation still works without retrieved context, just with
            # less grounding, so this must not abort processing.
            print(f"[Task] WARNING: IR engine unavailable ({ir_error}); "
                  f"quizzes will be generated without retrieved context")

        # Step 3: Process each chapter
        for ch_data in chapters_data:
            # Get or create Chapter
            chapter, _ = Chapter.objects.get_or_create(
                textbook=textbook,
                number=ch_data['number'],
                defaults={
                    'title': ch_data['title'],
                    'page_start': ch_data['page_start'],
                    'page_end': ch_data['page_end'],
                },
            )

            # Chunk this chapter's text
            chapter_text = get_chapter_text(pages, ch_data['page_start'], ch_data['page_end'])
            chunks = chunk_pages(pages, ch_data['page_start'], ch_data['page_end'])

            # Index into ChromaDB
            if chunks:
                index_chunks(str(textbook_id), ch_data['number'], chunks)
                print(f"[Task] Indexed {len(chunks)} chunks for Chapter {ch_data['number']}")

            # Curriculum Agent: extract concepts
            concepts_data = analyze_chapter(ch_data['title'], chapter_text[:3000])
            print(f"[Task] Chapter {ch_data['number']}: {len(concepts_data)} concepts found")

            for concept_data in concepts_data:
                concept, created = Concept.objects.get_or_create(
                    chapter=chapter,
                    title=concept_data['title'],
                    defaults={
                        'description': concept_data.get('description', ''),
                        'difficulty': concept_data.get('difficulty', 'intermediate'),
                        'order': concept_data.get('order', 1),
                        'page_references': [ch_data['page_start']],
                    },
                )

                if created:
                    # Retrieve context (hybrid + reranked) and generate quiz
                    ctx_text = ''
                    if ir_engine is not None:
                        try:
                            ctx_text = ir_engine.context_for(
                                concept_data['title'], max_words=400, top_k=3)
                        except Exception as ctx_error:
                            print(f"[Task] context retrieval failed for "
                                  f"'{concept_data['title']}': {ctx_error}")
                    quizzes = generate_quiz(
                        concept_title=concept_data['title'],
                        concept_description=concept_data.get('description', ''),
                        context_text=ctx_text,
                        level='intermediate',
                        num_questions=10,
                    )
                    for q in quizzes:
                        Quiz.objects.create(
                            concept=concept,
                            question=q['question'],
                            options=q['options'],
                            correct_index=q['correct_index'],
                            explanation=q.get('explanation', ''),
                        )

        textbook.status = 'ready'
        textbook.save(update_fields=['status'])
        print(f"[Task] Textbook {textbook_id} processing complete!")

    except Exception as e:
        print(f"[Task] ERROR processing textbook {textbook_id}: {e}")
        import traceback
        traceback.print_exc()
        try:
            textbook = Textbook.objects.get(id=textbook_id)
            textbook.status = 'error'
            textbook.error_message = str(e)
            textbook.save(update_fields=['status', 'error_message'])
        except Exception:
            pass


def generate_comic(comic_strip_id: str):
    """
    Comic generation pipeline:
    1. Retrieve RAG context for the concept
    2. Run Storyboard Agent → JSON panels
    3. Run Prompt Agent → Animagine prompts
    4. Call ComfyUI API for each panel
    5. Add overlays (titles, bubbles, captions) via Pillow
    6. Save panels to database

    Called by Django Q2 in a background worker.
    """
    import django
    django.setup()

    from core.models import ComicStrip, ComicPanel
    from core import ir_service
    from agents.storyboard_agent import generate_storyboard
    from agents.prompt_agent import build_all_prompts
    from comicgen.comfyui_client import ComfyUIClient
    from comicgen.workflow_builder import build_animagine_workflow, get_panel_dimensions
    from comicgen.panel_composer import add_panel_overlay

    try:
        comic_strip = ComicStrip.objects.select_related('concept__chapter__textbook').get(
            id=comic_strip_id
        )
        comic_strip.status = 'generating'
        comic_strip.save(update_fields=['status'])

        concept = comic_strip.concept
        chapter = concept.chapter
        textbook = chapter.textbook

        # Step 1: Retrieve context via the hybrid pipeline.
        #
        # Replaces the dense-only path this task used to take. Two differences
        # that reach the generated comic: retrieval is BM25 + dense fused and
        # then cross-encoder reranked, and each block carries its citation, so
        # a caption can be traced back to a page. Same engine the search page
        # uses, so the measured retrieval numbers describe this path too.
        print(f"[Task] Retrieving context for: {concept.title}")
        context_text = ''
        try:
            context_text = ir_service.retrieve_context(
                textbook, concept.title, max_words=600, top_k=5)
        except Exception as ir_error:
            print(f"[Task] WARNING: retrieval unavailable ({ir_error}); "
                  f"the storyboard agent will work from the concept "
                  f"description alone")

        # Step 2: Generate storyboard
        print(f"[Task] Generating storyboard for: {concept.title}")
        storyboard = generate_storyboard(
            concept_title=concept.title,
            concept_description=concept.description,
            context_text=context_text,
            level=comic_strip.level,
            art_style=comic_strip.art_style,
            min_panels=4,
            max_panels=8,
        )

        # Save storyboard to model
        comic_strip.storyboard = storyboard
        comic_strip.title = f"{concept.title} — {comic_strip.get_level_display()}"
        comic_strip.save(update_fields=['storyboard', 'title'])

        # Step 3: Generate prompts
        print(f"[Task] Building prompts for {len(storyboard)} panels")
        prompts = build_all_prompts(storyboard, art_style=comic_strip.art_style)

        # Step 4: Check ComfyUI availability
        client = ComfyUIClient()
        if not client.is_available():
            raise RuntimeError(
                "ComfyUI is not running. Please start it with run_nvidia_gpu.bat"
            )

        # Step 5: Generate each panel
        width, height = get_panel_dimensions(comic_strip.art_style, 'standard')
        media_panels_dir = Path(settings.MEDIA_ROOT) / 'panels'
        media_panels_dir.mkdir(parents=True, exist_ok=True)

        for i, (panel_data, prompt_data) in enumerate(zip(storyboard, prompts)):
            print(f"[Task] Generating panel {i+1}/{len(storyboard)}: {panel_data.get('title', '')}")

            # Enrich panel with LLM content if key_points missing
            from agents.content_agent import enrich_panel
            panel_data = enrich_panel(
                panel_data,
                concept_title=concept.title,
                level=comic_strip.level,
            )

            workflow = build_animagine_workflow(
                positive_prompt=prompt_data['positive'],
                negative_prompt=prompt_data['negative'],
                width=width,
                height=height,
                steps=32,
                cfg=7.0,
            )

            # Generate via ComfyUI
            images = client.generate_image(workflow, timeout=300)

            if not images:
                print(f"[Task] Warning: No image returned for panel {i+1}")
                continue

            raw_image_bytes = images[0]

            # Add overlay (title banner, speech bubble, caption)
            try:
                composed_bytes = add_panel_overlay(raw_image_bytes, panel_data)
            except Exception as overlay_err:
                print(f"[Task] Overlay error (using raw): {overlay_err}")
                composed_bytes = raw_image_bytes

            # Save panel to DB
            panel_filename = f"panel_{comic_strip_id}_{i+1}.png"
            panel, _ = ComicPanel.objects.get_or_create(
                comic_strip=comic_strip,
                panel_number=i + 1,
                defaults={
                    'scene_description': panel_data.get('scene', ''),
                    'dialogue': panel_data.get('dialogue', ''),
                    'caption': panel_data.get('caption', ''),
                    'concept_label': panel_data.get('concept_label', ''),
                    'prompt_used': prompt_data['positive'],
                },
            )
            panel.image.save(panel_filename, ContentFile(composed_bytes), save=True)
            panel.generated_at = datetime.now()
            panel.save(update_fields=['generated_at'])

            print(f"[Task] Panel {i+1} saved: {panel_filename}")

        # Mark complete
        comic_strip.status = 'ready'
        comic_strip.save(update_fields=['status'])
        print(f"[Task] Comic {comic_strip_id} generation complete!")

    except Exception as e:
        print(f"[Task] ERROR generating comic {comic_strip_id}: {e}")
        import traceback
        traceback.print_exc()
        try:
            comic_strip = ComicStrip.objects.get(id=comic_strip_id)
            comic_strip.status = 'error'
            comic_strip.error_message = str(e)
            comic_strip.save(update_fields=['status', 'error_message'])
        except Exception:
            pass
