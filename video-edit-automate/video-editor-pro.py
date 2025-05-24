#!/usr/bin/env python3

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, colorchooser
import threading
import queue  # For communication between threads and GUI

# --- Standard library imports ---
import os
import subprocess
import traceback
from pathlib import Path
import shutil  # For copying files
from typing import Optional, Union, List, Tuple, Dict, Callable

# --- Third-party library imports script ---
try:
    from faster_whisper import WhisperModel
    from faster_whisper.transcribe import Word  # Using Word type from faster_whisper
except ImportError:
    messagebox.showerror("Dependency Error", "faster-whisper library not found. Please install it: pip install faster-whisper")
    exit(1)
try:
    from googletrans import Translator
except ImportError:
    messagebox.showerror("Dependency Error", "googletrans library not found. Please install it: pip install googletrans==4.0.0-rc1")
    exit(1)
try:
    import pysrt
except ImportError:
    messagebox.showerror("Dependency Error", "pysrt library not found. Please install it: pip install pysrt")
    exit(1)

# --- Constants ---
DEFAULT_WHISPER_MODEL = "base"
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
COMPUTE_TYPES_CPU = ["int8", "float32"]
COMPUTE_TYPES_CUDA = ["float16", "int8_float16", "int8"]
DEVICES = ["cpu", "cuda"]
APP_FONT_FAMILY = "Times New Roman"  # Global font family for the GUI

# Constants for Burn Tab (affect subtitles in video, not GUI font directly)
AVAILABLE_FONTS_FOR_BURN = [
    "Arial", "Arial Black", "Verdana", "Tahoma", "Times New Roman", "Georgia",
    "Courier New", "Helvetica", "Calibri", "Roboto", "Open Sans",
    "Lato", "Noto Sans", "DejaVu Sans", "Liberation Sans", "Impact"
]
DEFAULT_BURN_FONT_NAME_GUI = "Arial Black"
DEFAULT_BURN_FONT_SIZE_GUI = 15
DEFAULT_BURN_PRIMARY_COLOUR_GUI = "white"
DEFAULT_BURN_OUTLINE_COLOUR_GUI = "black"
DEFAULT_BURN_OUTLINE_WIDTH_GUI = 1
DEFAULT_BURN_SHADOW_DEPTH_GUI = 0
DEFAULT_BURN_BORDER_STYLE_DISPLAY_GUI = "1: Outline"
BORDER_STYLE_MAP_GUI: Dict[str, int] = {"1: Outline": 1, "3: Opaque Box": 3, "4: Individual Line Box": 4}
DEFAULT_BURN_BACK_COLOUR_GUI = "black"
DEFAULT_BURN_BACK_ALPHA_GUI = "80"

DEFAULT_MIN_SILENCE_DURATION_MS = 1000
DEFAULT_MAX_SUBTITLE_DURATION = None
DEFAULT_MAX_SUBTITLE_CHARS = None
DEFAULT_MAX_SUBTITLE_WORDS = None # <-- New constant

# --- Custom Dark Theme Colors ---
THEME_COLORS = {
    "root_bg": "#ffffff",
    "frame_bg": "#dcdcdc",
    "text_fg": "#2a2a2a",
    "label_fg": "#1e88e5",
    "disabled_fg": "#757575",
    "text_area_bg": "#ffffff",
    "text_area_fg": "#333333",
    "text_area_insert_bg": "#42a5f5",
    "text_area_select_bg": "#bdbdbd",
    "text_area_select_fg": "#000000",
    "entry_bg": "#ffffff",
    "entry_fg": "#333333",
    "entry_insert_bg": "#42a5f5",
    "entry_select_bg": "#bdbdbd",
    "entry_select_fg": "#000000",
    "button_bg": "#90a4ae",
    "button_fg": "#37474f",
    "button_active_bg": "#ffffff",
    "accent_button_bg": "#29b6f6",
    "accent_button_fg": "#0d47a1",
    "accent_button_active_bg": "#ffffff",
    "notebook_bg": "#f5f5f5",
    "notebook_tab_bg": "#bdbdbd",
    "notebook_tab_fg": "#616161",
    "notebook_tab_selected_bg": "#ffffff",
    "notebook_tab_selected_fg": "#1976d2",
    "labelframe_border": "#9e9e9e"
}
# --- UTILITY FUNCTIONS ---
def ensure_path_exists(path: Union[str, Path]) -> Path:
    """Ensures the parent directory of the given path exists."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def run_ffmpeg_process(command: List[str], log_callback: Callable[[str], None]) -> Tuple[bool, str, str]:
    """
    Runs an FFmpeg command using subprocess.
    Returns (success_status, stdout, stderr).
    """
    log_callback(f"Executing FFmpeg: {' '.join(command)}")
    stdout_full, stderr_full = "", ""
    try:
        env = os.environ.copy()
        env.pop("LD_LIBRARY_PATH", None)  # Avoids conflicts with system libs
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            universal_newlines=False,  # Read as bytes, then decode
            creationflags=creation_flags
        )
        # Stream output to avoid deadlocks with large outputs
        for line in iter(process.stdout.readline, b''):
            decoded_line = line.decode(errors='replace').strip()
            if decoded_line:
                log_callback(f"FFMPEG_STDOUT: {decoded_line}")
            stdout_full += decoded_line + "\n"
        for line in iter(process.stderr.readline, b''):
            decoded_line = line.decode(errors='replace').strip()
            if decoded_line:
                log_callback(f"FFMPEG_STDERR: {decoded_line}")
            stderr_full += decoded_line + "\n"

        process.stdout.close()
        process.stderr.close()
        process.wait()

        if process.returncode == 0:
            return True, stdout_full, stderr_full
        else:
            log_callback(f"FFmpeg error (code {process.returncode}). Check logs for stderr.")
            return False, stdout_full, stderr_full
    except FileNotFoundError:
        log_callback("Error: ffmpeg not found. Ensure FFmpeg is installed and in PATH.")
        return False, stdout_full, "ffmpeg not found"
    except Exception as e:
        log_callback(f"Unexpected error during FFmpeg execution: {e}\n{traceback.format_exc()}")
        return False, stdout_full, str(e)

# --- CORE PROCESSING FUNCTIONS ---
def format_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms == 1000:
        secs += 1
        ms = 0
    if secs == 60:
        mins += 1
        secs = 0
    if mins == 60:
        hrs += 1
        mins = 0
    return f"{hrs:02d}:{mins:02d}:{secs:02d}.{ms:03d}" if hrs > 0 else f"{mins:02d}:{secs:02d}.{ms:03d}"

def parse_timestamp_string(ts_str: str) -> float:
    ts_str = ts_str.strip()
    sec = 0.0
    parts = ts_str.split(':')
    try:
        if len(parts) == 3:  # HH:MM:SS.ms
            sec = float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2].replace(',', '.'))
        elif len(parts) == 2:  # MM:SS.ms
            sec = float(parts[0]) * 60 + float(parts[1].replace(',', '.'))
        elif len(parts) == 1:  # SS.ms or SS
            sec = float(parts[0].replace(',', '.'))
        else:
            raise ValueError("Invalid time format with colons")
    except ValueError as e:
        raise ValueError(f"Invalid time component in '{ts_str}': {e}") from e
    if sec < 0:
        raise ValueError(f"Timestamp cannot be negative: {ts_str}")
    return sec

def parse_cut_list(cuts_str: str) -> List[Tuple[float, float]]:
    cuts = []
    if not cuts_str:
        return cuts
    raw_parts = cuts_str.replace('\n', ',').split(',')
    for i, part_str in enumerate(filter(None, (p.strip() for p in raw_parts))):
        try:
            if '-' not in part_str:
                raise ValueError(f"Segment '{part_str}' needs a '-' separator.")
            start_str, end_str = part_str.split('-', 1)
            start_sec, end_sec = parse_timestamp_string(start_str), parse_timestamp_string(end_str)
            if start_sec >= end_sec:
                raise ValueError(f"End time must be after start time in '{part_str}'")
            cuts.append((start_sec, end_sec))
        except ValueError as e:
            raise ValueError(f"Invalid cut segment {i+1} ('{part_str}'): {e}") from e
    cuts.sort(key=lambda x: x[0])
    for i in range(len(cuts) - 1):
        if cuts[i][1] > cuts[i + 1][0]:
            raise ValueError(
                f"Overlapping cuts: {format_timestamp(cuts[i][0])}-{format_timestamp(cuts[i][1])} "
                f"and {format_timestamp(cuts[i+1][0])}-{format_timestamp(cuts[i+1][1])}"
            )
    return cuts

def cut_video_ffmpeg(input_video_path: Union[str, Path], cut_segments: List[Tuple[float, float]], output_video_path: Union[str, Path], log_callback=print) -> Optional[str]:
    input_file, output_file = Path(input_video_path), ensure_path_exists(output_video_path)
    if not input_file.exists():
        log_callback(f"Error: Input video file not found: {input_file}")
        return None
    if not cut_segments:
        log_callback("No cut segments provided. Skipping cutting.")
        return None

    log_callback(f"Cutting video '{input_file.name}' to create '{output_file.name}'...")
    select_filter_parts = [f"between(t,{s},{e})" for s, e in cut_segments]
    video_select_filter = "+".join(select_filter_parts)

    command = [
        'ffmpeg', '-y', '-i', str(input_file),
        '-vf', f"select='{video_select_filter}',setpts=N/FRAME_RATE/TB",
        '-af', f"aselect='{video_select_filter}',asetpts=N/FRAME_RATE/TB",
        '-c:v', 'libx264', '-crf', '23', '-preset', 'medium',
        '-c:a', 'aac', '-b:a', '192k', str(output_file)
    ]
    success, _, _ = run_ffmpeg_process(command, log_callback)
    if success:
        log_callback(f"Video cutting successful. Output: {output_file}")
        return str(output_file)
    log_callback(f"FFmpeg error during cutting for {input_file.name}.")
    return None

def translate_text_srt(text: str, dest_language: str, log_callback=print) -> Optional[str]:
    if not text.strip():
        return ""
    try:
        return Translator().translate(text, dest=dest_language).text
    except Exception as e:
        log_callback(f"Error translating '{text[:30]}...': {e}")
        return text  # Return original on error

def _finalize_sub_from_words(words_buffer: List[Word], sub_index: int) -> Optional[pysrt.SubRipItem]:
    if not words_buffer:
        return None
    text = " ".join(w.word.strip() for w in words_buffer if w.word and w.word.strip()).strip()
    if not text:
        return None

    start_sec, end_sec = words_buffer[0].start, words_buffer[-1].end
    if end_sec <= start_sec:
        end_sec = start_sec + 0.001

    start_time = pysrt.SubRipTime(seconds=start_sec)
    end_time = pysrt.SubRipTime(seconds=end_sec)

    if end_time.ordinal <= start_time.ordinal:
        end_time = pysrt.SubRipTime(seconds=start_time.seconds + start_time.milliseconds / 1000.0 + 0.001)

    return pysrt.SubRipItem(index=sub_index, start=start_time, end=end_time, text=text)

def transcribe_audio_to_srt(
    video_path: Union[str, Path], output_srt_path: Union[str, Path], model_name: str, language_code: Optional[str],
    device: str, compute_type: str, keep_audio_path: Union[str, Path], vad_filter: bool, min_silence_duration_ms: int,
    max_subtitle_duration: Optional[float], max_subtitle_chars: Optional[int], max_subtitle_words: Optional[int], # <-- Added max_subtitle_words
    log_callback=print
) -> bool:
    video_file, srt_file = Path(video_path), ensure_path_exists(output_srt_path)
    audio_file = ensure_path_exists(keep_audio_path)

    log_callback(f"\n--- Transcribing: {video_file.name} --- Model: {model_name}, Device: {device}, Compute: {compute_type}")
    if language_code:
        log_callback(f"Target Language Hint: {language_code}")

    use_word_level_segmentation = bool(max_subtitle_duration or max_subtitle_chars or max_subtitle_words) # <-- Updated

    log_callback(f"Transcription options: word_timestamps=True. VAD filter {'enabled' if vad_filter else 'disabled'}" +
                 (f", min_silence_duration_ms={min_silence_duration_ms}" if vad_filter else "") + ".")

    if use_word_level_segmentation:
        limit_details = []
        if max_subtitle_duration is not None: limit_details.append(f"Max Duration: {max_subtitle_duration}s")
        if max_subtitle_chars is not None: limit_details.append(f"Max Chars: {max_subtitle_chars}")
        if max_subtitle_words is not None: limit_details.append(f"Max Words: {max_subtitle_words}") # <-- Added
        log_callback(f"Applying custom word-level re-segmentation with limits: {', '.join(limit_details)}.")
    elif vad_filter:
        log_callback("Using segment boundaries from VAD (no custom re-segmentation limits).")
    else:
        log_callback("Using segment boundaries directly from faster-whisper (no VAD, no custom re-segmentation limits).")


    log_callback(f"Extracting audio to: {audio_file}...")
    ffmpeg_cmd_extract = [
        'ffmpeg', '-y', '-i', str(video_file),
        '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', str(audio_file)
    ]
    extract_success, _, _ = run_ffmpeg_process(ffmpeg_cmd_extract, log_callback)
    if not extract_success:
        log_callback(f"Audio extraction failed for {video_file.name}.")
        return False
    log_callback(f"Audio extraction complete: {audio_file}")

    try:
        log_callback("Loading Whisper model...")
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        log_callback("Whisper model loaded. Transcribing...")

        transcribe_options = {"beam_size": 5, "language": language_code, "word_timestamps": True}
        if vad_filter:
            transcribe_options["vad_filter"] = True
            transcribe_options["vad_parameters"] = {"min_silence_duration_ms": min_silence_duration_ms}

        segments_generator, info = model.transcribe(str(audio_file), **transcribe_options)
        log_callback(f"Transcription initiated. Language: {info.language} (Prob: {info.language_probability:.2f}), Duration: {info.duration:.2f}s")

        subs = pysrt.SubRipFile()
        sub_index = 0
        if use_word_level_segmentation:
            log_callback("Processing with custom word-level segmentation...")
            current_words_buffer: List[Word] = []
            current_text_buffer = ""
            for segment in segments_generator:
                words_to_process = segment.words if segment.words else []
                if not words_to_process and segment.text.strip(): # Handle segments without word timestamps but with text
                    sub_index += 1
                    start_t = pysrt.SubRipTime(seconds=segment.start)
                    end_t = pysrt.SubRipTime(seconds=max(segment.end, segment.start + 0.001))
                    subs.append(pysrt.SubRipItem(index=sub_index, start=start_t, end=end_t, text=segment.text.strip()))
                    continue

                for word_obj in words_to_process:
                    word_text = word_obj.word.strip()
                    if not word_text:
                        continue

                    prospective_text = (current_text_buffer + " " + word_text).strip() if current_words_buffer else word_text
                    prospective_word_count = len(current_words_buffer) + 1 # <-- Calculate prospective word count
                    start_time_buffer = current_words_buffer[0].start if current_words_buffer else word_obj.start
                    prospective_duration = word_obj.end - start_time_buffer

                    split_needed = False
                    if current_words_buffer: 
                        if max_subtitle_chars and len(prospective_text) > max_subtitle_chars and current_text_buffer:
                            log_callback(f"Splitting: char limit ({max_subtitle_chars}) met/exceeded. Current buffer text: '{current_text_buffer[:30]}...' adding '{word_text}'")
                            split_needed = True
                        if not split_needed and max_subtitle_duration and prospective_duration > max_subtitle_duration:
                            log_callback(f"Splitting: duration limit ({max_subtitle_duration:.2f}s) met/exceeded. Current buffer duration from {format_timestamp(start_time_buffer)} to {format_timestamp(word_obj.end)} ({prospective_duration:.2f}s)")
                            split_needed = True
                        if not split_needed and max_subtitle_words and prospective_word_count > max_subtitle_words and len(current_words_buffer) > 0: # <-- Word count check
                            log_callback(f"Splitting: word limit ({max_subtitle_words}) met/exceeded. Current buffer has {len(current_words_buffer)} words, adding one more makes {prospective_word_count}.")
                            split_needed = True

                    if split_needed:
                        finalized_sub = _finalize_sub_from_words(current_words_buffer, sub_index + 1)
                        if finalized_sub:
                            subs.append(finalized_sub)
                            sub_index += 1
                        current_words_buffer, current_text_buffer = [word_obj], word_text # Reset buffer with the current word
                    else:
                        current_words_buffer.append(word_obj)
                        current_text_buffer = prospective_text

            if current_words_buffer: # Finalize any remaining words in the buffer
                finalized_sub = _finalize_sub_from_words(current_words_buffer, sub_index + 1)
                if finalized_sub:
                    subs.append(finalized_sub)
                    sub_index += 1
        else:  # Default faster-whisper/VAD segmentation (no custom re-segmentation limits)
            log_callback("Using faster-whisper/VAD segmentations directly (no custom re-segmentation).")
            for segment in segments_generator:
                text = segment.text.strip()
                if not text:
                    continue
                sub_index += 1
                start_t = pysrt.SubRipTime(seconds=segment.start)
                end_t = pysrt.SubRipTime(seconds=max(segment.end, segment.start + 0.001)) # Ensure end time is after start
                subs.append(pysrt.SubRipItem(index=sub_index, start=start_t, end=end_t, text=text))

        if not subs:
            log_callback("Warning: No subtitles generated.")
        subs.save(str(srt_file), encoding='utf-8')
        log_callback(f"Transcription complete. Subtitles saved: {srt_file}")
        return True
    except Exception as e:
        log_callback(f"Error during transcription processing: {e}\n{traceback.format_exc()}")
        return False
    finally:
        if audio_file.exists():
            try:
                audio_file.unlink()
                log_callback(f"Temporary audio file deleted: {audio_file}")
            except OSError as e_del:
                log_callback(f"Warning: Could not delete temp audio {audio_file}: {e_del}")

def process_srt_file(input_srt_path: Union[str, Path], output_srt_path: Union[str, Path], target_lang: Optional[str], delay_seconds: float, stack_languages: bool, log_callback=print) -> bool:
    in_file, out_file = Path(input_srt_path), ensure_path_exists(output_srt_path)
    if not in_file.exists() or in_file.stat().st_size == 0:
        log_callback(f"Input SRT '{in_file}' missing/empty. Cannot process.")
        return False

    if not target_lang and delay_seconds == 0.0:
        try:
            shutil.copyfile(in_file, out_file)
            log_callback(f"No SRT processing required. Copied '{in_file}' to '{out_file}'.")
            return True
        except Exception as e_copy:
            log_callback(f"Error copying SRT: {e_copy}. Proceeding to process (if applicable) or will fail at save.")

    try:
        subs_orig = pysrt.open(str(in_file), encoding='utf-8')
    except Exception as e:
        log_callback(f"Error opening SRT '{in_file}': {e}")
        return False
    if not subs_orig:
        log_callback(f"No subs in '{in_file}'. Saving empty to '{out_file}'.")
        out_file.write_text("", encoding='utf-8')
        return True

    final_subs = pysrt.SubRipFile()
    if delay_seconds != 0.0:
        log_callback(f"Applying delay: {delay_seconds:.3f}s.")
        subs_orig.shift(seconds=delay_seconds)
    if target_lang:
        log_callback(f"Translating to '{target_lang}'" + (" and stacking." if stack_languages else "."))

    for i, item_orig in enumerate(subs_orig):
        new_item = pysrt.SubRipItem(index=i + 1, start=item_orig.start, end=item_orig.end, text=item_orig.text)
        processed_text = original_text = new_item.text.strip()

        if target_lang:
            translated_text = translate_text_srt(original_text, target_lang, log_callback)
            if translated_text and translated_text.strip().lower() != original_text.lower():
                processed_text = f"{original_text}\n{translated_text.strip()}" if stack_languages else translated_text.strip()
            elif not translated_text or not translated_text.strip():
                log_callback(f"Warning: Translation for sub #{new_item.index} empty/failed. Keeping original.")

        new_item.text = processed_text
        if new_item.start.ordinal < 0:
            new_item.start = pysrt.SubRipTime(0)
        if new_item.end.ordinal <= new_item.start.ordinal:
            new_item.end = pysrt.SubRipTime(seconds=new_item.start.seconds + new_item.start.milliseconds / 1000.0 + 0.001)
        final_subs.append(new_item)

    try:
        if not final_subs:
            log_callback("Warning: Processed subs empty. Saving empty SRT.")
        final_subs.save(str(out_file), encoding='utf-8')
        log_callback(f"SRT processing complete. Saved: {out_file}")
        return True
    except Exception as e_save:
        log_callback(f"Error saving processed SRT '{out_file}': {e_save}")
        return False

def get_ffmpeg_color_hex(color_name_or_hex: str, alpha_hex: str = "00") -> str:
    color_input = color_name_or_hex.lower().strip()
    alpha = alpha_hex.strip().upper().zfill(2)
    if not (len(alpha) == 2 and all(c in "0123456789ABCDEF" for c in alpha)):
        alpha = "00"

    named_colors = {
        "black": "000000", "white": "FFFFFF", "red": "FF0000", "green": "00FF00",
        "blue": "0000FF", "yellow": "FFFF00", "cyan": "00FFFF", "magenta": "FF00FF"
    }

    hex_rgb = named_colors.get(color_input)
    if not hex_rgb:
        temp_color = color_input.lstrip('#')
        if len(temp_color) == 6 and all(c in "0123456789abcdef" for c in temp_color):
            hex_rgb = temp_color

    if hex_rgb:
        try:
            r_h, g_h, b_h = hex_rgb[0:2], hex_rgb[2:4], hex_rgb[4:6]
            int(r_h, 16); int(g_h, 16); int(b_h, 16)
            return f"&H{alpha}{b_h.upper()}{g_h.upper()}{r_h.upper()}"
        except ValueError:
            pass

    fallback_bgr = "000000" if color_input == "black" else "FFFFFF"
    r_h, g_h, b_h = fallback_bgr[0:2], fallback_bgr[2:4], fallback_bgr[4:6]
    return f"&H{alpha}{b_h.upper()}{g_h.upper()}{r_h.upper()}"

def burn_subtitles_to_video_ffmpeg(
    video_path: Union[str, Path], srt_file_path: Union[str, Path], output_video_path: Union[str, Path],
    font_name: str, font_size: int, primary_color_name: str,
    outline_color_name: str, outline_width: int, shadow_depth: int,
    border_style_value: int, back_color_name: str, back_alpha_hex: str,
    force_style: Optional[str], log_callback=print
) -> Optional[str]:
    vid_f, srt_f = Path(video_path), Path(srt_file_path)
    out_f = ensure_path_exists(output_video_path)

    if not vid_f.exists():
        log_callback(f"Error: Input video not found: {vid_f}")
        return None

    if not srt_f.exists() or srt_f.stat().st_size == 0:
        log_callback(f"SRT file '{srt_f.name}' missing/empty. Copying video instead.")
        cmd_copy = ['ffmpeg', '-y', '-i', str(vid_f), '-c', 'copy', str(out_f)]
        copy_success, _, _ = run_ffmpeg_process(cmd_copy, log_callback)
        return str(out_f) if copy_success else None

    log_callback(f"Burning subs from '{srt_f.name}' to '{vid_f.name}', output: '{out_f.name}'")
    log_callback(f"Font: '{font_name}', Size: {font_size}, Primary Color: {primary_color_name}")

    if force_style:
        style_str = force_style.strip()
        log_callback(f"Using provided FFmpeg force_style string: '{style_str}'")
    else:
        styles_parts = [
            f"Fontname='{font_name}'", f"Fontsize={font_size}",
            f"PrimaryColour={get_ffmpeg_color_hex(primary_color_name, '00')}",
            "MarginV=25", "Alignment=2" # Alignment=2 is bottom-center for ASS
        ]
        if border_style_value == 1:  # Outline
            log_callback(f"Using Outline Style. Outline: {outline_color_name}, Width: {outline_width}, Shadow: {shadow_depth}")
            styles_parts.extend([
                f"BorderStyle=1",
                f"OutlineColour={get_ffmpeg_color_hex(outline_color_name, '00')}",
                f"Outline={outline_width}", f"Shadow={shadow_depth}"
            ])
        elif border_style_value == 3:  # Opaque Box
            log_callback(f"Using Opaque Box Style. Background: {back_color_name}, Alpha: {back_alpha_hex}")
            styles_parts.extend([
                "BorderStyle=3",
                f"BackColour={get_ffmpeg_color_hex(back_color_name, back_alpha_hex)}",
                "Outline=0", "Shadow=0"
            ])
        style_str = ",".join(styles_parts)
    log_callback(f"Constructed FFmpeg style string: '{style_str}'")

    srt_path_filt = Path(srt_f).as_posix()
    if os.name == 'nt': 
        srt_path_filt = srt_path_filt.replace(':', '\\:') # Escape drive letter colon

    cmd_burn = [
        'ffmpeg', '-y', '-i', str(vid_f),
        '-vf', f"subtitles='{srt_path_filt}':force_style='{style_str}'",
        '-c:v', 'libx264', '-crf', '23', '-preset', 'medium',
        '-c:a', 'aac', '-b:a', '192k', str(out_f)
    ]
    burn_success, _, _ = run_ffmpeg_process(cmd_burn, log_callback)
    if burn_success:
        log_callback(f"FFmpeg subtitle burn successful: {out_f}")
        return str(out_f)
    log_callback(f"FFmpeg error during burn for {vid_f.name}.")
    return None

# --- GUI Application ---
class VideoEditorApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("Video Editor Pro ✨")
        self.root.geometry("1000x1200+100+50") 
        self.log_queue = queue.Queue()

        self.menu_bar = tk.Menu(self.root)
        self.settings_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.settings_menu.add_command(label="Change GUI Background Color", command=self._change_gui_background_color)
        self.menu_bar.add_cascade(label="Settings", menu=self.settings_menu)
        self.root.config(menu=self.menu_bar)

        self._apply_theme_and_fonts()

        self.notebook = ttk.Notebook(self.root)
        self.tab_transcribe = ttk.Frame(self.notebook, padding="10")
        self.tab_burn = ttk.Frame(self.notebook, padding="10")
        self.tab_cut = ttk.Frame(self.notebook, padding="10")

        self.notebook.add(self.tab_transcribe, text='Transcribe & Process SRT')
        self.notebook.add(self.tab_burn, text='Burn SRT to Video')
        self.notebook.add(self.tab_cut, text='Cut Video')

        self.notebook.pack(fill='x', expand=False, padx=10, pady=10)

        self._create_transcribe_tab()
        self._create_burn_tab()
        self._create_cut_tab()

        log_frame = ttk.LabelFrame(self.root, text="Log / Status", padding="5")
        log_frame.pack(fill='both', expand=True, padx=10, pady=(0, 10))

        self.log_text = scrolledtext.ScrolledText(
            log_frame, height=10, wrap=tk.WORD, state=tk.DISABLED
        )
        self.log_text.pack(fill='both', expand=True)

        self._apply_theme_and_fonts()
        self.root.after(100, self._process_log_queue)

    def _change_gui_background_color(self):
        new_root_bg_tuple = colorchooser.askcolor(
            title="Choose Main GUI Background Color",
            initialcolor=THEME_COLORS["root_bg"]
        )
        if new_root_bg_tuple and new_root_bg_tuple[1]:
            new_root_bg_hex = new_root_bg_tuple[1]
            THEME_COLORS["root_bg"] = new_root_bg_hex

            new_frame_bg_tuple = colorchooser.askcolor(
                title="Choose Frame/Element Background Color",
                initialcolor=THEME_COLORS["frame_bg"]
            )
            if new_frame_bg_tuple and new_frame_bg_tuple[1]:
                 THEME_COLORS["frame_bg"] = new_frame_bg_tuple[1]
            else:
                 self._add_log("Frame background color change cancelled. Using root color for frames.")
                 THEME_COLORS["frame_bg"] = new_root_bg_hex

            self._add_log(f"GUI background colors updated. Root: {THEME_COLORS['root_bg']}, Frame: {THEME_COLORS['frame_bg']}")
            self._apply_theme_and_fonts()
        else:
            self._add_log("GUI background color change cancelled.")

    def _apply_theme_and_fonts(self):
        s = ttk.Style(self.root)
        try:
            s.theme_use('clam')
        except tk.TclError:
            self._add_log_safe("Warning: 'clam' theme not available. Using default.")

        font_size = 16
        font_bold = (APP_FONT_FAMILY, font_size, "bold")
        font_normal = (APP_FONT_FAMILY, font_size)
        font_log = (APP_FONT_FAMILY, font_size - 1)

        self.root.configure(background=THEME_COLORS["root_bg"])
        s.configure('.', background=THEME_COLORS["frame_bg"], foreground=THEME_COLORS["text_fg"], font=font_normal)
        s.configure('TFrame', background=THEME_COLORS["frame_bg"])
        s.configure('TLabel', background=THEME_COLORS["frame_bg"], foreground=THEME_COLORS["label_fg"], font=font_normal)
        s.configure('TLabelFrame', background=THEME_COLORS["frame_bg"], bordercolor=THEME_COLORS["labelframe_border"])
        s.configure('TLabelFrame.Label', background=THEME_COLORS["frame_bg"], foreground=THEME_COLORS["label_fg"], font=font_bold)

        s.configure('TButton', background=THEME_COLORS["button_bg"], foreground=THEME_COLORS["button_fg"], font=font_normal, padding=(10, 5))
        s.map('TButton', background=[('active', THEME_COLORS["button_active_bg"]), ('pressed', THEME_COLORS["button_active_bg"])])
        s.configure('Accent.TButton', background=THEME_COLORS["accent_button_bg"], foreground=THEME_COLORS["accent_button_fg"], font=font_bold, padding=(12, 5))
        s.map('Accent.TButton', background=[('active', THEME_COLORS["accent_button_active_bg"]), ('pressed', THEME_COLORS["accent_button_active_bg"])])

        s.configure('TEntry', fieldbackground=THEME_COLORS["entry_bg"], foreground=THEME_COLORS["entry_fg"], insertcolor=THEME_COLORS["entry_insert_bg"], selectbackground=THEME_COLORS["entry_select_bg"], selectforeground=THEME_COLORS["entry_select_fg"], font=font_normal, padding=(5, 8))
        s.configure('TCombobox', fieldbackground=THEME_COLORS["entry_bg"], foreground=THEME_COLORS["entry_fg"], selectbackground=THEME_COLORS["entry_bg"], selectforeground=THEME_COLORS["entry_fg"], arrowcolor=THEME_COLORS["text_fg"], insertcolor=THEME_COLORS["entry_insert_bg"], font=font_normal,padding=(5, 8))
        self.root.option_add('*TCombobox*Listbox.font', font_normal)
        self.root.option_add('*TCombobox*Listbox.background', THEME_COLORS["entry_bg"])
        self.root.option_add('*TCombobox*Listbox.foreground', THEME_COLORS["entry_fg"])
        self.root.option_add('*TCombobox*Listbox.selectBackground', THEME_COLORS["accent_button_bg"])
        self.root.option_add('*TCombobox*Listbox.selectForeground', THEME_COLORS["accent_button_fg"])

        s.configure('TCheckbutton', background=THEME_COLORS["frame_bg"], foreground=THEME_COLORS["label_fg"], font=font_normal, indicatorcolor=THEME_COLORS["entry_bg"])
        s.map('TCheckbutton', indicatorcolor=[('selected', THEME_COLORS["accent_button_bg"]), ('active', THEME_COLORS["entry_bg"])], foreground=[('disabled', THEME_COLORS["disabled_fg"])])

        s.configure('TNotebook', background=THEME_COLORS["notebook_bg"])
        s.configure('TNotebook.Tab', background=THEME_COLORS["notebook_tab_bg"], foreground=THEME_COLORS["notebook_tab_fg"], padding=[10, 4], font=font_normal)
        s.map('TNotebook.Tab', background=[('selected', THEME_COLORS["notebook_tab_selected_bg"])], foreground=[('selected', THEME_COLORS["notebook_tab_selected_fg"])], expand=[("selected", [1, 1, 1, 0])])

        log_text_config = {
            "background": THEME_COLORS["text_area_bg"], "foreground": THEME_COLORS["text_area_fg"],
            "insertbackground": THEME_COLORS["text_area_insert_bg"],
            "selectbackground": THEME_COLORS["text_area_select_bg"],
            "selectforeground": THEME_COLORS["text_area_select_fg"], "font": font_log
        }
        if hasattr(self, 'log_text') and self.log_text:
            self.log_text.config(**log_text_config)

        if hasattr(self, 'cut_segments_text') and self.cut_segments_text:
            self.cut_segments_text.config(
                background=THEME_COLORS["text_area_bg"], foreground=THEME_COLORS["text_area_fg"],
                insertbackground=THEME_COLORS["text_area_insert_bg"],
                selectbackground=THEME_COLORS["text_area_select_bg"],
                selectforeground=THEME_COLORS["text_area_select_fg"],
                font=font_normal
            )

    def _add_log_safe(self, message: str):
        if hasattr(self, 'log_queue') and self.log_queue:
            self.log_queue.put(str(message))
        else:
            print(f"LOG (pre-queue): {message}")

    def _add_log(self, message: str):
        self.log_queue.put(str(message))

    def _process_log_queue(self):
        try:
            while True:
                message = self.log_queue.get_nowait()
                self.log_text.config(state=tk.NORMAL)
                self.log_text.insert(tk.END, message + "\n")
                self.log_text.see(tk.END)
                self.log_text.config(state=tk.DISABLED)
        except queue.Empty:
            pass
        self.root.after(100, self._process_log_queue)

    def _set_button_loading(self, button: Optional[ttk.Button], text="Processing..."):
        if button:
            button.config(state=tk.DISABLED, text=text)

    def _reset_button_state(self, button: Optional[ttk.Button], original_text: str):
        if button:
            self.root.after(0, lambda b=button, t=original_text: b.config(state=tk.NORMAL, text=t))

    def _execute_gui_task_wrapper(self, button_widget: ttk.Button, original_text: str, loading_text: str,
                                 task_callable: Callable, task_args: tuple, operation_name: str):
        self._add_log(f"{operation_name} process initiated...")
        self._set_button_loading(button_widget, loading_text)

        def worker():
            try:
                task_callable(*task_args)
            except Exception as e:
                tb = traceback.format_exc()
                self._add_log(f"CRITICAL UNHANDLED ERROR during {operation_name}: {e}\n{tb}")
                self.root.after(0, lambda: messagebox.showerror("Critical Error", f"An unexpected critical error occurred during {operation_name}: {e}"))
            finally:
                self._add_log(f"--- {operation_name} task finished ---")
                self._reset_button_state(button_widget, original_text)
        threading.Thread(target=worker, daemon=True).start()

    def _create_input_row(self, parent_frame, label_text: str, var_type: str = "string", default_value=None, options_list: Optional[List[str]] = None, label_width: int = 22) -> Tuple[tk.Variable, tk.Widget]:
        frame = ttk.Frame(parent_frame)
        frame.pack(fill='x', pady=2)
        ttk.Label(frame, text=label_text, width=label_width).pack(side=tk.LEFT, padx=(0, 5))

        tk_var: tk.Variable
        entry_width, button_width = 48, 10

        if var_type.startswith("file_") or var_type.startswith("srt_"):
            default_val_str = str(default_value or "")
            tk_var = tk.StringVar(value=default_val_str)
            input_widget = ttk.Entry(frame, textvariable=tk_var, width=entry_width - button_width - 1)

            file_types_map = {
                "file_open": [("Video/Audio files", "*.mp4 *.mkv *.avi *.mov *.webm *.flv *.mp3 *.wav *.aac *.ogg *.flac"), ("All files", "*.*")],
                "file_save_srt": [("SRT files", "*.srt")],
                "file_save_video": [("Video files", "*.mp4 *.mkv"), ("All files", "*.*")],
                "srt_open": [("SRT files", "*.srt"), ("All files", "*.*")]
            }
            browse_action = self._browse_save_file if "save" in var_type else self._browse_file
            def_ext = ""
            if var_type == "file_save_srt":
                def_ext = ".srt"
            elif var_type == "file_save_video":
                def_ext = ".mp4"

            browse_lambda = (lambda v=tk_var, ft=file_types_map[var_type], de=def_ext: browse_action(v, ft, de)) if "save" in var_type \
                            else (lambda v=tk_var, ft=file_types_map[var_type]: browse_action(v, ft))

            input_widget.pack(side=tk.LEFT, expand=True, fill='x')
            ttk.Button(frame, text="Browse...", command=browse_lambda, width=button_width).pack(side=tk.LEFT, padx=(5, 0))
            return tk_var, input_widget

        elif var_type == "boolean":
            tk_var = tk.BooleanVar(value=bool(default_value))
            input_widget = ttk.Checkbutton(frame, variable=tk_var)
            input_widget.pack(side=tk.LEFT)
        elif var_type == "combobox":
            tk_var = tk.StringVar(value=str(default_value or ""))
            input_widget = ttk.Combobox(frame, textvariable=tk_var, values=options_list or [], state="readonly", width=entry_width - 2)
            input_widget.pack(side=tk.LEFT, expand=True, fill='x')
        else:  # string, float, int
            tk_var = tk.StringVar(value=str(default_value or ""))
            input_widget = ttk.Entry(frame, textvariable=tk_var, width=entry_width)
            input_widget.pack(side=tk.LEFT, expand=True, fill='x')
        return tk_var, input_widget

    def _browse_file(self, var_to_set: tk.StringVar, filetypes: List[Tuple[str, str]]):
        filepath = filedialog.askopenfilename(filetypes=filetypes)
        if filepath:
            var_to_set.set(filepath)

    def _browse_save_file(self, var_to_set: tk.StringVar, filetypes: List[Tuple[str, str]], defaultextension: str):
        filepath = filedialog.asksaveasfilename(defaultextension=defaultextension, filetypes=filetypes)
        if filepath:
            var_to_set.set(filepath)

    def _update_compute_types(self, *_):
        selected_device = self.ts_device_var.get()
        types = COMPUTE_TYPES_CUDA if selected_device == "cuda" else COMPUTE_TYPES_CPU
        self.ts_compute_type_combo['values'] = types
        if self.ts_compute_type_var.get() not in types:
            self.ts_compute_type_var.set(types[0])
        if selected_device == "cuda":
            self._add_log_safe("CUDA selected! Faster Whisper will use GPU for transcription. Nice for speed!")
        else:
            self._add_log_safe("CPU selected for Whisper. For large files, GPU (cuda) would be much faster if available.")


    def _create_transcribe_tab(self):
        parent = self.tab_transcribe
        self.ts_video_in_var, _ = self._create_input_row(parent, "Input Video/Audio:", "file_open")
        self.ts_srt_out_var, _ = self._create_input_row(parent, "Output SRT Path:", "file_save_srt", default_value="output.srt")
        self.ts_audio_out_var, _ = self._create_input_row(parent, "Temp Audio Path:", "file_save_video", default_value="temp_audio_for_transcription.wav")

        model_settings_frame = ttk.LabelFrame(parent, text="Transcription Model Settings", padding="5")
        model_settings_frame.pack(fill='x', pady=5, expand=True)
        self.ts_model_var, self.ts_model_combo = self._create_input_row(model_settings_frame, "Whisper Model:", "combobox", default_value=DEFAULT_WHISPER_MODEL, options_list=WHISPER_MODELS)
        self.ts_device_var, self.ts_device_combo = self._create_input_row(model_settings_frame, "Device:", "combobox", default_value=DEVICES[0], options_list=DEVICES)
        self.ts_device_var.trace_add("write", self._update_compute_types)
        self.ts_compute_type_var, self.ts_compute_type_combo = self._create_input_row(model_settings_frame, "Compute Type:", "combobox")
        self._update_compute_types()
        self.ts_lang_var, _ = self._create_input_row(model_settings_frame, "Language Hint (e.g. en):", "string", default_value="")

        seg_settings_frame = ttk.LabelFrame(parent, text="Segmentation & VAD Settings", padding="5")
        seg_settings_frame.pack(fill='x', pady=5, expand=True)
        self.ts_vad_filter_var, _ = self._create_input_row(seg_settings_frame, "Enable VAD Filter:", "boolean", default_value=True)
        self.ts_min_silence_var, _ = self._create_input_row(seg_settings_frame, "Min Silence (ms):", "string", default_value=str(DEFAULT_MIN_SILENCE_DURATION_MS))
        self.ts_max_duration_str_var, _ = self._create_input_row(seg_settings_frame, "Max Sub Duration (s, opt):", "string", default_value=str(DEFAULT_MAX_SUBTITLE_DURATION or ""))
        self.ts_max_chars_str_var, _ = self._create_input_row(seg_settings_frame, "Max Sub Chars (opt):", "string", default_value=str(DEFAULT_MAX_SUBTITLE_CHARS or ""))
        self.ts_max_words_str_var, _ = self._create_input_row(seg_settings_frame, "Max Sub Words (opt):", "string", default_value=str(DEFAULT_MAX_SUBTITLE_WORDS or "")) # <-- New field

        post_proc_frame = ttk.LabelFrame(parent, text="SRT Post-Processing", padding="5")
        post_proc_frame.pack(fill='x', pady=5, expand=True)
        self.ts_target_lang_var, _ = self._create_input_row(post_proc_frame, "Translate to (e.g. es):", "string", default_value="")
        self.ts_stack_langs_var, _ = self._create_input_row(post_proc_frame, "Stack Translations:", "boolean", default_value=False)
        self.ts_delay_srt_str_var, _ = self._create_input_row(post_proc_frame, "Delay SRT (s, e.g. -1.5):", "string", default_value="0.0")

        self.transcribe_button = ttk.Button(parent, text="Start Transcription & Processing", command=self._run_transcribe_op, style="Accent.TButton")
        self.transcribe_button.pack(pady=(15, 10), ipady=5)

    def _run_transcribe_op(self):
        video_in = self.ts_video_in_var.get()
        srt_out = self.ts_srt_out_var.get()
        audio_out = self.ts_audio_out_var.get()
        if not all([video_in, srt_out, audio_out]):
            messagebox.showerror("Input Error", "All file/path inputs are required.")
            return
        try:
            min_sil = int(self.ts_min_silence_var.get() or DEFAULT_MIN_SILENCE_DURATION_MS)
            max_dur_s = float(d) if (d := self.ts_max_duration_str_var.get().strip()) else None
            max_chars = int(c) if (c := self.ts_max_chars_str_var.get().strip()) else None
            max_words = int(w) if (w := self.ts_max_words_str_var.get().strip()) else None # <-- Parse new field
            delay_s = float(dl) if (dl := self.ts_delay_srt_str_var.get().strip()) else 0.0
            if min_sil < 0:
                min_sil = DEFAULT_MIN_SILENCE_DURATION_MS
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid numeric setting: {e}")
            return

        args = (
            video_in, srt_out, audio_out, self.ts_model_var.get(), self.ts_lang_var.get().strip() or None,
            self.ts_device_var.get(), self.ts_compute_type_var.get(), self.ts_vad_filter_var.get(),
            min_sil, max_dur_s, max_chars, max_words, # <-- Pass new arg
            self.ts_target_lang_var.get().strip() or None,
            self.ts_stack_langs_var.get(), delay_s
        )
        self._execute_gui_task_wrapper(self.transcribe_button, "Start Transcription & Processing", "Transcribing...",
                                     self._transcribe_task, args, "Transcription & Processing")

    def _transcribe_task(self, video_in, srt_out, audio_out_temp, model, lang_hint, device, compute,
                         vad, min_silence, max_dur, max_chars, max_words, # <-- Receive new arg
                         target_lang, stack_langs, delay_srt):
        try:
            ensure_path_exists(srt_out)
            ensure_path_exists(audio_out_temp)
            trans_ok = transcribe_audio_to_srt(
                video_in, srt_out, model, lang_hint, device, compute,
                audio_out_temp, vad, min_silence, max_dur, max_chars, max_words, # <-- Pass new arg
                self._add_log
            )
            if not trans_ok:
                self._add_log("Transcription failed.")
                self.root.after(0, lambda: messagebox.showerror("Error", "Transcription failed. Check logs."))
                return

            self._add_log(f"Transcription successful: {srt_out}")
            final_srt_path = srt_out
            if target_lang or delay_srt != 0.0:
                proc_srt_path = Path(srt_out).with_stem(Path(srt_out).stem + "_processed")
                self._add_log(f"Post-processing SRT. Output to: {proc_srt_path}")
                if process_srt_file(srt_out, str(proc_srt_path), target_lang, delay_srt, stack_langs, self._add_log):
                    self._add_log(f"SRT post-processing successful: {proc_srt_path}")
                    final_srt_path = str(proc_srt_path)
                else:
                    self._add_log("SRT post-processing failed. Using raw SRT for final message.")

            msg = f"Transcription complete!\nRaw SRT: {srt_out}" + (f"\nProcessed SRT: {final_srt_path}" if final_srt_path != srt_out else "")
            self.root.after(0, lambda m=msg: messagebox.showinfo("Success!", m))

        except Exception as e_task:
            tb = traceback.format_exc()
            self._add_log(f"ERROR in transcription task steps: {e_task}\n{tb}")
            self.root.after(0, lambda err=str(e_task): messagebox.showerror("Task Error", f"Transcription task error: {err}. Check logs."))

    
    def _on_burn_border_style_change(self, *_):
        selected_style_display = self.burn_border_style_var.get()
        selected_style_value = BORDER_STYLE_MAP_GUI.get(selected_style_display)
        is_box_style = (selected_style_value == 3 or selected_style_value == 4)
        is_outline_style = (selected_style_value == 1)

        widget_config_map = {
            'burn_back_colour_entry': is_box_style,
            'burn_back_alpha_entry': is_box_style,
            'burn_outline_colour_entry': is_outline_style,
            'burn_outline_width_entry': is_outline_style,
            'burn_shadow_depth_entry': is_outline_style,
        }

        for widget_name_str, should_be_normal in widget_config_map.items():
            widget = getattr(self, widget_name_str, None)
            if widget and isinstance(widget, (ttk.Entry, ttk.Combobox, tk.Widget)):
                new_state = tk.NORMAL if should_be_normal else tk.DISABLED
                if isinstance(widget, ttk.Combobox) and should_be_normal:
                    new_state = 'readonly'
                widget.config(state=new_state)
            elif not widget:
                self._add_log_safe(f"Warning: Widget '{widget_name_str}' not found during style change.")

    def _create_burn_tab(self):
        parent = self.tab_burn
        lw = 25  # label_width
        self.burn_video_in_var, _ = self._create_input_row(parent, "Input Video:", "file_open", label_width=lw)
        self.burn_srt_in_var, _ = self._create_input_row(parent, "Input SRT File:", "srt_open", label_width=lw)
        self.burn_video_out_var, _ = self._create_input_row(parent, "Output Video Path:", "file_save_video", default_value="output_with_subs.mp4", label_width=lw)

        style_frame = ttk.LabelFrame(parent, text="Subtitle Style Options", padding="10")
        style_frame.pack(fill='x', pady=(10, 5), expand=True)

        self.burn_font_name_var, self.burn_font_name_combo = self._create_input_row(style_frame, "Font Name:", "combobox", default_value=DEFAULT_BURN_FONT_NAME_GUI, options_list=AVAILABLE_FONTS_FOR_BURN, label_width=lw)
        self.burn_font_size_var, _ = self._create_input_row(style_frame, "Font Size:", "string", default_value=str(DEFAULT_BURN_FONT_SIZE_GUI), label_width=lw)
        self.burn_primary_colour_var, _ = self._create_input_row(style_frame, "Primary Colour:", "string", default_value=DEFAULT_BURN_PRIMARY_COLOUR_GUI, label_width=lw)
        self.burn_border_style_var, self.burn_border_style_combo = self._create_input_row(
            style_frame,
            "Border Style:",
            "combobox",
            default_value=DEFAULT_BURN_BORDER_STYLE_DISPLAY_GUI, 
            options_list=list(BORDER_STYLE_MAP_GUI.keys()),
            label_width=lw
        )
        self.burn_border_style_var.trace_add("write", self._on_burn_border_style_change)

        self.burn_outline_colour_var, self.burn_outline_colour_entry = self._create_input_row(style_frame, "Outline Colour:", "string", default_value=DEFAULT_BURN_OUTLINE_COLOUR_GUI, label_width=lw)
        self.burn_outline_width_var, self.burn_outline_width_entry = self._create_input_row(style_frame, "Outline Width (px):", "string", default_value=str(DEFAULT_BURN_OUTLINE_WIDTH_GUI), label_width=lw)
        self.burn_shadow_depth_var, self.burn_shadow_depth_entry = self._create_input_row(style_frame, "Shadow Depth (px):", "string", default_value=str(DEFAULT_BURN_SHADOW_DEPTH_GUI), label_width=lw)

        self.burn_back_colour_var, self.burn_back_colour_entry = self._create_input_row(style_frame, "Background Colour:", "string", default_value=DEFAULT_BURN_BACK_COLOUR_GUI, label_width=lw)
        self.burn_back_alpha_var, self.burn_back_alpha_entry = self._create_input_row(style_frame, "Background Alpha (00-FF):", "string", default_value=DEFAULT_BURN_BACK_ALPHA_GUI, label_width=lw)
        
        self._on_burn_border_style_change() 

        self.burn_force_style_var, _ = self._create_input_row(style_frame, "FFmpeg Style Override:", "string", default_value="", label_width=lw)
        example_style_text = "(Example override: Fontsize=20,PrimaryColour=&H00FFFFFF&,Alignment=8)"
        style_frame.winfo_reqwidth()
        force_style_help_label = ttk.Label(style_frame, text=example_style_text, wraplength=230, font = (APP_FONT_FAMILY, 13))
        force_style_help_label.pack(fill='x', padx=(lw * 8 + 25 if parent.winfo_exists() and lw*8+25 < parent.winfo_width() else 5), pady=(0,5))


        self.burn_button = ttk.Button(parent, text="Burn Subtitles to Video", command=self._run_burn_op, style="Accent.TButton")
        self.burn_button.pack(pady=(15, 10), ipady=5)

    def _run_burn_op(self):
        vid_in = self.burn_video_in_var.get()
        srt_in = self.burn_srt_in_var.get()
        vid_out = self.burn_video_out_var.get()
        if not all([vid_in, srt_in, vid_out]):
            messagebox.showerror("Input Error", "Input Video and Output Video paths are required. SRT can be optional (video will be copied).")
            return
        try:
            font_size = int(self.burn_font_size_var.get() or DEFAULT_BURN_FONT_SIZE_GUI)
            outline_w = int(self.burn_outline_width_var.get() or DEFAULT_BURN_OUTLINE_WIDTH_GUI)
            shadow_d = int(self.burn_shadow_depth_var.get() or DEFAULT_BURN_SHADOW_DEPTH_GUI)
            back_alpha = self.burn_back_alpha_var.get().strip().upper()
            if not (len(back_alpha) == 2 and all(c in "0123456789ABCDEFabcdef" for c in back_alpha)):
                self._add_log(f"Invalid Background Alpha '{back_alpha}', using default '{DEFAULT_BURN_BACK_ALPHA_GUI}'.")
                back_alpha = DEFAULT_BURN_BACK_ALPHA_GUI
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid numeric style setting: {e}")
            return

        args = (
            vid_in, srt_in, vid_out, self.burn_font_name_var.get(), font_size, self.burn_primary_colour_var.get(),
            self.burn_outline_colour_var.get(), outline_w, shadow_d,
            BORDER_STYLE_MAP_GUI.get(self.burn_border_style_var.get(), 1),
            self.burn_back_colour_var.get(), back_alpha, self.burn_force_style_var.get().strip() or None
        )
        self._execute_gui_task_wrapper(self.burn_button, "Burn Subtitles to Video", "Burning...",
                                     self._burn_task, args, "Subtitle Burning")

    def _burn_task(self, vid_in, srt_in, vid_out, font_n, font_s, prim_c, out_c, out_w, shad_d, border_val, back_c, back_a, force_s):
        try:
            ensure_path_exists(vid_out)
            res_path = burn_subtitles_to_video_ffmpeg(
                vid_in, srt_in, vid_out, font_n, font_s, prim_c, out_c, out_w, shad_d,
                border_val, back_c, back_a, force_s, self._add_log
            )
            if res_path:
                msg = f"Subtitle burn successful!\nOutput: {res_path}"
                self._add_log(msg.replace('\n', ' '))
                self.root.after(0, lambda m=msg: messagebox.showinfo("Success!", m))
            else:
                self._add_log("Subtitle burning failed.")
                self.root.after(0, lambda: messagebox.showerror("Error", "Subtitle burning failed. Check logs."))
        except Exception as e_task:
            tb = traceback.format_exc()
            self._add_log(f"ERROR in burn task steps: {e_task}\n{tb}")
            self.root.after(0, lambda err=str(e_task): messagebox.showerror("Task Error", f"Burn task error: {err}. Check logs."))

    def _create_cut_tab(self):
        parent = self.tab_cut
        lw = 25 # label_width
        self.cut_video_in_var, _ = self._create_input_row(parent, "Input Video:", "file_open", label_width=lw)

        cuts_frame = ttk.Frame(parent)
        cuts_frame.pack(fill='x', pady=2)
        ttk.Label(cuts_frame, text="Cut Segments (e.g., 0:10-0:45, 1m30.5s-1:55):", width=lw).pack(side=tk.LEFT, anchor='nw', padx=(0, 5))
        self.cut_segments_text = tk.Text(cuts_frame, height=5, relief=tk.SOLID, borderwidth=1, undo=True, wrap=tk.WORD)
        self.cut_segments_text.pack(side=tk.LEFT, fill='x', expand=True, pady=(0, 3))

        self.cut_video_out_var, _ = self._create_input_row(parent, "Output Cut Video Path:", "file_save_video", default_value="cut_video.mp4", label_width=lw)
        self.cut_button = ttk.Button(parent, text="Cut Video", command=self._run_cut_op, style="Accent.TButton")
        self.cut_button.pack(pady=(15, 10), ipady=5)

    def _run_cut_op(self):
        vid_in = self.cut_video_in_var.get()
        cuts_str = self.cut_segments_text.get("1.0", tk.END).strip()
        vid_out = self.cut_video_out_var.get()
        if not all([vid_in, cuts_str, vid_out]):
            messagebox.showerror("Input Error", "All inputs for cutting are required.")
            return
        try:
            cut_list = parse_cut_list(cuts_str)
            if not cut_list:
                messagebox.showinfo("Info", "No valid cut segments provided.")
                return
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid cut segments: {e}")
            return

        self._execute_gui_task_wrapper(self.cut_button, "Cut Video", "Cutting...",
                                     self._cut_task, (vid_in, cut_list, vid_out), "Video Cutting")

    def _cut_task(self, vid_in, segments, vid_out):
        try:
            ensure_path_exists(vid_out)
            res_path = cut_video_ffmpeg(vid_in, segments, vid_out, self._add_log)
            if res_path:
                msg = f"Video cutting successful!\nOutput: {res_path}"
                self._add_log(msg.replace('\n', ' '))
                self.root.after(0, lambda p=msg: messagebox.showinfo("Success!", p))
            else:
                self._add_log("Video cutting failed.")
                self.root.after(0, lambda: messagebox.showerror("Error", "Video cutting failed. Check logs."))
        except Exception as e_task:
            tb = traceback.format_exc()
            self._add_log(f"ERROR in cut task steps: {e_task}\n{tb}")
            self.root.after(0, lambda err=str(e_task): messagebox.showerror("Task Error", f"Cut task error: {err}. Check logs."))

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoEditorApp(root)
    root.mainloop()