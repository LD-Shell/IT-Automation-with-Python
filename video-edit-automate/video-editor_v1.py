#!/usr/bin/env python3

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue # For communication between threads and GUI

# --- Standard library imports---
import os
import subprocess
import traceback
from pathlib import Path
from typing import Optional, Union, List, Tuple

# --- Third-party library imports---
try:
    from faster_whisper import WhisperModel
    from faster_whisper.transcribe import Word
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

# --- Constants---
DEFAULT_WHISPER_MODEL = "base"
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
COMPUTE_TYPES_CPU = ["int8", "float32"]
COMPUTE_TYPES_CUDA = ["float16", "int8_float16", "int8"] # Order for display
DEVICES = ["cpu", "cuda"]

DEFAULT_FONT_NAME = "Arial Black"
DEFAULT_FONT_SIZE = 22
DEFAULT_FONT_COLOR = "white"
DEFAULT_SUBTITLE_BACKGROUND_COLOR = "black"
DEFAULT_SUBTITLE_BACKGROUND_ALPHA = "A0"
DEFAULT_MIN_SILENCE_DURATION_MS = 1000
DEFAULT_MAX_SUBTITLE_DURATION = None
DEFAULT_MAX_SUBTITLE_CHARS = None

# --- CORE PROCESSING FUNCTIONS---

# --- Timestamp Parsing and Formatting ---
def format_timestamp(seconds: float) -> str:
    """Converts seconds to HH:MM:SS.ms or MM:SS.ms string."""
    if seconds < 0: seconds = 0.0 # Ensure non-negative for formatting
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000)) # Round ms to nearest integer
    
    if ms == 1000: # Handle rounding up to the next second
        secs += 1
        ms = 0
        if secs == 60: # Handle seconds rolling over
            mins +=1
            secs = 0
            if mins == 60: # Handle minutes rolling over
                hrs +=1
                mins = 0
    
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}.{ms:03d}"
    else:
        return f"{mins:02d}:{secs:02d}.{ms:03d}"

def parse_timestamp_string(ts_str: str) -> float:
    ts_str = ts_str.strip()
    if ':' in ts_str:
        parts = ts_str.split(':')
        sec = 0.0
        try:
            if len(parts) == 3: sec = float(parts[0])*3600 + float(parts[1])*60 + float(parts[2].replace(',','.'))
            elif len(parts) == 2: sec = float(parts[0])*60 + float(parts[1].replace(',','.'))
            else: raise ValueError("Invalid time format with colons")
        except ValueError as e: raise ValueError(f"Invalid time component in '{ts_str}': {e}") from e
    else:
        try: sec = float(ts_str.replace(',','.'))
        except ValueError: raise ValueError(f"Invalid seconds format: {ts_str}")
    if sec < 0: raise ValueError(f"Timestamp negative: {ts_str}")
    return sec

def parse_cut_list(cuts_str: str) -> List[Tuple[float, float]]:
    cuts = []
    if not cuts_str: return cuts
    parts = cuts_str.replace('\n',',').split(',')
    for i, part in enumerate(parts):
        part = part.strip()
        if not part: continue
        try:
            if '-' not in part: raise ValueError(f"Segment {i+1} ('{part}') needs '-' separator.")
            start_str, end_str = part.split('-',1)
            start_sec, end_sec = parse_timestamp_string(start_str), parse_timestamp_string(end_str)
            if start_sec >= end_sec: raise ValueError(f"End time must be after start in '{part}'")
            cuts.append((start_sec, end_sec))
        except ValueError as e: raise ValueError(f"Invalid cut segment {i+1} ('{part}'): {e}") from e
    cuts.sort(key=lambda x: x[0])
    for i in range(len(cuts)-1):
        if cuts[i][1] > cuts[i+1][0]:
            raise ValueError(f"Overlapping cuts: {format_timestamp(cuts[i][0])}-{format_timestamp(cuts[i][1])} and {format_timestamp(cuts[i+1][0])}-{format_timestamp(cuts[i+1][1])}")
    return cuts

# --- FFmpeg Operations ---
def cut_video_ffmpeg(input_video_path: Union[str, Path], cut_segments: List[Tuple[float, float]], output_video_path: Union[str, Path], log_callback=print) -> Optional[str]:
    input_file, output_file = Path(input_video_path), Path(output_video_path)
    if not cut_segments: log_callback("No cut segments. Skipping cutting."); return None
    log_callback(f"Cutting video '{input_file.name}' to create '{output_file.name}'...")
    select_filter_parts = [f"between(t,{s},{e})" for s,e in cut_segments]
    video_select_filter = "+".join(select_filter_parts)
    audio_select_filter = "+".join(select_filter_parts)
    command = ['ffmpeg','-y','-i',str(input_file),'-vf',f"select='{video_select_filter}',setpts=N/FRAME_RATE/TB",'-af',f"aselect='{audio_select_filter}',asetpts=N/FRAME_RATE/TB",'-c:v','libx264','-crf','23','-preset','medium','-c:a','aac','-b:a','192k',str(output_file)]
    log_callback(f"Executing FFmpeg for cutting: {' '.join(command)}")
    try:
        env = os.environ.copy(); env.pop("LD_LIBRARY_PATH",None)
        # For Windows, prevent console window from popping up for ffmpeg
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, universal_newlines=True, text=True, creationflags=creation_flags)
        _, stderr = process.communicate() 
        if process.returncode == 0: log_callback(f"Video cutting successful. Output: {output_file}"); return str(output_file)
        else: log_callback(f"FFmpeg error during cutting (code {process.returncode}):\n{stderr}"); return None
    except FileNotFoundError: log_callback("Error: ffmpeg not found."); return None
    except Exception as e: log_callback(f"FFmpeg error cutting: {e}\n{traceback.format_exc()}"); return None

def translate_text_srt(text: str, dest_language: str, log_callback=print) -> Optional[str]:
    if not text.strip(): return ""
    try:
        translator = Translator()
        translated = translator.translate(text, dest=dest_language)
        return translated.text
    except Exception as e:
        log_callback(f"Error during translation: '{text[:30]}...': {e}")
        return text 

def _finalize_sub_from_words(words_buffer: List[Word], sub_index: int) -> Optional[pysrt.SubRipItem]:
    if not words_buffer: return None
    text = " ".join(w.word.strip() for w in words_buffer if w.word.strip()).strip()
    if not text: return None
    start_time_sec, end_time_sec = words_buffer[0].start, words_buffer[-1].end
    if end_time_sec <= start_time_sec: end_time_sec = start_time_sec + 0.1
    start_pysrt_time, end_pysrt_time = pysrt.SubRipTime(seconds=start_time_sec), pysrt.SubRipTime(seconds=end_time_sec)
    if end_pysrt_time.ordinal <= start_pysrt_time.ordinal:
        end_pysrt_time = pysrt.SubRipTime(seconds=start_pysrt_time.seconds + start_pysrt_time.milliseconds/1000.0 + 0.1)
    return pysrt.SubRipItem(index=sub_index, start=start_pysrt_time, end=end_pysrt_time, text=text)

def transcribe_audio_to_srt(
    video_path: Union[str, Path], output_srt_path: Union[str, Path],
    model_name: str, language_code: Optional[str], device: str, compute_type: str,
    keep_audio_path: Union[str, Path], vad_filter: bool, min_silence_duration_ms: int,
    max_subtitle_duration: Optional[float], max_subtitle_chars: Optional[int],
    log_callback=print
) -> bool:
    video_file, audio_file, srt_file = Path(video_path), Path(keep_audio_path), Path(output_srt_path)
    log_callback(f"\n--- Transcribing Video: {video_file.name} ---")
    log_callback(f"Model: {model_name}, Device: {device}, Compute: {compute_type}")
    if language_code: log_callback(f"Lang hint: {language_code}")
    use_word_seg = bool(max_subtitle_duration or max_subtitle_chars)
    if use_word_seg:
        log_callback(f"Word-level re-segmentation: Max Duration: {max_subtitle_duration}s, Max Chars: {max_subtitle_chars}")
    elif vad_filter: log_callback(f"VAD enabled: Min Silence: {min_silence_duration_ms}ms")

    if not audio_file: log_callback("Error: Path for audio not specified."); return False
    try:
        log_callback(f"Extracting audio from '{video_file.name}' to: {audio_file}...")
        audio_file.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg_cmd = ['ffmpeg','-y','-i',str(video_file),'-vn','-acodec','pcm_s16le','-ar','16000','-ac','1',str(audio_file)]
        env = os.environ.copy(); env.pop("LD_LIBRARY_PATH",None)
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, universal_newlines=True, text=True, creationflags=creation_flags)
        _, stderr = process.communicate()
        if process.returncode != 0: log_callback(f"Audio extraction error: {stderr}"); return False
        log_callback(f"Audio extraction complete: {audio_file}")

        log_callback("Loading Whisper model...")
        # *** CORRECTED LINE HERE ***
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        # *** END CORRECTION ***

        log_callback("Starting transcription...");
        transcribe_opts = {"beam_size":5,"language":language_code,"vad_filter":vad_filter,"word_timestamps":True}
        if vad_filter: transcribe_opts["vad_parameters"] = {"min_silence_duration_ms":min_silence_duration_ms}
        segments_gen, info = model.transcribe(str(audio_file), **transcribe_opts)
        log_callback(f"Detected lang: {info.language} (Prob: {info.language_probability:.2f}), Audio duration: {info.duration:.2f}s")
        
        subs, sub_idx = pysrt.SubRipFile(), 0
        if use_word_seg:
            word_buf, text_buf = [], ""
            for seg in segments_gen:
                if not seg.words:
                    if seg.text.strip(): log_callback(f"Warning: Segment '{seg.text[:30]}...' no word timestamps.")
                    continue
                for word in seg.words:
                    w_text = word.word.strip()
                    if not w_text: continue
                    prosp_text = (text_buf + " " + w_text).strip() if word_buf else w_text
                    prosp_dur = word.end - (word_buf[0].start if word_buf else word.start)
                    new_sub_made = False
                    if word_buf:
                        text_exceed = max_subtitle_chars and len(prosp_text) > max_subtitle_chars and len(text_buf) > 0
                        dur_exceed = max_subtitle_duration and prosp_dur > max_subtitle_duration and (word.end - word.start < max_subtitle_duration if max_subtitle_duration else True)
                        if text_exceed or dur_exceed:
                            sub_idx+=1; item=_finalize_sub_from_words(word_buf,sub_idx);
                            if item: subs.append(item)
                            word_buf, text_buf = [word], w_text; new_sub_made=True
                    if not new_sub_made: word_buf.append(word); text_buf=prosp_text
            if word_buf: sub_idx+=1; item=_finalize_sub_from_words(word_buf,sub_idx);
            if item: subs.append(item)
        else:
            log_callback("Using Whisper's default segmentation.")
            for seg in segments_gen:
                sub_idx+=1; start=pysrt.SubRipTime(seconds=seg.start); end=pysrt.SubRipTime(seconds=seg.end)
                if end.ordinal <= start.ordinal: end=pysrt.SubRipTime(seconds=seg.start+0.1)
                subs.append(pysrt.SubRipItem(index=sub_idx,start=start,end=end,text=seg.text.strip()))
        
        if not subs: log_callback("Warning: No subs. Saving empty SRT."); srt_file.write_text("",encoding='utf-8'); return True
        subs.save(str(srt_file), encoding='utf-8'); log_callback(f"Transcription complete. SRT: {srt_file}"); return True
    except Exception as e: log_callback(f"Transcription error: {e}\n{traceback.format_exc()}"); return False
    finally:
        if audio_file.exists(): log_callback(f"Audio file kept: {audio_file}")

def process_srt_file(
    input_srt_path: Union[str, Path], output_srt_path: Union[str, Path],
    target_lang: Optional[str], delay_seconds: float, stack_languages: bool,
    log_callback=print
) -> bool:
    in_file, out_file = Path(input_srt_path), Path(output_srt_path)
    try: subs_orig = pysrt.open(str(in_file), encoding='utf-8')
    except Exception as e: log_callback(f"Error opening SRT '{in_file}': {e}"); return False
    if not subs_orig: log_callback(f"No subs in '{in_file}'. Saving empty."); out_file.write_text("",'utf-8'); return True
    
    needs_proc = target_lang or delay_seconds != 0.0
    if not needs_proc:
        try: import shutil; shutil.copyfile(in_file, out_file); log_callback(f"No SRT processing. Copied '{in_file}' to '{out_file}'."); return True
        except Exception as e: log_callback(f"Copy error: {e}. Processing manually.");
        
    final_subs = pysrt.SubRipFile()
    if target_lang: log_callback(f"Translating to '{target_lang}'" + (" and stacking..." if stack_languages else "..."))
    if delay_seconds != 0.0: log_callback(f"Applying delay: {delay_seconds:.3f}s."); subs_orig.shift(seconds=delay_seconds)
    
    for i, item in enumerate(subs_orig):
        orig_text, proc_text = item.text.strip(), item.text.strip()
        if target_lang:
            translated = translate_text_srt(orig_text, target_lang, log_callback)
            if translated and translated.strip() != orig_text:
                proc_text = f"{orig_text}\n{translated.strip()}" if stack_languages else translated.strip()
            elif not translated: log_callback(f"Warning: Translation failed for sub #{i+1}.")
        start, end = item.start, item.end
        if start.ordinal < 0: start = pysrt.SubRipTime(0)
        if end.ordinal <= start.ordinal: end = pysrt.SubRipTime(seconds=start.seconds + start.milliseconds/1000.0 + 0.1)
        final_subs.append(pysrt.SubRipItem(index=i+1, start=start, end=end, text=proc_text))
        
    try:
        if not final_subs: log_callback("Warning: Processed subs empty. Saving empty SRT.")
        final_subs.save(str(out_file), encoding='utf-8'); log_callback(f"Processed SRT saved: {out_file}"); return True
    except Exception as e: log_callback(f"Error saving processed SRT: {e}"); return False

def get_ffmpeg_color_hex(color_name_or_hex: str, alpha_hex: str = "00") -> str:
    color, alpha = color_name_or_hex.lower().strip(), alpha_hex.zfill(2)
    if color.startswith("#") and len(color) == 7:
        try: r,g,b=color[1:3],color[3:5],color[5:7]; int(r,16);int(g,16);int(b,16); return f"&H{alpha}{b}{g}{r}".upper()
        except ValueError: return f"&H{alpha}FFFFFF" 
    named = {"white":"FFFFFF","black":"000000","red":"FF0000","green":"00FF00","blue":"0000FF","yellow":"FFFF00","cyan":"00FFFF","magenta":"FF00FF"} 
    if color in named: rgb=named[color]; return f"&H{alpha}{rgb[4:6]}{rgb[2:4]}{rgb[0:2]}".upper()
    return f"&H{alpha}FFFFFF" 

def burn_subtitles_to_video_ffmpeg(
    video_path: Union[str, Path], srt_file_path: Union[str, Path], output_video_path: Union[str, Path],
    font_name: str, font_size: int, font_color: str, enable_subtitle_background: bool,
    subtitle_background_color: str, background_alpha_hex: str, force_style: Optional[str],
    log_callback=print
) -> Optional[str]:
    vid_f, srt_f, out_f = Path(video_path), Path(srt_file_path), Path(output_video_path)
    if not vid_f.exists(): log_callback(f"Video '{vid_f}' not found."); return None
    if not srt_f.exists() or srt_f.stat().st_size==0: log_callback(f"SRT '{srt_f}' not found/empty."); return None
    log_callback(f"Burning subs from '{srt_f.name}' to '{vid_f.name}'..."); log_callback(f"Font: {font_name}, Size: {font_size}, Color: {font_color}")
    
    f_color = get_ffmpeg_color_hex(font_color, "00")
    styles = [f"Fontname='{font_name}'",f"Fontsize={font_size}",f"PrimaryColour={f_color}","MarginV=25","Alignment=2"]
    if enable_subtitle_background:
        log_callback(f"Sub BG: Enabled, Color: {subtitle_background_color}, Alpha: {background_alpha_hex}")
        bg_color = get_ffmpeg_color_hex(subtitle_background_color, background_alpha_hex)
        styles.extend([f"BorderStyle=3",f"BackColour={bg_color}", "Shadow=0","Outline=0"])
    else:
        outline_color = get_ffmpeg_color_hex("black","00")
        styles.extend(["BorderStyle=1",f"OutlineColour={outline_color}","Outline=3","Shadow=0"])
    style_str = force_style if force_style else ",".join(styles)
    
    srt_path_filt = str(srt_f.resolve()).replace('\\','/')
    if os.name=='nt' and ':' in srt_path_filt: parts=srt_path_filt.split(':',1); srt_path_filt=f"{parts[0]}\\:{parts[1]}"
    
    cmd = ['ffmpeg','-y','-i',str(vid_f),'-vf',f"subtitles='{srt_path_filt}':force_style='{style_str}'",'-c:v','libx264','-crf','23','-preset','medium','-c:a','aac','-b:a','192k',str(out_f)]
    log_callback(f"Executing FFmpeg for burning: {' '.join(cmd)}")
    try:
        env = os.environ.copy(); env.pop("LD_LIBRARY_PATH",None)
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, universal_newlines=True, text=True, creationflags=creation_flags)
        _, stderr = process.communicate()
        if process.returncode == 0: log_callback(f"FFmpeg burn success. Output: {out_f}"); return str(out_f)
        else: log_callback(f"FFmpeg burn error (code {process.returncode}):\n{stderr}"); return None
    except FileNotFoundError: log_callback("Error: ffmpeg not found."); return None
    except Exception as e: log_callback(f"FFmpeg burn error: {e}\n{traceback.format_exc()}"); return None

# --- GUI Application ---
class VideoEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Editor Pro")
        self.root.geometry("800x750") 

        self.log_queue = queue.Queue()

        self.notebook = ttk.Notebook(root)
        
        self.tab_transcribe = ttk.Frame(self.notebook, padding="10")
        self.tab_burn = ttk.Frame(self.notebook, padding="10")
        self.tab_cut = ttk.Frame(self.notebook, padding="10")

        self.notebook.add(self.tab_transcribe, text='Transcribe & Process SRT')
        self.notebook.add(self.tab_burn, text='Burn SRT to Video')
        self.notebook.add(self.tab_cut, text='Cut Video')
        self.notebook.pack(expand=True, fill='both', padx=10, pady=5)

        self._create_transcribe_tab()
        self._create_burn_tab()
        self._create_cut_tab()

        log_frame = ttk.LabelFrame(root, text="Log / Status", padding="5")
        log_frame.pack(fill='x', padx=10, pady=(0,10))
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, wrap=tk.WORD, state=tk.DISABLED)
        self.log_text.pack(fill='x', expand=True)
        
        self.root.after(100, self._process_log_queue)

    def _add_log(self, message):
        self.log_queue.put(message)

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

    def _create_input_row(self, parent, label_text, var_type="string"):
        frame = ttk.Frame(parent)
        frame.pack(fill='x', pady=2)
        label = ttk.Label(frame, text=label_text, width=20)
        label.pack(side=tk.LEFT, padx=(0,5))
        
        var = None
        widget = None

        if var_type == "file_open":
            var = tk.StringVar()
            widget = ttk.Entry(frame, textvariable=var, width=50)
            widget.pack(side=tk.LEFT, expand=True, fill='x')
            button = ttk.Button(frame, text="Browse...", command=lambda v=var, e=widget: self._browse_file(v, e, filetypes=[("Video files", "*.mp4 *.mkv *.avi *.mov"),("All files", "*.*")]))
            button.pack(side=tk.LEFT, padx=(5,0))
        elif var_type == "file_save_srt":
            var = tk.StringVar()
            widget = ttk.Entry(frame, textvariable=var, width=50)
            widget.pack(side=tk.LEFT, expand=True, fill='x')
            button = ttk.Button(frame, text="Browse...", command=lambda v=var, e=widget: self._browse_save_file(v, e, defaultextension=".srt", filetypes=[("SRT files", "*.srt")]))
            button.pack(side=tk.LEFT, padx=(5,0))
        elif var_type == "file_save_video":
            var = tk.StringVar()
            widget = ttk.Entry(frame, textvariable=var, width=50)
            widget.pack(side=tk.LEFT, expand=True, fill='x')
            button = ttk.Button(frame, text="Browse...", command=lambda v=var, e=widget: self._browse_save_file(v, e, defaultextension=".mp4", filetypes=[("Video files", "*.mp4 *.mkv"), ("All files", "*.*")]))
            button.pack(side=tk.LEFT, padx=(5,0))
        elif var_type == "srt_open":
            var = tk.StringVar()
            widget = ttk.Entry(frame, textvariable=var, width=50)
            widget.pack(side=tk.LEFT, expand=True, fill='x')
            button = ttk.Button(frame, text="Browse...", command=lambda v=var, e=widget: self._browse_file(v, e, filetypes=[("SRT files", "*.srt"),("All files", "*.*")]))
            button.pack(side=tk.LEFT, padx=(5,0))
        elif var_type == "boolean":
            var = tk.BooleanVar()
            widget = ttk.Checkbutton(frame, variable=var)
            widget.pack(side=tk.LEFT)
        elif var_type == "combobox":
            var = tk.StringVar()
            widget = ttk.Combobox(frame, textvariable=var, state="readonly", width=48) 
            widget.pack(side=tk.LEFT, expand=True, fill='x')
            return var, widget # Special case for combobox, return widget too
        else: # string, float, int
            if var_type == "string": var = tk.StringVar()
            elif var_type == "float": var = tk.DoubleVar()
            elif var_type == "int": var = tk.IntVar()
            else: var = tk.StringVar() # Fallback
            widget = ttk.Entry(frame, textvariable=var, width=50)
            widget.pack(side=tk.LEFT, expand=True, fill='x')
        return var

    def _browse_file(self, var, entry_widget, filetypes):
        filepath = filedialog.askopenfilename(filetypes=filetypes)
        if filepath: var.set(filepath)

    def _browse_save_file(self, var, entry_widget, defaultextension, filetypes):
        filepath = filedialog.asksaveasfilename(defaultextension=defaultextension, filetypes=filetypes)
        if filepath: var.set(filepath)
            
    def _update_compute_types(self, *args):
        selected_device = self.ts_device_var.get()
        if selected_device == "cuda":
            self.ts_compute_type_combo['values'] = COMPUTE_TYPES_CUDA
            if self.ts_compute_type_var.get() not in COMPUTE_TYPES_CUDA:
                 self.ts_compute_type_var.set(COMPUTE_TYPES_CUDA[0])
        else: 
            self.ts_compute_type_combo['values'] = COMPUTE_TYPES_CPU
            if self.ts_compute_type_var.get() not in COMPUTE_TYPES_CPU:
                self.ts_compute_type_var.set(COMPUTE_TYPES_CPU[0])

    def _create_transcribe_tab(self):
        parent = self.tab_transcribe
        self.ts_video_in_var = self._create_input_row(parent, "Input Video:", "file_open")
        self.ts_srt_out_var = self._create_input_row(parent, "Output SRT Path:", "file_save_srt")
        self.ts_audio_out_var = self._create_input_row(parent, "Keep Audio Path:", "file_save_video")
        self.ts_audio_out_var.set("audio_temp.wav") 
        self.ts_model_var, self.ts_model_combo = self._create_input_row(parent, "Whisper Model:", "combobox")
        self.ts_model_combo['values'] = WHISPER_MODELS
        self.ts_model_var.set(DEFAULT_WHISPER_MODEL)
        self.ts_device_var, self.ts_device_combo = self._create_input_row(parent, "Device:", "combobox")
        self.ts_device_combo['values'] = DEVICES
        self.ts_device_var.set(DEVICES[0])
        self.ts_device_var.trace_add("write", self._update_compute_types)
        self.ts_compute_type_var, self.ts_compute_type_combo = self._create_input_row(parent, "Compute Type:", "combobox")
        self._update_compute_types() 
        self.ts_lang_var = self._create_input_row(parent, "Transcription Lang (e.g., en):")
        self.ts_vad_filter_var = self._create_input_row(parent, "Enable VAD Filter:", "boolean")
        self.ts_min_silence_var = self._create_input_row(parent, "Min Silence (ms for VAD):", "int")
        self.ts_min_silence_var.set(DEFAULT_MIN_SILENCE_DURATION_MS)
        self.ts_max_duration_var = self._create_input_row(parent, "Max Sub Duration (s, optional):", "float")
        self.ts_max_chars_var = self._create_input_row(parent, "Max Sub Chars (optional):", "int")
        trans_frame = ttk.LabelFrame(parent, text="Translation Options", padding="5")
        trans_frame.pack(fill='x', pady=5, expand=True)
        self.ts_target_lang_var = self._create_input_row(trans_frame, "Target Language (e.g., es):")
        self.ts_stack_langs_var = self._create_input_row(trans_frame, "Stack Languages:", "boolean")
        self.ts_delay_srt_var = self._create_input_row(trans_frame, "Delay SRT (s, optional):", "float")
        action_button = ttk.Button(parent, text="Start Transcription & Processing", command=self._run_transcribe_thread)
        action_button.pack(pady=10)

    def _run_transcribe_thread(self):
        video_in = self.ts_video_in_var.get()
        srt_out = self.ts_srt_out_var.get()
        audio_out = self.ts_audio_out_var.get()
        model = self.ts_model_var.get()
        device = self.ts_device_var.get()
        compute = self.ts_compute_type_var.get()
        lang = self.ts_lang_var.get() or None
        vad = self.ts_vad_filter_var.get()
        min_silence = self.ts_min_silence_var.get()
        
        # Handle potential empty strings for float/int vars before conversion
        max_dur_val = self.ts_max_duration_var.get()
        max_chars_val = self.ts_max_chars_var.get()
        delay_srt_val = self.ts_delay_srt_var.get()

        max_dur = float(max_dur_val) if max_dur_val else None # Handles 0.0 from DoubleVar if empty
        max_chars = int(max_chars_val) if max_chars_val else None # Handles 0 from IntVar if empty
        delay_srt = float(delay_srt_val) if delay_srt_val else 0.0


        target_lang = self.ts_target_lang_var.get() or None
        stack_langs = self.ts_stack_langs_var.get()

        if not video_in: messagebox.showerror("Input Error", "Please select an input video."); return
        if not srt_out: messagebox.showerror("Input Error", "Please specify an output SRT path."); return
        if not audio_out: messagebox.showerror("Input Error", "Please specify a path to keep audio."); return
        
        self._add_log("Starting transcription process...")
        thread = threading.Thread(target=self._transcribe_task, args=(
            video_in, srt_out, audio_out, model, lang, device, compute, vad, min_silence, max_dur, max_chars,
            target_lang, stack_langs, delay_srt
        ), daemon=True)
        thread.start()

    def _transcribe_task(self, video_in, srt_out, audio_out, model, lang, device, compute, vad, min_silence, max_dur, max_chars, target_lang, stack_langs, delay_srt):
        trans_success = transcribe_audio_to_srt( video_in, srt_out, model, lang, device, compute, audio_out, vad, min_silence, max_dur, max_chars, self._add_log )
        if not trans_success: self._add_log("Transcription failed."); messagebox.showerror("Error", "Transcription failed."); return
        self._add_log(f"Transcription successful. SRT: {srt_out}")
        if target_lang or delay_srt != 0.0:
            proc_srt_path = str(Path(srt_out).with_stem(Path(srt_out).stem + "_processed"))
            self._add_log(f"Processing SRT. Output: {proc_srt_path}")
            proc_success = process_srt_file( srt_out, proc_srt_path, target_lang, delay_srt, stack_langs, self._add_log )
            if proc_success: self._add_log(f"SRT processing successful: {proc_srt_path}"); messagebox.showinfo("Success", f"Transcription & SRT processing complete!\nGenerated: {srt_out}\nProcessed: {proc_srt_path}")
            else: self._add_log("SRT processing failed."); messagebox.showerror("Error", "SRT processing failed post-transcription.")
        else: messagebox.showinfo("Success", f"Transcription complete!\nGenerated SRT: {srt_out}")
        self._add_log("--- Transcribe & Process SRT task finished ---")

    def _create_burn_tab(self):
        parent = self.tab_burn
        self.burn_video_in_var = self._create_input_row(parent, "Input Video:", "file_open")
        self.burn_srt_in_var = self._create_input_row(parent, "Input SRT File:", "srt_open")
        self.burn_video_out_var = self._create_input_row(parent, "Output Video Path:", "file_save_video")
        style_frame = ttk.LabelFrame(parent, text="Subtitle Style Options", padding="5")
        style_frame.pack(fill='x', pady=5, expand=True)
        self.burn_font_name_var = self._create_input_row(style_frame, "Font Name:")
        self.burn_font_name_var.set(DEFAULT_FONT_NAME)
        self.burn_font_size_var = self._create_input_row(style_frame, "Font Size:", "int")
        self.burn_font_size_var.set(DEFAULT_FONT_SIZE)
        self.burn_font_color_var = self._create_input_row(style_frame, "Font Color:")
        self.burn_font_color_var.set(DEFAULT_FONT_COLOR)
        self.burn_bg_enable_var = self._create_input_row(style_frame, "Enable Background Box:", "boolean")
        self.burn_bg_color_var = self._create_input_row(style_frame, "Background Color:")
        self.burn_bg_color_var.set(DEFAULT_SUBTITLE_BACKGROUND_COLOR)
        self.burn_bg_alpha_var = self._create_input_row(style_frame, "Background Alpha (00-FF):")
        self.burn_bg_alpha_var.set(DEFAULT_SUBTITLE_BACKGROUND_ALPHA)
        self.burn_force_style_var = self._create_input_row(style_frame, "Force Style (override):")
        action_button = ttk.Button(parent, text="Burn Subtitles to Video", command=self._run_burn_thread)
        action_button.pack(pady=10)

    def _run_burn_thread(self):
        video_in, srt_in, video_out = self.burn_video_in_var.get(), self.burn_srt_in_var.get(), self.burn_video_out_var.get()
        font_name, font_size, font_color = self.burn_font_name_var.get(), self.burn_font_size_var.get(), self.burn_font_color_var.get()
        bg_enable, bg_color, bg_alpha = self.burn_bg_enable_var.get(), self.burn_bg_color_var.get(), self.burn_bg_alpha_var.get()
        force_style = self.burn_force_style_var.get() or None
        if not video_in or not srt_in or not video_out: messagebox.showerror("Input Error", "All video/SRT paths must be specified."); return
        self._add_log("Starting subtitle burning...")
        thread = threading.Thread(target=self._burn_task, args=( video_in, srt_in, video_out, font_name, font_size, font_color, bg_enable, bg_color, bg_alpha, force_style ), daemon=True)
        thread.start()

    def _burn_task(self, video_in, srt_in, video_out, font_name, font_size, font_color, bg_enable, bg_color, bg_alpha, force_style):
        res_path = burn_subtitles_to_video_ffmpeg( video_in, srt_in, video_out, font_name, font_size, font_color, bg_enable, bg_color, bg_alpha, force_style, self._add_log )
        if res_path: self._add_log(f"Burn successful! Output: {res_path}"); messagebox.showinfo("Success", f"Subtitles burned!\nOutput: {res_path}")
        else: self._add_log("Burn failed."); messagebox.showerror("Error", "Subtitle burning failed.")
        self._add_log("--- Burn SRT task finished ---")

    def _create_cut_tab(self):
        parent = self.tab_cut
        self.cut_video_in_var = self._create_input_row(parent, "Input Video:", "file_open")
        ttk.Label(parent, text="Cut Segments (e.g., 0:10-0:45,1:30.5-1:55):").pack(fill='x', pady=(5,0))
        self.cut_segments_text = tk.Text(parent, height=4, width=60)
        self.cut_segments_text.pack(fill='x', pady=2)
        self.cut_video_out_var = self._create_input_row(parent, "Output Cut Video Path:", "file_save_video")
        action_button = ttk.Button(parent, text="Cut Video", command=self._run_cut_thread)
        action_button.pack(pady=10)

    def _run_cut_thread(self):
        video_in, cuts_str, video_out = self.cut_video_in_var.get(), self.cut_segments_text.get("1.0",tk.END).strip(), self.cut_video_out_var.get()
        if not video_in or not cuts_str or not video_out: messagebox.showerror("Input Error", "All video/cut paths must be specified."); return
        try:
            cut_segments = parse_cut_list(cuts_str)
            if not cut_segments: messagebox.showinfo("Info", "No valid cuts."); return
        except ValueError as e: messagebox.showerror("Input Error", f"Invalid cuts format: {e}"); return
        self._add_log("Starting video cutting...")
        thread = threading.Thread(target=self._cut_task, args=(video_in, cut_segments, video_out), daemon=True)
        thread.start()

    def _cut_task(self, video_in, cut_segments, video_out):
        res_path = cut_video_ffmpeg(video_in, cut_segments, video_out, self._add_log)
        if res_path: self._add_log(f"Cut successful! Output: {res_path}"); messagebox.showinfo("Success", f"Video cut!\nOutput: {res_path}")
        else: self._add_log("Cut failed."); messagebox.showerror("Error", "Video cutting failed.")
        self._add_log("--- Cut Video task finished ---")

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoEditorApp(root)
    root.mainloop()
