import time
import threading
import tkinter as tk
from tkinter import ttk
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import pygetwindow as gw
import ollama
import chromadb
import keyboard
import winsound
import subprocess
import os
import json
import psutil
from PIL import Image, ImageDraw, ImageGrab
import webbrowser
from datetime import datetime
import pystray
from pystray import MenuItem as item
from pathlib import Path
from kokoro_onnx import Kokoro
import asyncio
import pyperclip
import warnings
import queue

# --- AUDIO DUCKING SETUP ---
try:
    from pycaw.pycaw import AudioUtilities
except ImportError:
    AudioUtilities = None
    print("[System Warning] pycaw not found. Audio ducking disabled.")

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- CONFIGURATION ---
WHISPER_MODEL_SIZE = "tiny.en"
SAMPLE_RATE = 16000
TASKS_FILE = "remu_tasks.json"
CONFIG_FILE = "remu_config.json"
MACROS_FILE = "remu_macros.json"
DAILY_LOG_FILE = "daily_log.md"

print("[System] Loading local Faster-Whisper model...")
whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
print("[System] Whisper model loaded successfully!")

print("[System] Loading Kokoro Neural TTS engine...")
try:
    kokoro_engine = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
    print("[System] Kokoro TTS loaded successfully!")
except Exception as e:
    print(f"[Kokoro Warning]: Model files not found locally ({e}).")
    kokoro_engine = None

print("[System] Initializing local databases and config...")
chroma_client = chromadb.PersistentClient(path="./remu_memory")
core_memory = chroma_client.get_or_create_collection(name="remu_companion_logs")
project_memory = core_memory
active_project_name = "Global"

try:
    core_memory.upsert(
        documents=["User Profile: The user's name is Sean Brandon Reyes. Always address him directly as Sean."],
        ids=["user_profile_sean"]
    )
except Exception: pass

if not os.path.exists(TASKS_FILE):
    with open(TASKS_FILE, "w") as f: json.dump([], f)

default_macros = {
    "ship it": "git status",
    "clean slate": "run pytest",
    "check status": "cpu"
}
if not os.path.exists(MACROS_FILE):
    with open(MACROS_FILE, "w") as f: json.dump(default_macros, f, indent=4)

default_config = {
    "speech_speed": 1.0,
    "voice_name": "af_heart",
    "bond_xp": 0,
    "bond_level": 1,
    "treats": 3,
    "distraction_keywords": ["youtube", "reddit", "game", "netflix", "twitch", "discord", "facebook"]
}
if not os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "w") as f: json.dump(default_config, f, indent=4)

def load_config():
    try:
        with open(CONFIG_FILE, "r") as f: return json.load(f)
    except Exception: return default_config

def save_config(config_data):
    with open(CONFIG_FILE, "w") as f: json.dump(config_data, f, indent=4)

def load_tasks():
    try:
        with open(TASKS_FILE, "r") as f: return json.load(f)
    except Exception: return []

def save_tasks(tasks):
    with open(TASKS_FILE, "w") as f: json.dump(tasks, f, indent=4)

def load_macros():
    try:
        with open(MACROS_FILE, "r") as f: return json.load(f)
    except Exception: return default_macros

def index_workspace_code():
    try:
        workspace_path = Path(".")
        allowed_extensions = {".py", ".json", ".md", ".js", ".php", ".html"}
        count = 0
        for file_path in workspace_path.glob("**/*"):
            if file_path.is_file() and file_path.suffix in allowed_extensions:
                if ".venv" in file_path.parts or ".git" in file_path.parts: continue
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                doc_id = f"code_{file_path.name}_{int(time.time())}"
                project_memory.upsert(
                    documents=[f"Project File [{file_path.name}]: {content[:1500]}?"],
                    ids=[doc_id]
                )
                count += 1
        print(f"[System] RAG Indexer: Successfully indexed {count} workspace files into {active_project_name}!")
    except Exception as e: print(f"[RAG Error]: {e}")

# --- STATE & SETTINGS ---
session_start_time = time.time()
distraction_streak = 0
current_mood = "Earnest & Sweet"
focus_mode_active = False
pomodoro_active = False
pomodoro_seconds_left = 25 * 60
pomodoro_is_break = False
is_talking = False
is_interrupted = False
is_being_petted = False
current_rms = 0.0
ducked_volumes = {}

speech_lock = threading.Lock()
audio_queue = queue.Queue()

config = load_config()
speech_speed = config.get("speech_speed", 1.0)
selected_voice_name = config.get("voice_name", "af_heart")
bond_xp = config.get("bond_xp", 0)
bond_level = config.get("bond_level", 1)
treats_count = config.get("treats", 3)
distraction_keywords = config.get("distraction_keywords", ["youtube", "reddit", "game"])

def set_audio_ducking(duck=True):
    global ducked_volumes
    if not AudioUtilities: return
    try:
        sessions = AudioUtilities.GetAllSessions()
        my_pid = os.getpid()
        for session in sessions:
            if session.Process and session.Process.pid != my_pid:
                vol_interface = session.SimpleAudioVolume
                if duck:
                    current = vol_interface.GetMasterVolume()
                    ducked_volumes[session.Process.pid] = current
                    vol_interface.SetMasterVolume(current * 0.25, None)
                else:
                    if session.Process.pid in ducked_volumes:
                        vol_interface.SetMasterVolume(ducked_volumes[session.Process.pid], None)
    except Exception as e: print(f"[Audio Ducking Error]: {e}")

# --- CONTINUOUS AUDIO QUEUE PIPELINE ---
def audio_playback_worker():
    global current_rms, is_interrupted, is_talking
    chunk_size = 1024
    try:
        with sd.OutputStream(samplerate=24000, channels=1, dtype='float32') as stream:
            while True:
                samples = audio_queue.get()
                if samples is None: continue
                
                if not is_talking:
                    is_talking = True
                    set_audio_ducking(True)

                for i in range(0, len(samples), chunk_size):
                    if is_interrupted:
                        with audio_queue.mutex: audio_queue.queue.clear()
                        break
                    
                    chunk = samples[i:i+chunk_size]
                    if len(chunk) > 0:
                        current_rms = float(np.sqrt(np.mean(chunk**2)))
                        stream.write(chunk)
                
                current_rms = 0.0
                if not is_interrupted:
                    stream.write(np.zeros(int(24000 * 0.15), dtype='float32'))
                
                if audio_queue.empty():
                    is_talking = False
                    set_audio_ducking(False)
    except Exception as e: print(f"Audio Playback Error: {e}")

threading.Thread(target=audio_playback_worker, daemon=True).start()

def add_bond_xp(amount=10):
    global bond_xp, bond_level, treats_count
    bond_xp += amount
    next_level_xp = bond_level * 100
    if bond_xp >= next_level_xp:
        bond_level += 1; bond_xp = 0; treats_count += 2
        play_sound("levelup")
        speak(f"Yay! Our bond reached Level {bond_level}! I got 2 new treats, Sean!", "Remu-chan ❤️ (Level Up)")
    cfg = load_config()
    cfg["bond_xp"] = bond_xp; cfg["bond_level"] = bond_level; cfg["treats"] = treats_count
    save_config(cfg)

def play_sound(cue_type="beep"):
    try:
        if cue_type == "success": winsound.Beep(880, 150); winsound.Beep(1175, 200)
        elif cue_type == "levelup":
            winsound.Beep(523, 100); winsound.Beep(659, 100); winsound.Beep(784, 150); winsound.Beep(1046, 250)
        elif cue_type == "rag": winsound.Beep(600, 80); winsound.Beep(850, 100)
        elif cue_type == "mood": winsound.Beep(587, 100)
        elif cue_type == "alert": winsound.Beep(900, 300); winsound.Beep(700, 300)
        else: winsound.Beep(440, 120)
    except Exception: pass

def update_mood(is_distracted: bool):
    global distraction_streak, current_mood, is_being_petted
    if is_being_petted: return
    if is_distracted: distraction_streak += 1
    else: distraction_streak = max(0, distraction_streak - 1)
    
    work_minutes = int((time.time() - session_start_time) / 60)
    old_mood = current_mood
    
    if focus_mode_active or pomodoro_active: current_mood = "Pomodoro / Focus Locked In 🍅🔒"
    elif distraction_streak >= 2: current_mood = "Sassy Toddler Mode (Pouting)"
    elif work_minutes > 25: current_mood = "Proud Toddler Mode (Focus Champion)"
    else: current_mood = "Earnest & Sweet"
        
    if old_mood != current_mood:
        play_sound("mood")
        if ui: ui.update_widget_color(current_mood)

def generate_daily_journal():
    try:
        tasks = load_tasks()
        completed_tasks = [t for t in tasks if t["completed"]]
        pending_tasks = [t for t in tasks if not t["completed"]]
        work_hours = round((time.time() - session_start_time) / 3600, 2)
        
        log_content = f"""# Remu-chan Daily Activity Log — {datetime.now().strftime('%Y-%m-%d')}
- **User:** Sean Brandon Reyes
- **Total Session Duration:** {work_hours} hours
- **Completed Tasks:** {len(completed_tasks)}
- **Pending Tasks:** {len(pending_tasks)}
- **Bond Level:** {bond_level} (XP: {bond_xp})
- **Affinity Treats:** {treats_count}

## Completed Checkpoints:\n"""
        for t in completed_tasks: log_content += f"- [x] {t['task']}\n"
        log_content += "\n## Pending Checkpoints:\n"
        for t in pending_tasks: log_content += f"- [ ] {t['task']}\n"
            
        with open(DAILY_LOG_FILE, "w", encoding="utf-8") as f: f.write(log_content)
        speak("Generated your end-of-day journal log, Sean!", "Remu-chan 📖 (Journal)")
        store_memory("Generated daily productivity journal markdown file.")
    except Exception as e: print(f"[Journal Error]: {e}")

# --- GUI: FLOATING SYSTEM HUD ---
class SystemHUD:
    def __init__(self, parent_root):
        self.win = tk.Toplevel(parent_root)
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        self.win.attributes("-alpha", 0.8)
        
        # Moved to Bottom-Left
        sh = self.win.winfo_screenheight()
        self.win.geometry(f"140x70+20+{sh - 120}")
        
        self.canvas = tk.Canvas(self.win, bg="#09090b", highlightthickness=1, highlightbackground="#3b82f6")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.text_id = self.canvas.create_text(8, 8, anchor="nw", text="Initializing...", fill="#3b82f6", font=("Consolas", 8, "bold"))
        
        # 🛑 WINDOWS API: Make the HUD 100% click-through (Ghosted)
        try:
            import ctypes
            hwnd = ctypes.windll.user32.GetParent(self.win.winfo_id())
            style = ctypes.windll.user32.GetWindowLongW(hwnd, -20)
            ctypes.windll.user32.SetWindowLongW(hwnd, -20, style | 0x00000020 | 0x00080000)
        except Exception: pass
        
        self.update_loop()
    
    def update_loop(self):
        try:
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory().percent
            pomo_str = f"🍅 {int(pomodoro_seconds_left//60)}:{int(pomodoro_seconds_left%60):02d}" if pomodoro_active else "🍅 Idle"
            self.canvas.itemconfig(self.text_id, text=f"⚡ SYSTEM HUD\nCPU: {cpu}%\nRAM: {ram}%\n{pomo_str}")
            hour = datetime.now().hour
            color = "#a855f7" if (hour >= 20 or hour < 6) else "#3b82f6"
            self.canvas.config(highlightbackground=color)
            self.canvas.itemconfig(self.text_id, fill=color)
        except Exception: pass
        self.win.after(1000, self.update_loop)

# --- GUI: TRUE TRANSPARENT MASCOT (DYNAMIC SCALING ENGINE) ---
class RemuWidget:
    def __init__(self):
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        
        self.chroma_key = "#000001"  
        self.root.attributes("-transparentcolor", self.chroma_key)
        self.root.config(bg=self.chroma_key)
        
        # Load saved size from config (defaults to 500)
        self.mascot_size = load_config().get("mascot_size", 500)
        self.raw_sprites = {}  # RAM Cache for instant resizing
        self.sprites = {}
        
        self.outfits = ["seifuku_2", "summer_dress", "pajama", "pe_uniform", "winter_outfit", "sswimsuit", "towel", "costume"]
        self.hairs = ["silver", "blondie", "brown", "long_hair", "twin_tail", "short_bob", "short_hair", "long_hair___hime_cut"]
        self.current_outfit = 0
        self.current_hair = 0
        self.text_timer = None
        
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        self.x = self.screen_width - 800
        self.y = self.screen_height - 600
        self.vx = 0; self.vy = 0
        self.is_dragging = False
        
        self.canvas = tk.Canvas(self.root, bg=self.chroma_key, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Load raw images into RAM, setup UI, and apply the initial size
        self.load_raw_sprites()
        self.setup_ui_elements()
        self.apply_resize()
        
        # RIGHT-CLICK CONTEXT MENU
        self.context_menu = tk.Menu(self.root, tearoff=0, bg="#18181b", fg="#ffffff", font=("Segoe UI", 10), activebackground="#3b82f6", bd=0)
        self.context_menu.add_command(label="💬 Chat", command=open_chat_window)
        self.context_menu.add_command(label="📝 Tasks", command=open_task_window)
        self.context_menu.add_command(label="🧠 Brain", command=open_memory_window)
        self.context_menu.add_command(label="🍬 Feed Treat", command=feed_treat)
        self.context_menu.add_command(label="🍅 Toggle Pomodoro", command=toggle_pomodoro)
        self.context_menu.add_command(label="📊 Summary", command=lambda: threading.Thread(target=lambda: handle_input("give me a summary"), daemon=True).start())
        self.context_menu.add_command(label="⚙️ Settings", command=open_settings_window)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="🛑 Stop Speech", command=interrupt_speech)
        self.context_menu.add_command(label="❌ Close Remu-chan", command=self.on_close)

        self.hud = SystemHUD(self.root)
        
        # MOUSE BINDINGS
        self.canvas.bind("<Button-1>", self.start_move)
        self.canvas.bind("<B1-Motion>", self.do_move)
        self.canvas.bind("<ButtonRelease-1>", self.stop_move)
        self.canvas.bind("<Double-Button-1>", self.pet_remu)   
        self.canvas.bind("<Button-3>", self.show_context_menu) 
        self.canvas.bind("<Button-2>", self.change_outfit)
        self.canvas.bind("<Shift-Button-2>", self.change_hair)
        
        # 🛑 INSTANT SCROLL-WHEEL RESIZING BINDINGS
        self.canvas.bind("<MouseWheel>", self.on_scroll)      # Windows
        self.canvas.bind("<Button-4>", self.on_scroll)        # Linux Up
        self.canvas.bind("<Button-5>", self.on_scroll)        # Linux Down

        self.anim_state = 0
        self.root.after(50, self.animate_canvas_art)

    def load_raw_sprites(self):
        """Loads original images into RAM once so resizing is instantaneous."""
        from PIL import Image
        import os
        required_sprites = [
            "base_body", "normal", "smile", "laugh", "sleepy", "angry", "annoyed", "delighted", "shocked", "smug", "sad",
            "1", "2", "flower", "choker", "black_glasses", "red_glasses", "circle_glasses"
        ] + self.outfits + self.hairs
        
        for name in required_sprites:
            path = f"sprites/{name}.png"
            if os.path.exists(path):
                try:
                    self.raw_sprites[name] = Image.open(path).convert("RGBA")
                except Exception: pass

    def setup_ui_elements(self):
        """Creates the canvas objects invisibly at 0,0 before sizing them."""
        self.body_item = self.canvas.create_image(0, 0, anchor=tk.S)
        self.clothes_item = self.canvas.create_image(0, 0, anchor=tk.S)
        self.face_item = self.canvas.create_image(0, 0, anchor=tk.S)
        self.blush_item = self.canvas.create_image(0, 0, anchor=tk.S)
        self.hair_item = self.canvas.create_image(0, 0, anchor=tk.S)

        self.name_shadow = self.canvas.create_text(0, 0, text="Remu-chan 🌸", fill="#000000", font=("Segoe UI", 11, "bold"), anchor="nw")
        self.name_text = self.canvas.create_text(0, 0, text="Remu-chan 🌸", fill="#3b82f6", font=("Segoe UI", 11, "bold"), anchor="nw")
        
        self.msg_shadow = self.canvas.create_text(0, 0, text="Initializing...", fill="#000000", font=("Segoe UI", 12, "bold"), anchor="nw", width=340)
        self.msg_text = self.canvas.create_text(0, 0, text="Initializing...", fill="#ffffff", font=("Segoe UI", 12, "bold"), anchor="nw", width=340)

    def apply_resize(self):
        """Dynamically recalculates all math, scales images, and updates UI positions."""
        from PIL import ImageTk, Image
        
        mascot_w = int(self.mascot_size * 0.8)
        text_w = 360
        
        self.width = mascot_w + text_w
        self.height = self.mascot_size
        
        # Scale the invisible window to fit her new size
        self.root.geometry(f"{self.width}x{self.height}+{int(self.x)}+{int(self.y)}")
        self.canvas.config(width=self.width, height=self.height)
        
        # Scale RAM images to new size using high-quality Lanczos resampling
        for name, raw_img in self.raw_sprites.items():
            aspect = raw_img.width / raw_img.height
            new_width = int(self.mascot_size * aspect)
            resized = raw_img.resize((new_width, self.mascot_size), Image.Resampling.LANCZOS)
            self.sprites[name] = ImageTk.PhotoImage(resized)
            
        # Update Images
        self.canvas.itemconfig(self.body_item, image=self.sprites.get('base_body'))
        self.canvas.itemconfig(self.clothes_item, image=self.sprites.get(self.outfits[self.current_outfit]))
        self.canvas.itemconfig(self.face_item, image=self.sprites.get('normal'))
        self.canvas.itemconfig(self.hair_item, image=self.sprites.get(self.hairs[self.current_hair]))
        
        # Reposition Character
        mascot_x = mascot_w // 2
        mascot_y = self.height 
        self.canvas.coords(self.body_item, mascot_x, mascot_y)
        self.canvas.coords(self.clothes_item, mascot_x, mascot_y)
        self.canvas.coords(self.face_item, mascot_x, mascot_y)
        self.canvas.coords(self.blush_item, mascot_x, mascot_y)
        self.canvas.coords(self.hair_item, mascot_x, mascot_y)
        
        # Reposition Subtitles perfectly relative to her new size
        text_x = mascot_w - 40
        text_y = int(self.mascot_size * 0.25)
        self.canvas.coords(self.name_shadow, text_x + 2, text_y + 2)
        self.canvas.coords(self.name_text, text_x, text_y)
        self.canvas.coords(self.msg_shadow, text_x + 2, text_y + 27)
        self.canvas.coords(self.msg_text, text_x, text_y + 25)

    def on_scroll(self, event):
        """Instantly resizes the mascot when the mouse wheel is scrolled."""
        # Handle both Windows (event.delta) and Linux (Button-4/5) scroll events
        if event.num == 4 or getattr(event, 'delta', 0) > 0:
            self.mascot_size += 25
        elif event.num == 5 or getattr(event, 'delta', 0) < 0:
            self.mascot_size -= 25
            
        # Hard limits so she doesn't disappear or crash the UI
        if self.mascot_size < 150: self.mascot_size = 150
        if self.mascot_size > 1200: self.mascot_size = 1200
        
        self.apply_resize()
        
        # Save new size to config seamlessly
        cfg = load_config()
        cfg["mascot_size"] = self.mascot_size
        save_config(cfg)

    def show_context_menu(self, event):
        self.context_menu.post(event.x_root, event.y_root)

    def change_outfit(self, event):
        self.current_outfit = (self.current_outfit + 1) % len(self.outfits)
        new_fit = self.outfits[self.current_outfit]
        if self.sprites.get(new_fit):
            self.canvas.itemconfig(self.clothes_item, image=self.sprites[new_fit])
            play_sound("success")
            self.update_text(f"Changed into my {new_fit.replace('_', ' ')}!", "Remu-chan 👗 (Wardrobe)")

    def change_hair(self, event):
        self.current_hair = (self.current_hair + 1) % len(self.hairs)
        new_hair = self.hairs[self.current_hair]
        if self.sprites.get(new_hair):
            self.canvas.itemconfig(self.hair_item, image=self.sprites[new_hair])
            play_sound("success")
            self.update_text(f"How does the {new_hair.replace('_', ' ')} look on me?", "Remu-chan ✂️ (Salon)")

    def set_face(self, face_name):
        if self.sprites.get(face_name):
            self.canvas.itemconfig(self.face_item, image=self.sprites[face_name])

    def trigger_spotlight_ui(self):
        if hasattr(self, 'spotlight') and self.spotlight.winfo_exists():
            self.spotlight.focus_force()
            return
            
        self.spotlight = tk.Toplevel(self.root)
        self.spotlight.overrideredirect(True)
        self.spotlight.attributes("-topmost", True)
        self.spotlight.attributes("-alpha", 0.95)
        self.spotlight.config(bg="#09090b")
        
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        w, h = 600, 60
        self.spotlight.geometry(f"{w}x{h}+{int(sw/2 - w/2)}+{int(sh/3)}")
        
        entry = tk.Entry(self.spotlight, font=("Segoe UI", 16), bg="#18181b", fg="#ffffff", insertbackground="white", relief="flat")
        entry.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        entry.focus_force()
        
        def on_enter(e):
            txt = entry.get().strip()
            self.spotlight.destroy()
            if txt:
                macros = load_macros()
                if txt.lower() in macros: txt = macros[txt.lower()]
                
                if txt.startswith("/term "): execute_safe_terminal_command(txt.replace("/term ", ""))
                elif txt.startswith("/memory "):
                    res = recall_memories(txt.replace("/memory ", ""), memory_target="core")
                    speak(f"Memory results: {res[:200]}", "Remu-chan 🧠 (Brain)")
                else: threading.Thread(target=handle_input, args=(txt,), daemon=True).start()
                    
        def on_esc(e): self.spotlight.destroy()
        entry.bind("<Return>", on_enter); entry.bind("<Escape>", on_esc)
        self.spotlight.bind("<FocusOut>", on_esc)

    def pet_remu(self, event):
        global is_being_petted
        is_being_petted = True
        add_bond_xp(3); play_sound("levelup")
        
        self.set_face("delighted")
        if self.sprites.get('2'):
            self.canvas.itemconfig(self.blush_item, image=self.sprites['2'])
            
        self.canvas.itemconfig(self.name_text, fill="#ec4899")
        self.update_text("Hehe, that feels nice... thanks for the headpats, Sean!", "Remu-chan 💖 (Happy)")
        
        def reset_pet():
            global is_being_petted
            is_being_petted = False
            self.canvas.itemconfig(self.blush_item, image=None)
            self.update_widget_color(current_mood)
        self.root.after(3000, reset_pet)

    def start_move(self, event):
        self.is_dragging = True
        self.drag_offset_x = event.x
        self.drag_offset_y = event.y

    def do_move(self, event):
        if self.is_dragging:
            self.x = self.root.winfo_pointerx() - self.drag_offset_x
            self.y = self.root.winfo_pointery() - self.drag_offset_y
            self.root.geometry(f"+{int(self.x)}+{int(self.y)}")

    def stop_move(self, event): self.is_dragging = False

    def animate_canvas_art(self):
        try:
            global is_talking, is_being_petted, current_rms
            hour = datetime.now().hour
            is_night_mode = hour >= 20 or hour < 6

            if not is_being_petted:
                if is_talking:
                    if current_rms > 0.06: self.set_face("laugh")      
                    elif current_rms > 0.02: self.set_face("smile")    
                    else: self.set_face("normal")                      
                else:
                    if "Sassy" in current_mood: 
                        self.set_face("angry")
                        if self.sprites.get('1'):
                            self.canvas.itemconfig(self.blush_item, image=self.sprites['1'])
                    elif focus_mode_active or pomodoro_active or is_night_mode:
                        self.set_face("sleepy")
                        self.canvas.itemconfig(self.blush_item, image=None)
                    else:
                        self.canvas.itemconfig(self.blush_item, image=None)
                        if (self.anim_state // 2) % 16 == 0: 
                            self.set_face("sleepy") 
                        else: 
                            self.set_face("normal")

            self.anim_state += 1
        except Exception: pass
        self.root.after(50, self.animate_canvas_art)

    def update_widget_color(self, mood):
        global is_being_petted
        if is_being_petted: return
        hour = datetime.now().hour
        is_night = hour >= 20 or hour < 6

        if "Sassy" in mood: target_color = "#ef4444"
        elif "Proud" in mood or "Focus" in mood or "Pomodoro" in mood: target_color = "#eab308"
        elif is_night: target_color = "#6d28d9"
        else: target_color = "#3b82f6"

        self.canvas.itemconfig(self.name_text, fill=target_color)

    def hide_text(self):
        """Fades out subtitles when she is done speaking."""
        self.canvas.itemconfig(self.name_shadow, state="hidden")
        self.canvas.itemconfig(self.name_text, state="hidden")
        self.canvas.itemconfig(self.msg_shadow, state="hidden")
        self.canvas.itemconfig(self.msg_text, state="hidden")

    def update_text(self, message, subtitle="Remu-chan 🌸"):
        full_sub = f"{subtitle} [Lv. {bond_level}] (Treats: {treats_count})"
        
        self.canvas.itemconfig(self.name_shadow, text=full_sub, state="normal")
        self.canvas.itemconfig(self.name_text, text=full_sub, state="normal")
        self.canvas.itemconfig(self.msg_shadow, text=message, state="normal")
        self.canvas.itemconfig(self.msg_text, text=message, state="normal")
        
        if self.text_timer:
            self.root.after_cancel(self.text_timer)
        self.text_timer = self.root.after(8000, self.hide_text) 
        
        self.root.update_idletasks()

    def on_close(self):
        print("\n[System] Shutting down Remu-chan...")
        os._exit(0)

    def run(self): self.root.mainloop()

ui = None
chat_box_history = None

def trigger_spotlight():
    if ui: ui.root.after(0, ui.trigger_spotlight_ui)

def feed_treat():
    global treats_count
    if treats_count > 0:
        treats_count -= 1; add_bond_xp(35); play_sound("levelup")
        speak("Yum! Thank you for the treat, Sean! I feel so happy!", "Remu-chan (≧◡≦)💖 (Treat)")
    else: speak("You're out of treats right now, Sean! Complete tasks or level up to earn more!", "Remu-chan 🌸")

def open_task_window():
    def refresh_list():
        for row in tree.get_children(): tree.delete(row)
        tasks = load_tasks()
        for t in tasks:
            status_text = "✅ Complete" if t["completed"] else "⏳ Pending"
            tree.insert("", "end", values=(t["id"], t["task"], status_text))

    def complete_selected():
        selected = tree.selection()
        if not selected: return
        item = tree.item(selected)
        task_id = item['values'][0]
        tasks = load_tasks()
        for t in tasks:
            if t["id"] == task_id:
                t["completed"] = True
                break
        save_tasks(tasks); refresh_list(); add_bond_xp(25)
        speak(f"Marked task {task_id} as complete! I'm so proud of you, Sean!", "Remu-chan 🎉")

    def delete_selected():
        selected = tree.selection()
        if not selected: return
        item = tree.item(selected)
        task_id = item['values'][0]
        tasks = [t for t in load_tasks() if t["id"] != task_id]
        save_tasks(tasks); refresh_list()

    t_win = tk.Toplevel()
    t_win.title("Remu-chan's Task Checklist")
    t_win.geometry("450x300")
    t_win.attributes("-topmost", True)
    t_win.config(bg="#1a1a1c")

    tk.Label(t_win, text="Sean's Task Checklist 📝", fg="#3b82f6", bg="#1a1a1c", font=("Segoe UI", 11, "bold")).pack(pady=8)
    columns = ("ID", "Task", "Status")
    tree = ttk.Treeview(t_win, columns=columns, show="headings", height=8)
    tree.heading("ID", text="ID"); tree.heading("Task", text="Task Description"); tree.heading("Status", text="Status")
    tree.column("ID", width=30); tree.column("Task", width=260); tree.column("Status", width=90)
    tree.pack(padx=10, pady=5)

    refresh_list()
    b_frame = tk.Frame(t_win, bg="#1a1a1c"); b_frame.pack(pady=8)
    tk.Button(b_frame, text="✅ Complete", command=complete_selected, bg="#27272a", fg="#ffffff", font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=5)
    tk.Button(b_frame, text="🗑️ Delete", command=delete_selected, bg="#7f1d1d", fg="#ffffff", font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=5)

def open_memory_window():
    m_win = tk.Toplevel()
    m_win.title(f"Vector Brain - {active_project_name}")
    m_win.geometry("500x350")
    m_win.attributes("-topmost", True)
    m_win.config(bg="#1a1a1c")
    
    tk.Label(m_win, text=f"ChromaDB Context: {active_project_name} 🧠", fg="#3b82f6", bg="#1a1a1c", font=("Segoe UI", 11, "bold")).pack(pady=8)
    columns = ("ID", "Memory")
    tree = ttk.Treeview(m_win, columns=columns, show="headings", height=10)
    tree.heading("ID", text="Memory ID"); tree.heading("Memory", text="Vector Context Preview")
    tree.column("ID", width=120); tree.column("Memory", width=340)
    tree.pack(padx=10, pady=5)
    
    def refresh():
        for row in tree.get_children(): tree.delete(row)
        try:
            data = project_memory.get()
            for i in range(len(data['ids'])):
                snippet = (data['documents'][i][:80] + "...") if len(data['documents'][i]) > 80 else data['documents'][i]
                tree.insert("", "end", values=(data['ids'][i], snippet))
        except Exception: pass
    
    def delete_selected():
        selected = tree.selection()
        if not selected: return
        item = tree.item(selected)
        mem_id = item['values'][0]
        try:
            project_memory.delete(ids=[str(mem_id)])
            refresh(); speak("I've wiped that memory from my vector brain, Sean!", "Remu-chan 🧠 (Brain)")
        except Exception: pass
        
    refresh()
    b_frame = tk.Frame(m_win, bg="#1a1a1c"); b_frame.pack(pady=8)
    tk.Button(b_frame, text="🗑️ Delete Memory", command=delete_selected, bg="#7f1d1d", fg="#ffffff", font=("Segoe UI", 9)).pack()

def open_chat_window():
    global chat_box_history
    def send_msg(event=None):
        txt = entry.get().strip()
        if not txt: return
        chat_box_history.insert(tk.END, f"Sean: {txt}\n")
        entry.delete(0, tk.END)
        threading.Thread(target=handle_input, args=(txt,), daemon=True).start()

    c_win = tk.Toplevel()
    c_win.title("Remu-chan Chat Box")
    c_win.geometry("420x380")
    c_win.attributes("-topmost", True)
    c_win.config(bg="#1a1a1c")

    tk.Label(c_win, text="Remu-chan Direct Chat 💬", fg="#3b82f6", bg="#1a1a1c", font=("Segoe UI", 11, "bold")).pack(pady=8)
    chat_box_history = tk.Text(c_win, bg="#27272a", fg="#ffffff", font=("Segoe UI", 9), wrap=tk.WORD, height=15)
    chat_box_history.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

    bottom_frame = tk.Frame(c_win, bg="#1a1a1c")
    bottom_frame.pack(fill=tk.X, padx=10, pady=8)
    entry = tk.Entry(bottom_frame, font=("Segoe UI", 10), bg="#27272a", fg="#ffffff", insertbackground="white")
    entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
    entry.bind("<Return>", send_msg)
    tk.Button(bottom_frame, text="Send", command=send_msg, bg="#3b82f6", fg="#ffffff", font=("Segoe UI", 9)).pack(side=tk.RIGHT)

def toggle_pomodoro():
    global pomodoro_active
    pomodoro_active = not pomodoro_active
    update_mood(False)
    if pomodoro_active:
        play_sound("success"); speak("Pomodoro timer started! 25 minutes of deep work, Sean!", "Remu-chan 🍅 (Pomodoro)")
    else: speak("Pomodoro timer stopped, Sean!", "Remu-chan 🌸")

def pomodoro_timer_loop():
    global pomodoro_active, pomodoro_seconds_left, pomodoro_is_break
    while True:
        if pomodoro_active:
            time.sleep(1); pomodoro_seconds_left -= 1
            if pomodoro_seconds_left <= 0:
                play_sound("success")
                if not pomodoro_is_break:
                    pomodoro_is_break = True; pomodoro_seconds_left = 5 * 60
                    speak("Time's up, Sean! Take a 5 minute break!", "Remu-chan 🍅 (Break Time)")
                else:
                    pomodoro_is_break = False; pomodoro_seconds_left = 25 * 60
                    speak("Break over, Sean! Back to work!", "Remu-chan 🍅 (Work Time)")
        else: time.sleep(2)

def startup_briefing():
    time.sleep(2.0)
    tasks = load_tasks()
    pending = [t for t in tasks if not t["completed"]]
    cpu = psutil.cpu_percent(interval=0.5)
    
    hour = datetime.now().hour
    if hour >= 20 or hour < 6:
        msg = f"Good evening, Sean. System CPU is at {cpu} percent, and you have {len(pending)} tasks left. Don't forget your fasting window starts soon!"
        sub = "Remu-chan 🌙 (Night Mode)"
    else:
        msg = f"Good to see you, Sean! You have {len(pending)} pending tasks on your checklist, and system CPU is at {cpu} percent. Let's make today productive!"
        sub = "Remu-chan 🌅 (Briefing)"
        
    speak(msg, sub)
    store_memory("Executed automated startup briefing.")

def git_health_monitor_loop():
    while True:
        time.sleep(600)
        try:
            result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                changes_count = len(result.stdout.strip().split("\n"))
                speak(f"Hey Sean, I noticed you have {changes_count} uncommitted changes in your git repository. Don't forget to commit!", "Remu-chan 💻 (Git)")
                store_memory(f"Git monitor noticed {changes_count} uncommitted changes.")
        except Exception: pass

def clipboard_monitor_loop():
    last_clip = ""
    while True:
        time.sleep(2)
        try:
            current_clip = pyperclip.paste()
            if current_clip and current_clip != last_clip and len(current_clip) > 10:
                last_clip = current_clip; lower_clip = current_clip.lower()
                if "traceback (most recent call last)" in lower_clip or "exception:" in lower_clip or "error:" in lower_clip:
                    play_sound("alert")
                    prompt = f"Explain this error concisely and suggest a fix under 30 words: {current_clip[:1000]}"
                    reply = query_ollama(prompt, "You are Remu-chan, an expert debugging assistant.")
                    speak(reply, "Remu-chan 🐛 (Debug)")
                    store_memory(f"Proactive Error Fix: {reply}")
        except Exception: pass

def workflow_copilot_loop():
    global active_project_name, project_memory
    last_file_context = ""
    last_ram_warning = time.time()
    
    while True:
        time.sleep(25)
        try:
            window_title = get_active_window().lower()
            if "visual studio code" in window_title or "code" in window_title:
                parts = window_title.split("-")
                if len(parts) >= 2:
                    proj_name = parts[-2].strip().title()
                    if proj_name != active_project_name and len(proj_name) > 1:
                        active_project_name = proj_name
                        clean_name = "".join([c if c.isalnum() else "_" for c in proj_name])[:50]
                        project_memory = chroma_client.get_or_create_collection(name=f"remu_code_{clean_name}")
                        play_sound("rag")
                        speak(f"Switched context to new project workspace: {proj_name}.", "Remu-chan 📂 (Workspace)")
        except Exception: pass

        try:
            mem = psutil.virtual_memory()
            if mem.percent > 85.0 and (time.time() - last_ram_warning) > 300: 
                top_proc = None; max_mem = 0
                for proc in psutil.process_iter(['name', 'memory_info']):
                    try:
                        mem_usage = proc.info['memory_info'].rss
                        if mem_usage > max_mem: max_mem = mem_usage; top_proc = proc.info['name']
                    except Exception: pass
                if top_proc:
                    play_sound("alert")
                    speak(f"Sean, system memory is at {mem.percent} percent. {top_proc} is using a lot of it. Just say 'Kill {top_proc.replace('.exe','')}' if you want me to terminate it.", "Remu-chan ⚠️ (System)")
                    last_ram_warning = time.time()
        except Exception: pass

        try:
            window_title = get_active_window().lower()
            if "visual studio code" in window_title or "code" in window_title:
                if "-" in window_title:
                    filename = window_title.split("-")[0].strip()
                    if filename != last_file_context and len(filename) > 2:
                        last_file_context = filename
                        if filename.endswith(".py"): tip = "Python tip: Keep your functions modular and remember type hints, Sean!"
                        elif filename.endswith(".js") or filename.endswith(".ts"): tip = "JS/TS tip: Watch out for asynchronous promises and null checks, Sean!"
                        elif filename.endswith(".json"): tip = "JSON tip: Double check your commas and syntax brackets, Sean!"
                        else: tip = f"You're making great progress working on {filename}, Sean!"
                        play_sound("rag"); speak(tip, "Remu-chan 💡 (Copilot)")
                        store_memory(f"Proactive Copilot Tip for {filename}: {tip}")
        except Exception: pass

def open_settings_window():
    global speech_speed, distraction_keywords, selected_voice_name
    def save_settings():
        global speech_speed, distraction_keywords, selected_voice_name
        try:
            speech_speed = float(speed_slider.get())
            raw_keywords = kw_entry.get()
            distraction_keywords = [kw.strip().lower() for kw in raw_keywords.split(",") if kw.strip()]
            selected_voice_name = voice_combo.get()
            cfg = load_config()
            cfg["speech_speed"] = speech_speed; cfg["voice_name"] = selected_voice_name; cfg["distraction_keywords"] = distraction_keywords
            save_config(cfg)
            speak("Settings and neural voice updated, Sean!", "Remu-chan ⚙️")
            s_win.destroy()
        except Exception: pass

    kokoro_voices = ["af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky", "am_adam", "am_michael", "bf_emma", "bf_isabella", "bm_george"]
    s_win = tk.Toplevel()
    s_win.title("Remu-chan Neural Settings")
    s_win.geometry("380x340")
    s_win.attributes("-topmost", True)
    s_win.config(bg="#1a1a1c")

    tk.Label(s_win, text="Remu-chan Neural Settings ⚙️", fg="#3b82f6", bg="#1a1a1c", font=("Segoe UI", 11, "bold")).pack(pady=8)
    tk.Label(s_win, text="Speech Speed (Multiplier):", fg="#ffffff", bg="#1a1a1c", font=("Segoe UI", 9)).pack()
    speed_slider = tk.Scale(s_win, from_=0.7, to=1.5, resolution=0.1, orient=tk.HORIZONTAL, bg="#27272a", fg="#ffffff", highlightbackground="#1a1a1c")
    speed_slider.set(speech_speed)
    speed_slider.pack(pady=2)

    tk.Label(s_win, text="Select Neural Voice (Kokoro):", fg="#ffffff", bg="#1a1a1c", font=("Segoe UI", 9)).pack(pady=3)
    voice_combo = ttk.Combobox(s_win, values=kokoro_voices, width=35, state="readonly")
    if selected_voice_name in kokoro_voices: voice_combo.set(selected_voice_name)
    else: voice_combo.set("af_heart")
    voice_combo.pack(pady=2)

    tk.Label(s_win, text="Distraction Keywords (comma separated):", fg="#ffffff", bg="#1a1a1c", font=("Segoe UI", 9)).pack(pady=3)
    kw_entry = tk.Entry(s_win, width=42, font=("Segoe UI", 9))
    kw_entry.insert(0, ", ".join(distraction_keywords))
    kw_entry.pack(pady=2)
    tk.Button(s_win, text="Save Settings", command=save_settings, bg="#3b82f6", fg="#ffffff", font=("Segoe UI", 9)).pack(pady=10)

def interrupt_speech():
    global is_interrupted
    is_interrupted = True
    with audio_queue.mutex: audio_queue.queue.clear()
    print("\n[System] Remu-chan speech interrupted by Sean!")
    if ui: ui.update_text("(Interrupted)", "Remu-chan 🛑")

def speak(text: str, subtitle="Remu-chan 🌸"):
    global is_interrupted
    is_interrupted = False
    print(f"\n[Remu-chan]: {text}")
    if ui: ui.update_text(text, subtitle)
    if chat_box_history:
        chat_box_history.insert(tk.END, f"Remu-chan: {text}\n")
        chat_box_history.see(tk.END)
    
    def generator_thread():
        try:
            if kokoro_engine:
                async def stream_audio():
                    stream = kokoro_engine.create_stream(text, voice=selected_voice_name, speed=speech_speed, lang="en-us")
                    async for samples, sample_rate in stream:
                        if is_interrupted: break
                        audio_queue.put(samples)
                asyncio.run(stream_audio())
            else: winsound.Beep(440, 200)
        except Exception as e: print(f"[Kokoro TTS Error]: {e}")

    threading.Thread(target=generator_thread, daemon=True).start()

def get_active_window() -> str:
    try:
        window = gw.getActiveWindow()
        return window.title.strip() if window and window.title else "Unknown"
    except Exception: return "Unknown"

def recall_memories(query: str, n_results=3, memory_target="project") -> str:
    try:
        target_db = project_memory if memory_target == "project" else core_memory
        results = target_db.query(query_texts=[query], n_results=n_results)
        documents = results.get("documents", [[]])[0]
        return " | ".join(documents) if documents else "No prior context found."
    except Exception: return "No prior context found."

def store_memory(text_to_store: str, target="core"):
    try:
        doc_id = f"mem_{int(time.time() * 1000)}"
        db = project_memory if target == "project" else core_memory
        db.add(documents=[text_to_store], ids=[doc_id])
    except Exception as e: print(f"[Memory Error]: {e}")

# --- OLLAMA TOOL DEFINITIONS & EXECUTION ---
def tool_open_app(app_name: str):
    app_lower = app_name.lower().strip()
    if "notepad" in app_lower: 
        subprocess.Popen(["notepad.exe"])
        return "Opened Notepad successfully."
    elif "code" in app_lower or "vs code" in app_lower: 
        subprocess.Popen(["code"])
        return "Opened VS Code successfully."
    elif "folder" in app_lower or "explorer" in app_lower: 
        subprocess.Popen(["explorer.exe", "."])
        return "Opened current folder successfully."
    
    if app_lower.startswith("http://") or app_lower.startswith("https://"):
        webbrowser.open(app_lower)
        return f"Navigated directly to URL: {app_lower}"
    elif ".com" in app_lower or ".org" in app_lower or ".net" in app_lower:
        webbrowser.open(f"https://{app_lower}")
        return f"Navigated directly to website: {app_lower}"
    
    common_sites = ["youtube", "facebook", "twitter", "reddit", "github", "twitch"]
    if app_lower in common_sites:
        webbrowser.open(f"https://www.{app_lower}.com")
        return f"Opened {app_lower}.com successfully."
    
    return f"Could not resolve application: {app_name}"

def tool_google_search(query: str):
    import urllib.parse
    safe_query = urllib.parse.quote(query)
    webbrowser.open(f"https://www.google.com/search?q={safe_query}")
    return f"Executed a Google search for: {query}"

def tool_check_system():
    cpu = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory().percent
    return f"CPU usage is {cpu}%, RAM usage is {ram}%."

def tool_run_command(cmd: str):
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        output = res.stdout.strip() or res.stderr.strip()
        return f"Command output: {output[:300]}"
    except Exception as e:
        return f"Command failed: {e}"

ollama_tools = [
    {
        'type': 'function',
        'function': {
            'name': 'tool_open_app',
            'description': 'Open local apps (notepad, vs code) or direct websites (youtube.com).',
            'parameters': {
                'type': 'object',
                'properties': {'app_name': {'type': 'string', 'description': 'App or website name'}},
                'required': ['app_name']
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'tool_google_search',
            'description': 'Search Google for tutorials, videos, or information.',
            'parameters': {
                'type': 'object',
                'properties': {'query': {'type': 'string', 'description': 'The search query string'}},
                'required': ['query']
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'tool_check_system',
            'description': 'Check current system CPU and RAM usage percentages.',
            'parameters': {'type': 'object', 'properties': {}}
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'tool_run_command',
            'description': 'Run a safe terminal command like git status or pytest.',
            'parameters': {
                'type': 'object',
                'properties': {'cmd': {'type': 'string', 'description': 'The terminal command to execute'}},
                'required': ['cmd']
            }
        }
    }
]

def query_ollama_with_tools(prompt: str, system_prompt: str) -> str:
    import ctypes
    import re
    try:
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt}
        ]
        response = ollama.chat(
            model='llama3.2:3b',
            messages=messages,
            tools=ollama_tools
        )
        
        msg = response.get('message', {})
        content = msg.get('content', '') or ""
        content = content.strip()
        
        # Hallucination JSON filter
        if "{" in content and '"name"' in content and "tool_" in content:
            return "I got a little confused trying to process that command! Can we try again, Sean?"

        if msg.get('tool_calls'):
            for tool in msg['tool_calls']:
                fname = tool['function']['name']
                fargs = tool['function']['arguments']
                
                prompt_msg = f"Remu-chan wants to execute a system command.\n\nAction: {fname}\nParameters: {fargs}\n\nDo you allow this?"
                user_approval = ctypes.windll.user32.MessageBoxW(0, prompt_msg, "Remu-chan Action Approval", 4 | 0x30 | 0x40000)
                
                if user_approval == 6:  # IDYES
                    tool_result = ""
                    if fname == 'tool_open_app': tool_result = tool_open_app(**fargs)
                    elif fname == 'tool_google_search': tool_result = tool_google_search(**fargs)
                    elif fname == 'tool_check_system': tool_result = tool_check_system()
                    elif fname == 'tool_run_command': tool_result = tool_run_command(**fargs)
                else:
                    tool_result = "The user DENIED your request. Apologize to Sean."
                
                messages.append(msg)
                messages.append({'role': 'tool', 'content': str(tool_result), 'name': fname})
                
                final_response = ollama.chat(model='llama3.2:3b', messages=messages)
                content = final_response['message']['content'].strip()

        # 🛑 EXPRESSION PARSER: Translate asterisks into physical sprite changes
        lower_content = content.lower()
        if "pout" in lower_content or "angry" in lower_content:
            if ui: ui.root.after(0, lambda: ui.set_face("angry"))
        elif "laugh" in lower_content or "giggle" in lower_content:
            if ui: ui.root.after(0, lambda: ui.set_face("laugh"))
        elif "smile" in lower_content or "happy" in lower_content:
            if ui: ui.root.after(0, lambda: ui.set_face("smile"))
        elif "sad" in lower_content:
            if ui: ui.root.after(0, lambda: ui.set_face("sad"))
            
        # Strip out text asterisks so she never speaks stage directions aloud
        clean_content = re.sub(r'\*[^*]+\*', '', content).strip()
        return clean_content if clean_content else content
    except Exception as e:
        print(f"[Ollama Tool Error]: {e}")
        return "I'm having trouble processing that request..."

def analyze_screen_vision(user_prompt: str):
    try:
        speak("Let me look at your active window, Sean...", "Remu-chan (O_O)👁️")
        os.makedirs("screenshots", exist_ok=True)
        img_path = os.path.abspath(f"screenshots/vision_{int(time.time())}.png")
        
        bbox = None
        try:
            win = gw.getActiveWindow()
            if win and win.title and win.title != "Remu-chan":
                left, top = max(0, win.left), max(0, win.top)
                right, bottom = win.right, win.bottom
                if right > left and bottom > top: bbox = (left, top, right, bottom)
        except Exception: pass

        if bbox: ImageGrab.grab(bbox=bbox).save(img_path)
        else: ImageGrab.grab().save(img_path)
        
        response = ollama.chat(
            model='llava',
            messages=[{'role': 'user', 'content': user_prompt or "What do you see on my screen? Give a brief summary.", 'images': [img_path]}]
        )
        reply = response['message']['content'].strip()
        speak(reply[:130], "Remu-chan 👁️ (Vision)")
        store_memory(f"Vision Analysis: {user_prompt} | Result: {reply[:100]}")
    except Exception as e:
        speak("I couldn't load my vision model right now, Sean!", "Remu-chan 🌸")
        print(f"[Vision Error]: {e}")

def execute_local_action(user_text: str) -> bool:
    text_lower = user_text.lower()

    if "generate commit" in text_lower or "write commit" in text_lower:
        play_sound("success")
        speak("Looking at your git changes, Sean...", "Remu-chan 💻 (Git)")
        try:
            diff_cmd = subprocess.run(["git", "diff", "--staged"], capture_output=True, text=True)
            diff_text = diff_cmd.stdout.strip()
            if not diff_text:
                diff_cmd = subprocess.run(["git", "diff"], capture_output=True, text=True)
                diff_text = diff_cmd.stdout.strip()
            
            if not diff_text: speak("You don't have any code changes to commit right now!", "Remu-chan 🌸")
            else:
                prompt = f"Write a single, concise conventional commit message for these changes. Output ONLY the message, no quotes, no markdown: \n{diff_text[:2000]}"
                commit_msg = query_ollama_with_tools(prompt, "You are an expert developer.")
                pyperclip.copy(commit_msg)
                speak(f"Done! I copied the generated commit message to your clipboard.", "Remu-chan ✨ (Git)")
                store_memory(f"Generated Git Commit: {commit_msg}", "project")
        except Exception: speak("I ran into an issue checking your git diff, Sean!", "Remu-chan 🌸")
        return True

    if any(cmd in text_lower for cmd in ["pause music", "play music", "stop music", "resume music"]):
        play_sound("success"); keyboard.send("play/pause media")
        speak("Toggled your media playback, Sean!", "Remu-chan 🎵")
        return True
    if "mute" in text_lower and "unmute" not in text_lower:
        keyboard.send("volume mute"); speak("Muted the system volume for you.", "Remu-chan 🔇")
        return True
    if "unmute" in text_lower:
        keyboard.send("volume mute"); speak("Unmuted the system volume.", "Remu-chan 🔊")
        return True
    if "volume up" in text_lower:
        for _ in range(5): keyboard.send("volume up")
        speak("Turned the volume up!", "Remu-chan 🔊")
        return True
    if "volume down" in text_lower:
        for _ in range(5): keyboard.send("volume down")
        speak("Turned the volume down!", "Remu-chan 🔉")
        return True

    if text_lower.startswith("kill ") or text_lower.startswith("close process "):
        app_target = text_lower.replace("kill ", "").replace("close process ", "").strip()
        killed = False
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if app_target in proc.info['name'].lower():
                    psutil.Process(proc.info['pid']).terminate()
                    killed = True
            except Exception: pass
        if killed: speak(f"Terminated {app_target} for you, Sean!", "Remu-chan 💥")
        else: speak(f"I couldn't find a process named {app_target}.", "Remu-chan 🌸")
        return True

    if "look at my screen" in text_lower or "what's on my screen" in text_lower or "debug this" in text_lower:
        play_sound("success"); analyze_screen_vision(user_text)
        return True

    if "pomodoro" in text_lower: toggle_pomodoro(); return True
    elif "journal" in text_lower or "log" in text_lower: generate_daily_journal(); return True

    return False

def handle_input(user_text: str):
    if not user_text or len(user_text.strip()) < 2: return

    interrupt_speech()
    print(f"\n[Sean]: {user_text}")
    add_bond_xp(5)

    # Check Custom Voice Macros (`remu_macros.json`)
    macros = load_macros()
    if user_text.lower() in macros:
        user_text = macros[user_text.lower()]

    if execute_local_action(user_text): return

    speech_lower = user_text.lower()
    if any(phrase in speech_lower for phrase in ["remember to", "add task", "todo", "task:", "remind me"]):
        play_sound("success")
        cleanup_prompt = (f"Extract the core task or reminder from this sentence: '{user_text}'. "
                          "Strip out filler words. Keep it concise and clear under 8 words.")
        clean_task_text = query_ollama_with_tools(user_text, cleanup_prompt)
        tasks = load_tasks()
        new_task = {"id": len(tasks) + 1, "task": clean_task_text, "completed": False}
        tasks.append(new_task); save_tasks(tasks)
        store_memory(f"User Task: {clean_task_text}")
        add_bond_xp(15)
        speak(f"Got it, Sean! Added task: {clean_task_text}", "Remu-chan 📝 (Task Saved)")
        return

    if any(phrase in speech_lower for phrase in ["my tasks", "what are my tasks", "checklist", "todos"]):
        play_sound("success")
        tasks = load_tasks()
        active_tasks = [t for t in tasks if not t["completed"]]
        if not active_tasks: speak("You have zero pending tasks right now, Sean!", "Remu-chan 📝 (Checklist)")
        else:
            task_list_str = ", ".join([f"Task {t['id']}: {t['task']}" for t in active_tasks[:3]])
            speak(f"Here are your pending tasks, Sean: {task_list_str}", "Remu-chan 📝 (Checklist)")
        return

    if "summary" in speech_lower or "productivity" in speech_lower or "status" in speech_lower:
        play_sound("mood")
        work_mins = int((time.time() - session_start_time) / 60)
        recent_logs = recall_memories("Window Reaction", n_results=4, memory_target="core")
        summary_prompt = (f"You are Remu-chan (Bond Level {bond_level}). Sean's session duration: {work_mins} minutes. Recent activity: [{recent_logs}]. "
                          "Give a concise recap of how Sean is doing with deep affection. Under 25 words.")
        reply = query_ollama_with_tools("Give me a summary", summary_prompt)
        speak(reply, "Remu-chan 📊 (Summary)")
        store_memory(f"Command: Productivity Summary | Reply: {reply}")
        return

    context = recall_memories(user_text, memory_target="project")
    system_prompt = (f"You are Remu-chan, Sean's adorable desktop companion. Current Mood: [{current_mood}]. "
                     f"Bond Level: [{bond_level}]. The user's name is Sean. Always address him as Sean. "
                     f"Relevant RAG context: [{context}]. Answer Sean naturally using autonomous tools if needed, keeping under 25 words.")
    
    reply = query_ollama_with_tools(f"Sean said: {user_text}", system_prompt)
    store_memory(f"Sean: {user_text} | Remu: {reply}")
    speak(reply, f"Remu-chan 🛠️ ({current_mood.split()[0]})")

def listen_and_respond():
    if ui: ui.update_text("Calibrating & listening (VAD)...", "Remu-chan (o_o)🎙️")
    print("\n[System] Calibrating to room noise for VAD...")
    try:
        ambient_chunks = []
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32') as stream:
            for _ in range(1):
                chunk, _ = stream.read(int(0.5 * SAMPLE_RATE))
                ambient_chunks.append(np.linalg.norm(chunk))
        
        noise_floor = np.mean(ambient_chunks)
        dynamic_threshold = max(0.015, noise_floor * 2.5)
        print(f"[System] Listening now! (Threshold: {dynamic_threshold:.4f})")
        if ui: ui.update_text("Listening to your voice...", "Remu-chan (o_o)🎙️")

        chunk_duration = 0.5; silence_limit = 1.5; audio_chunks = []; silent_chunks = 0; max_chunks = 30

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32') as stream:
            for _ in range(max_chunks):
                audio_chunk, _ = stream.read(int(chunk_duration * SAMPLE_RATE))
                volume = np.linalg.norm(audio_chunk)
                if volume > dynamic_threshold:
                    audio_chunks.append(audio_chunk); silent_chunks = 0
                else:
                    if audio_chunks:
                        silent_chunks += 1; audio_chunks.append(audio_chunk)
                        if silent_chunks >= (silence_limit / chunk_duration): break
                    else: continue

        if audio_chunks:
            full_audio = np.concatenate(audio_chunks, axis=0).flatten()
            segments, _ = whisper_model.transcribe(full_audio, beam_size=1)
            user_speech = " ".join([seg.text for seg in segments]).strip()
            handle_input(user_speech)
        else:
            if ui: ui.update_text("No voice detected.", "Remu-chan 🌸")
    except Exception as e: print(f"[VAD Mic Error]: {e}")

def wake_word_listener_loop():
    print("[System] Wake word listener active ('Hey Remu')...")
    while True:
        try:
            audio_data = sd.rec(int(3 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait()
            segments, _ = whisper_model.transcribe(audio_data.flatten(), beam_size=1)
            speech = " ".join([seg.text for seg in segments]).strip().lower()
            if "remu" in speech:
                print(f"[Wake Word Detected]: {speech}")
                play_sound("success"); speak("Yes, Sean? I'm listening!", "Remu-chan (o_o)🎙️")
                listen_and_respond()
        except Exception: time.sleep(1)

def terminal_input_loop():
    while True:
        try:
            user_input = input()
            if user_input.strip(): handle_input(user_input)
        except Exception: break

def quick_action_thread():
    try:
        play_sound("success"); keyboard.send('ctrl+c'); time.sleep(0.3)
        selected_text = pyperclip.paste()
        if selected_text and len(selected_text.strip()) > 0:
            speak("Analyzing that for you, Sean...", "Remu-chan 🔍 (Analysis)")
            prompt = f"Briefly explain or improve this snippet under 30 words: {selected_text[:1000]}"
            reply = query_ollama_with_tools(prompt, "You are Remu-chan, Sean's expert copilot.")
            speak(reply, "Remu-chan 💡 (Insight)")
    except Exception as e: print(f"Quick Action Error: {e}")

def trigger_quick_action(): threading.Thread(target=quick_action_thread, daemon=True).start()

def daemon_loop():
    startup_briefing()
    last_window = ""
    night_greeting_done = False
    
    keyboard.add_hotkey('ctrl+space', listen_and_respond)
    keyboard.add_hotkey('esc', interrupt_speech)
    keyboard.add_hotkey('ctrl+alt+e', trigger_quick_action)
    keyboard.add_hotkey('alt+space', trigger_spotlight)

    while True:
        hour = datetime.now().hour
        if (hour >= 20 or hour < 6) and not night_greeting_done:
            speak("It's getting late, Sean! Your intermittent fasting window should be starting soon. Don't forget to wrap up your coding and let your system rest.", "Remu-chan 🌙 (Night Mode)")
            night_greeting_done = True
        elif 6 <= hour < 20: night_greeting_done = False

        current_window = get_active_window()
        is_distracted = any(kw in current_window.lower() for kw in distraction_keywords)

        if current_window != last_window and current_window != "Unknown" and is_distracted and not pomodoro_active:
            time.sleep(2.0)
            if get_active_window() == current_window:
                update_mood(True)
                context = recall_memories(current_window, memory_target="core")
                system_prop = (f"You are Remu-chan. Current Mood: [{current_mood}]. The user's name is Sean. Always address him as Sean. "
                               f"Past context: [{context}]. Active window: {current_window}. Give Sean a cute pouting toddler nudge because he got distracted. Keep under 18 words.")
                reaction = query_ollama_with_tools(f"Distracted window: {current_window}", system_prop)
                store_memory(f"Distraction Window: {current_window} | Reaction: {reaction}")
                speak(reaction, f"Remu-chan (>_<)💢 (Sassy)")
                last_window = current_window
        time.sleep(10.0)

def create_tray_icon():
    image = Image.new('RGB', (64, 64), color=(59, 130, 246))
    dc = ImageDraw.Draw(image)
    dc.rectangle((16, 16, 48, 48), fill=(26, 26, 28))
    menu = (item('Open Task Checklist', lambda icon, item: open_task_window()),
            item('Settings', lambda icon, item: open_settings_window()),
            item('Quit Remu-chan', lambda icon, item: (icon.stop(), os._exit(0))))
    pystray.Icon("Remu-chan", image, "Remu-chan 🌸", menu).run()

if __name__ == "__main__":
    threading.Thread(target=index_workspace_code, daemon=True).start()
    threading.Thread(target=daemon_loop, daemon=True).start()
    threading.Thread(target=pomodoro_timer_loop, daemon=True).start()
    threading.Thread(target=git_health_monitor_loop, daemon=True).start()
    threading.Thread(target=workflow_copilot_loop, daemon=True).start()
    threading.Thread(target=clipboard_monitor_loop, daemon=True).start()
    threading.Thread(target=wake_word_listener_loop, daemon=True).start()
    threading.Thread(target=terminal_input_loop, daemon=True).start()
    threading.Thread(target=create_tray_icon, daemon=True).start()

    ui = RemuWidget()
    ui.run()