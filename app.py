
import tkinter as tk
from tkinter import messagebox
import subprocess
import sys
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent
DETECTION_FILE = PROJECT_DIR / "src" / "final_drowsiness_system.py"

process = None


# Colors
BG = "#101827"
CARD = "#1e293b"
ACCENT = "#38bdf8"
GREEN = "#22c55e"
RED = "#ef4444"
WHITE = "#f8fafc"
GRAY = "#94a3b8"


def start_detection():
    global process

    if process is not None and process.poll() is None:
        messagebox.showinfo(
            "Already Running",
            "The detection system is already running."
        )
        return

    if not DETECTION_FILE.exists():
        messagebox.showerror(
            "File Missing",
            "final_drowsiness_system.py was not found in the src folder."
        )
        return

    try:
        process = subprocess.Popen(
            [sys.executable, str(DETECTION_FILE)],
            cwd=str(PROJECT_DIR)
        )

        status_label.config(
            text="● Detection Running",
            fg=GREEN
        )

        start_button.config(state="disabled")
        stop_button.config(state="normal")

        root.after(1000, check_process)

    except Exception as error:
        messagebox.showerror("Error", str(error))


def check_process():
    global process

    if process is not None and process.poll() is None:
        root.after(1000, check_process)
    else:
        status_label.config(
            text="● Detection Stopped",
            fg=RED
        )
        start_button.config(state="normal")
        stop_button.config(state="disabled")


def stop_detection():
    global process

    if process is not None and process.poll() is None:
        process.terminate()

        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()

    status_label.config(
        text="● Detection Stopped",
        fg=RED
    )

    start_button.config(state="normal")
    stop_button.config(state="disabled")


def close_application():
    if process is not None and process.poll() is None:
        process.terminate()

        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()

    root.destroy()


# Main window
root = tk.Tk()
root.title("AI Driver Drowsiness Detection System")
root.geometry("850x620")
root.configure(bg=BG)
root.resizable(False, False)

# Header
header = tk.Frame(root, bg=BG)
header.pack(fill="x", pady=(35, 10))

title = tk.Label(
    header,
    text="DRIVER DROWSINESS DETECTION",
    font=("Arial", 23, "bold"),
    bg=BG,
    fg=WHITE
)
title.pack()

subtitle = tk.Label(
    header,
    text="AI-Powered Driver Monitoring System",
    font=("Arial", 12),
    bg=BG,
    fg=ACCENT
)
subtitle.pack(pady=8)

# Main card
card = tk.Frame(
    root,
    bg=CARD,
    padx=35,
    pady=30
)
card.pack(padx=55, pady=20, fill="both")

section_title = tk.Label(
    card,
    text="SYSTEM CONTROL PANEL",
    font=("Arial", 16, "bold"),
    bg=CARD,
    fg=WHITE
)
section_title.pack(pady=(0, 15))

description = tk.Label(
    card,
    text="Monitor eye closure, facial movements and yawning\nusing computer vision and trained CNN models.",
    font=("Arial", 11),
    bg=CARD,
    fg=GRAY,
    justify="center"
)
description.pack(pady=10)

# Start button
start_button = tk.Button(
    card,
    text="START DETECTION",
    font=("Arial", 12, "bold"),
    bg=GREEN,
    fg="white",
    activebackground="#16a34a",
    activeforeground="white",
    width=24,
    height=2,
    relief="flat",
    cursor="hand2",
    command=start_detection
)
start_button.pack(pady=(20, 10))

# Stop button
stop_button = tk.Button(
    card,
    text="STOP DETECTION",
    font=("Arial", 12, "bold"),
    bg=RED,
    fg="white",
    activebackground="#dc2626",
    activeforeground="white",
    width=24,
    height=2,
    relief="flat",
    cursor="hand2",
    state="disabled",
    command=stop_detection
)
stop_button.pack(pady=10)

# Status
status_label = tk.Label(
    card,
    text="● Detection Stopped",
    font=("Arial", 12, "bold"),
    bg=CARD,
    fg=RED
)
status_label.pack(pady=20)

# Features section
features_frame = tk.Frame(root, bg=BG)
features_frame.pack(pady=15)

features_title = tk.Label(
    features_frame,
    text="SYSTEM FEATURES",
    font=("Arial", 13, "bold"),
    bg=BG,
    fg=WHITE
)
features_title.pack(pady=10)

features = [
    "Face Detection",
    "Eye Closure Monitoring",
    "Yawn Detection",
    "EAR and MAR Calculation",
    "Drowsiness Alarm"
]

feature_text = tk.Label(
    features_frame,
    text="   |   ".join(features),
    font=("Arial", 9),
    bg=BG,
    fg=ACCENT
)
feature_text.pack()

# Footer
footer = tk.Label(
    root,
    text="AI-Based Driver Safety Monitoring | Academic Project",
    font=("Arial", 9),
    bg=BG,
    fg=GRAY
)
footer.pack(side="bottom", pady=15)

# Close event
root.protocol("WM_DELETE_WINDOW", close_application)

root.mainloop()
