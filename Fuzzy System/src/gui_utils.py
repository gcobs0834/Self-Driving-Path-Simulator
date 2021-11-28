import tkinter as tk


def add_text(frame, row, label, text):
    l = tk.Label(frame, width= 25,font=("Courier", 15))
    l["text"] = label
    l.grid(row=row, column=0, sticky=tk.N+tk.W)
    t = tk.Label(frame,width= 25,font=("Courier", 15))
    t["text"] = text
    t.grid(row=row, column=1, sticky=tk.N+tk.W)
    return l, t


def add_button(frame, row, label, text, func):
    l = tk.Label(frame, width= 25,font=("Courier", 15))
    l["text"] = label
    l.grid(row=row, column=0, sticky=tk.N+tk.W)
    b = tk.Button(frame,width= 25,font=("Courier", 15))
    b["text"] = text
    b.grid(row=row, column=1, sticky=tk.N+tk.W)
    b["command"] = func
    return l, b

