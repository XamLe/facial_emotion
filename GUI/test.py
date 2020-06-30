import tkinter as tk

root=tk.Tk()

top_frame = tk.Frame(root, bg='cyan', width=450, height=50, pady=3)
center = tk.Frame(root, bg='gray2', width=50, height=40, padx=3, pady=3)
top_frame.grid(row=0, sticky="ew")
center.grid(row=1, sticky="nsew")

root.mainloop()