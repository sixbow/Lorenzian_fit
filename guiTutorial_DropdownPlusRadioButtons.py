import tkinter as tk
from tkinter import ttk
my_window = tk.Tk()
my_window.geometry('200x100')

def callbackFunc(event):
    print("Nieuw format geselecteerd | {}".format(my_combo.get()))

numberoffiles = [("Single file", 101),
   	     ("Loop over multiple files", 102),]

def ShowChoice():
    print(value_loopcondition.get())

value_loopcondition = tk.IntVar()
value_loopcondition.set(1)  # initializing the radio button choice single shot, Loop over multiple files

labelTop = tk.Label(my_window, text = "Choose input program format")
labelTop.grid(column = 0 , row = 0)

my_combo = ttk.Combobox(my_window, values= ["Sonnet",
                                            "ADS"
                                            ,"MW-studio"])
my_combo.current(1)
#my_combo.grid(column = 0, row = 1)

my_combo.bind("<<ComboboxSelected>>", callbackFunc)

for language, val in numberoffiles:
    tk.Radiobutton(my_window, 
                   text=language,
                   padx = 20, 
                   variable=value_loopcondition, 
                   command=ShowChoice,
                   value=val).grid(column = 0 , row = 1)
my_window.mainloop()







# list = Listbox(my_window, selectmode = "multiple")

# list.pack(expand = YES, fill = "both")

# my_options = ["Sonnet", "ADS", "MW-studio"]
# for each_item in range(len(my_options)):
#     list.insert(END, my_options[each_item])

#     list.itemconfig(each_item, bg = "yellow" if each_item % 2 == 0 else "cyan")


# my_window.mainloop()

