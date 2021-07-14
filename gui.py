import os
import tkinter as tk
from tkinter import *

foregroundColor = "#eee"
backgroundColor = "#555"
root= tk.Tk()
root.configure(background='red')
root.title("Распознавание нарушений")
canvas1 = tk.Canvas(root, width = 300, height = 300, bg = 'gray90', relief = 'raised')
canvas1.pack()

def myCmd ():
    os.system('cmd /k "python test.py"')

def myCmd_photo ():
    os.system('cmd /k "python photo_train_test.py"')

def quit(self):
	self.root.destroy()
     
button1 = tk.Button(text='     Load Image  ', command=myCmd_photo, bg='green', fg='white', font=('helvetica', 12, 'bold'))
button2 = tk.Button(text='     Load Video ', command=myCmd, bg='green', fg='white', font=('helvetica', 12, 'bold'))
canvas1.create_window(150, 100, window=button1 and button2) # and button2


root.mainloop()