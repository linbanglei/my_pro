
import GUI
import tkinter as tk


def main():
    root=tk.Tk()
    root.geometry('1400x800+200+70')
    app=GUI.Application(master=root)
    app.mainloop()

if __name__=='__main__':
    main()