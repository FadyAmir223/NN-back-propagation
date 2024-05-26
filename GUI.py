import tkinter as tk
from tkinter import ttk
from tkinter import *
import code as c


Window = tk.Tk()
Window.title('Task 3')
Window.geometry('450x450')

############################# Neurons ######################################

ttk.Label(Window, text="Hidden Layer Neurons :", font=("Times New Roman", 10)).grid(column=0, row=1, padx=10, pady=25)
neuronList = tk.StringVar()
neuron_entry = tk.Entry(Window, textvariable=neuronList, font=('Times New Roman', 10, 'normal')).grid(row=1, column=1)

############################# learning rate ###############################

ttk.Label(Window, text="Learning Rate :", font=("Times New Roman", 10)).grid(column=0, row=2, padx=10, pady=25)
eta = tk.DoubleVar()
rate_entry = tk.Entry(Window, textvariable=eta, font=('Times New Roman', 10, 'normal')).grid(row=2, column=1)

############################# epochs #######################################

ttk.Label(Window, text="Epochs :", font=("Times New Roman", 10)).grid(column=0, row=3, padx=10, pady=25)
epochs = tk.IntVar()
epoch_entry = tk.Entry(Window, textvariable=epochs, font=('Times New Roman', 10, 'normal')).grid(row=3, column=1)

############################# bias #########################################

bias = IntVar()
Button_bias = Checkbutton(Window, text="Add Bias ", variable=bias, onvalue=1, offvalue=0, height=2, width=10).grid(row=5, column=1)

############################# function #####################################

ttk.Label(Window, text="Function :", font=("Times New Roman", 10)).grid(column=0, row=4, padx=10, pady=25)
fn = tk.StringVar()
fnc = ttk.Combobox(Window, width=27, textvariable=fn)
fnc['values'] = ('Sigmoid', 'Hyperbolic Tangent')
fnc.grid(column=1, row=4)
fnc.current()

############################# trigger ######################################


def run():
    neuron = neuronList.get().split(',')
    neurons = [int(i) for i in neuron]
    obj = c.BackPropagation(eta.get(), epochs.get(), bias.get(), neurons, fn.get(), c.read_file())
    obj.train()
    label_text = tk.StringVar()
    label = tk.Label(Window, textvariable=label_text)
    label_text.set('Accuracy = ' + str(obj.test()))
    label.config(font=("Times New Roman", 15))
    label.grid(row=11, column=1)

btn1 = Button(Window, text='Run', padx=14, pady=7, command=run).grid(row=7, column=1)

############################################################################

mainloop()
