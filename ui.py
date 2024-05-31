import tkinter as tk
from tkinter import font as tkFont
from tkinter import ttk
import threading


class UI(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.root = None
        self.fig = None
        self.ax = None
        self.simFrame = None
        self.canvas = None
        self.textbox = None
        self.buttonFrame = None
        self.simRunning = None
        self.btn1 = None
        self.btn2 = None
        self.textboxOut = None
        self.textboxLabel = None
        self.textboxOutLabel = None
        self.textFrame = None
        self.textFrameOut = None
        self.scrollTextBox = None
        self.scrollTextBoxOut = None
        self.text = None
        self.taskReady = False
        self.conversationReady = False
        self.btn3 = None
        self.infoText = None
        self.btn4 = None
        self.exit = None
        
        self.textboxTaskFrame = None
        self.textboxTask = None
        self.textboxTaskLabel = None
        self.textboxTaskScroll = None
        self.textboxTaskText = ""
        
        self.textboxConversationFrame = None
        self.textboxConversation = None
        self.textboxConversationLabel = None
        self.textboxConversationScroll = None
        self.textboxConversationText = ""
        

        self.start()

    def callback(self):
        self.root.quit()

    def run(self):
        self.root = tk.Tk()
        tkFont.nametofont('TkDefaultFont').configure(size=20)
        self.root.protocol("WM_DELETE_WINDOW", self.callback)
        self.root.geometry("800x800")
        self.root.title("UI")
        #helv36 = tkFont.Font(family='Helvetica', size=36, weight='bold')
        # ------------------ Text at the top Textbox ------------------
        #self.infoText = tk.Label(self.root, text="Please start tasks with the keyword \"TASK:\".", padx=20, pady=5)
        #self.infoText.pack()
        # ------------------ Input Textbox ------------------
        self.textFrame = tk.Frame(self.root)
        self.textboxLabel = tk.Label(self.textFrame, text= "User Input:",padx=10, pady=10)
        self.textbox = tk.Text(self.textFrame, height=4,width=50,padx=10,pady=10)
        self.textbox.bind("<KeyPress>", self.shortcut)
        self.scrollTextBox = tk.ttk.Scrollbar(self.textFrame, orient='vertical', command=self.textbox.yview)
        self.textbox['yscrollcommand'] = self.scrollTextBox.set
        self.textboxLabel.grid(column=0,row=0)
        self.textbox.grid(column=1, row=0, sticky='nwes')
        self.scrollTextBox.grid(column=2, row=0, sticky='ns')
        self.textFrame.grid_columnconfigure(0, weight=1)
        self.textFrame.grid_rowconfigure(0, weight=1)
        self.textFrame.pack(padx=10, pady=20)

        # ------------------ Last Input Textbox ------------------
        self.textFrameOut = tk.Frame(self.root)
        self.textboxOutLabel = tk.Label(self.textFrameOut, text="Last Input:", padx=10,pady=10)
        self.textboxOut = tk.Text(self.textFrameOut, height=4, width=50, padx=10, pady=10)
        self.scrollTextBoxOut = tk.ttk.Scrollbar(self.textFrameOut, orient='vertical', command=self.textboxOut.yview)
        self.textboxOut['yscrollcommand'] = self.scrollTextBoxOut.set
        self.textboxOutLabel.grid(column=0, row=0)
        self.textboxOut.grid(column=1, row=0, sticky='nwes')
        self.scrollTextBoxOut.grid(column=2, row=0, sticky='ns')
        self.textFrameOut.grid_columnconfigure(0, weight=1)
        self.textFrameOut.grid_rowconfigure(0, weight=1)
        self.textFrameOut.pack(padx=10, pady=20)

        #------------------ Buttons ------------------
        self.buttonFrame = tk.Frame(self.root)
        self.buttonFrame.columnconfigure(0, weight=1)
        self.buttonFrame.columnconfigure(1, weight=1)
        self.simRunning = True
        self.btn1 = tk.Button(self.buttonFrame, text="Pause Simulation", command=self.toggleSimulation)
        self.btn1.grid(row=0, column=0, sticky=tk.W+tk.E)
        #self.btn1["font"] =helv36

        self.text = ""
        self.btn2 = tk.Button(self.buttonFrame, text="Send Task", command=self.sendTask)
        self.btn2.grid(row=0, column=1, sticky=tk.W+tk.E)

        self.btn3 = tk.Button(self.buttonFrame, text="Send Question", command=self.sendQuestion)
        self.btn3.grid(row=1, column=1, sticky=tk.W + tk.E)
        
        self.btn4 = tk.Button(self.buttonFrame, text="Exit Program", command=self.exitProgram)
        self.btn4.grid(row=1, column=0, sticky=tk.W + tk.E)

        self.buttonFrame.pack(fill='x')
        
        #------------------- llm output textbox sizes -------------------
        llmOutHeight = 15
        llmOutWidth = 125
        
        # ------------------ Task text field ------------------
        self.textboxTaskFrame = tk.Frame(self.root)
        self.textboxTaskLabel = tk.Label(self.textboxTaskFrame, text="Task LLM \n     Output:     ", padx=11,pady=10)
        self.textboxTask = tk.Text(self.textboxTaskFrame, height=llmOutHeight, width=llmOutWidth, padx=10, pady=10)
        self.textboxTaskScroll = tk.ttk.Scrollbar(self.textboxTaskFrame, orient='vertical', command=self.textboxTask.yview)
        self.textboxTask['yscrollcommand'] = self.textboxTaskScroll.set
        self.textboxTaskLabel.grid(column=0, row=0)
        self.textboxTask.grid(column=1, row=0, sticky='nwes')
        self.textboxTaskScroll.grid(column=2, row=0, sticky='ns')
        self.textboxTaskFrame.grid_columnconfigure(0, weight=1)
        self.textboxTaskFrame.grid_rowconfigure(0, weight=1)
        self.textboxTaskFrame.pack(padx=10, pady=20)
        
        # ------------------ Conversation text field ------------------
        self.textboxConversationFrame = tk.Frame(self.root)
        self.textboxConversationLabel = tk.Label(self.textboxConversationFrame, text="Conversation \nLLM Output:", padx=10,pady=10)
        self.textboxConversation = tk.Text(self.textboxConversationFrame, height=llmOutHeight, width=llmOutWidth, padx=10, pady=10)
        self.textboxConversationScroll = tk.ttk.Scrollbar(self.textboxConversationFrame, orient='vertical', command=self.textboxConversation.yview)
        self.textboxConversation['yscrollcommand'] = self.textboxConversationScroll.set
        self.textboxConversationLabel.grid(column=0, row=0)
        self.textboxConversation.grid(column=1, row=0, sticky='nwes')
        self.textboxConversationScroll.grid(column=2, row=0, sticky='ns')
        self.textboxConversationFrame.grid_columnconfigure(0, weight=1)
        self.textboxConversationFrame.grid_rowconfigure(0, weight=1)
        self.textboxConversationFrame.pack(padx=10, pady=20)
        
        # ------------------ End of setup ------------------
        self.root.mainloop()

    def toggleSimulation(self):
        if self.simRunning is False:
            self.simRunning = True
            self.btn1.config(text="Pause Simulation")
        else:
            self.simRunning = False
            self.btn1.config(text="Start Simulation")

    def updateOutbox(self):
        self.textboxOut.configure(state='normal')
        self.textboxOut.delete(1.0, tk.END)
        self.textboxOut.insert(1.0, self.text)
        self.textboxOut.configure(state='disabled')

    def sendTask(self):
        self.text = "Task: " + self.textbox.get(1.0, tk.END)
        self.taskReady = True
        self.clearTextbox()
        self.updateOutbox()

    def sendQuestion(self):
        self.text = "Conversation: " + self.textbox.get(1.0, tk.END)
        self.conversationReady = True
        self.clearTextbox()
        self.updateOutbox()

    def clearTextbox(self):
        self.textbox.delete(1.0, tk.END)

    def shortcut(self, event):
        if event.char == "\r" and event.keysym == "Return" and event.keycode == 13:
            self.sendQuestion()
            self.root.after(0, self.clearTextbox)

    def exitProgram(self):
        self.exit = True
        self.root.destroy()
    
    def updateTaskTextbox(self,newText):
        self.textboxTaskText += newText
        self.textboxTask.configure(state='normal')
        self.textboxTask.delete(1.0, tk.END)
        self.textboxTask.insert(1.0, self.textboxTaskText)
        self.textboxTask.see(tk.END)
        self.textboxTask.configure(state='disabled')
        
    def updateConversationTextbox(self,newText):
        self.textboxConversationText += newText
        self.textboxConversation.configure(state='normal')
        self.textboxConversation.delete(1.0, tk.END)
        self.textboxConversation.insert(1.0, self.textboxConversationText)
        self.textboxConversation.see(tk.END)
        self.textboxConversation.configure(state='disabled')
        