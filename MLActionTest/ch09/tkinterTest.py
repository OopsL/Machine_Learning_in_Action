import tkinter
import numpy as np
from ch09 import regTree


# root = tkinter.Tk()
# myLabel = tkinter.Label(root, text ="Hello World")
# myLabel.grid()
# root.mainloop()

def reDraw(tolS,tolN):
    pass

def drawNewTree():
    pass

root = tkinter.Tk()
tkinter.Label(root,text="Plot Place Holder").grid(row=0,columnspan=3)
tkinter.Label(root,text="tolN").grid(row=1,column=0)
tolNentry = tkinter.Entry(root)
tolNentry.grid(row=1,column=1)
tolNentry.insert(0,'10')
tkinter.Label(root,text="tolS").grid(row=2,column=0)
tolSentry = tkinter.Entry(root)
tolSentry.grid(row=2,column=1)
tolSentry.insert(0,'10')
tkinter.Button(root,text="ReDraw",command=drawNewTree).grid(row=1,column=2,rowspan=3)
chkBtnVar = tkinter.IntVar()
chkBtnVar = tkinter.Checkbutton(root,text='Model Tree',variable=chkBtnVar)
chkBtnVar.grid(row=3,column=0,columnspan=2)

reDraw.rawDat = np.mat(regTree.loadDataSet('/Users/JD.K/Downloads/machinelearninginaction/Ch09/sine.txt'))
reDraw.testDat = np.arange(min(reDraw.rawDat[:,0]),max(reDraw.rawDat[:,0]),0.01)
reDraw(1.0,10)
root.mainloop()

















