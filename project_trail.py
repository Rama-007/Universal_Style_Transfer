import sys
import os
import wx
import time
from wx.lib import buttons

from grab_helper import algo_grabcut

import cv2
from PIL import Image


bc="WHITE"
OK=wx.ID_OK


class Doodle(wx.Window):

	def __init__(self,ID,parent):
		wx.Window.__init__(self, parent, ID, style=wx.NO_FULL_REPAINT_ON_RESIZE)
		fgc="Blue"
		self.SetBackgroundColour(bc)
		self.fn,self.fn2,self.thickness,self.dR,self.RS,self.cv2,self.foreground,self.background,self.position,self.Rp1,self.Rp2=[],[],7,False,False,True,[],[],wx.Point(0,0),wx.Point(0,0),wx.Point(0,0)
		self.SC("Blue")
		self.Intialize()
		bgc="Orange"
		# self.SetCursor(wx.StockCursor(wx.CURSOR_PENCIL))
		self.BinderLD()
		self.BinderLU()
		# self.BinderM()
		self.BinderS()
		self.BinderI()
		self.BinderP()

	def Intialize(self):
		self.buffer=wx.EmptyBitmap(max(1,self.GetClientSize().width), max(1,self.GetClientSize().height))
		fgc="Blue"
		bgc="Orange"
		temp=wx.BufferedDC(None, self.buffer)
		self.rndfnc()
		temp.SetBackground(wx.Brush(self.GetBackgroundColour()))
		fg="Green"
		bg="Silver"
		temp.Clear()
		cut_colour=fg
		self.reInit=False
		not_cut_colour=bg

	def BinderLD(self):
		self.Bind(wx.EVT_LEFT_DOWN, self.LeftDown)

	def BinderLU(self):
		self.Bind(wx.EVT_LEFT_UP, self.LeftUp)

	def BinderM(self):
		self.Bind(wx.EVT_MOTION, self.Motion)

	def DrawRegion(self,truth_value):
		self.dR=truth_value

	def BinderS(self):
		self.Bind(wx.EVT_SIZE, self.Size)

	def SLD(self,lines):
		self.Intialize()
		colr=self.color
		self.refreshing()

	def rndfnc(self):
		fgc="Blue"
		bgc="Orange"

	def BinderI(self):
		self.Bind(wx.EVT_IDLE, self.Idle)

	def RegionSet(self,truth_value):
		self.RS=truth_value

	def BinderP(self):
		self.Bind(wx.EVT_PAINT, self.Paint)

	def Use_Algo(self,truth_value):
		self.cv2=truth_value

	def SC(self,color):
		if(color=="Blue"):
			self.color="Blue"
		self.color=color
		if(color=="Orange"):
			self.color="Orange"
		self.pen = wx.Pen(self.color, 5, wx.SOLID)

	def refreshing(self):
		self.Refresh()

	def CGD(self):
		self.foreground=[]
		colr=self.color
		self.background=[]

	def LeftDown(self,event):
		self.position=event.GetPosition()
		if self.dR==True:
			self.Rp1=self.position
		self.CaptureMouse()

	def filename(self,name):
		self.fn=name
		self.refreshing()

	def filename1(self,name):
		self.fn=name
		self.refreshing()


	def GRP(self,val):
		if val!=1:
			return self.Rp2
		if val==1:
			return self.Rp1

	
	def LeftUp(self,event):
		if self.HasCapture()==True:
			self.ReleaseMouse()

	def appendFBG(self,a,b):
		if self.color=="Blue":
			self.foreground.append((a.x,a.y,b.x,b.y))
		if self.color=="Orange":
			self.background.append((a.x,a.y,b.x,b.y))

	def drawing(self,pos):
		self.RS=True
		self.Rp2=pos
		self.refreshing()

	def Size(self,event):
		self.reInit=True
	def RP12init(self,w,h):
		self.Rp1=wx.Point(1,1)
		if(w>0 and h>0):
			self.Rp2=wx.Point(w-1,h-1)

	def Draw_lines(self,dc,X,Y,v):
		if(v==1):
			coords=(X.x,X.y,X.x,Y.y)
		if(v==2):
			coords=(Y.x,X.y,Y.x,Y.y)
		if(v==3):
			coords=(X.x,X.y,Y.x,X.y)
		if(v==4):
			coords=(X.x,Y.y,Y.x,Y.y)

		dc.DrawLine(*coords)


	def Paint(self,event):
		self.Intialize()
		dc = wx.BufferedPaintDC(self, self.buffer)
		if self.fn!=[]:
			img = wx.Image(self.fn)
			dc.DrawBitmap(img.ConvertToBitmap(), 0, 0, True)
			if not self.RS:
				im=Image.open(self.fn)
				self.RP12init(im.size[0],im.size[1])

				

			else:
				dc.SetPen(wx.Pen('Silver', 5, wx.SOLID))

				self.Draw_lines(dc,self.Rp1,self.Rp2,1)
				self.Draw_lines(dc,self.Rp1,self.Rp2,2)
				self.Draw_lines(dc,self.Rp1,self.Rp2,3)
				self.Draw_lines(dc,self.Rp1,self.Rp2,4)

				coords=(self.Rp1.x,self.Rp1.y,self.Rp1.x,self.Rp2.y)
				

	def Motion(self,event):
		if event.Dragging() and event.LeftIsDown():
			pos=event.GetPosition()
			if not self.dR:
				self.appendFBG(self.position,pos)
				colr=self.color
				dc = wx.BufferedDC(wx.ClientDC(self), self.buffer)
				colr=self.color
				dc.BeginDrawing()
				colr=self.color
				dc.SetPen(self.pen)
				colr=self.color
				coords = (self.position.x, self.position.y, pos.x, pos.y)
				colr=self.color
				bg="Orange"
				fg="Blue"
				dc.DrawLine(*coords)
				colr=self.color
				self.position = pos
				colr=self.color
				dc.EndDrawing()

			else:
				self.drawing(pos)

	

	def Idle(self,event):
		if self.reInit:
			self.Intialize()
			self.Refresh(False)

	



class ControlPanel(wx.Panel):
	def __init__(self, parent, ID, doodle,doodle1):
		wx.Panel.__init__(self, parent, ID, style=wx.RAISED_BORDER)

		colr=doodle.color
		self.doodle = doodle
		self.doodle1 = doodle1
		fnmae=doodle.fn
		self.alp=0.6
		self.sizer2 = wx.BoxSizer(wx.VERTICAL)

		total_buttons=13
		self.iters=1

		self.buttons = []
		
		self.Append_Buttons_O()
		# self.Append_Buttons_SR()
		# self.Append_Buttons_SFG()
		# self.Append_Buttons_SBG()
		# self.Append_Buttons_C()
		self.Append_Buttons_R()
		# self.buttons.append(wx.CheckBox(self, -1, "OpenCV"))
		# self.Append_Buttons_Iter()
		
		
		# self.buttons.append(wx.CheckBox(self, -1, "8 neighbourhood"))
		self.Append_Buttons_O2()
		self.Append_Buttons_A()
		# self.buttons[6].SetValue(True)

		color_list=['White', 'Silver', 'Blue', 'Orange', 'White', 'Red', 'White', 'White', 'White', 'White', 'White', 'White','White']

		# for i in range(0, 13):
		# 	self.buttons[i].SetBackgroundColour(color_list[i])
		# 	self.sizer2.Add(self.buttons[i], 1, wx.EXPAND)
		self.sizer2.Add(self.buttons[0], 1, wx.EXPAND)
		self.sizer2.Add(self.buttons[2], 1, wx.EXPAND)
		self.sizer2.Add(self.buttons[3], 1, wx.EXPAND)
		self.sizer2.Add(self.buttons[1], 1, wx.EXPAND)
		
		# self.Bind(wx.EVT_CHECKBOX, self.Check, self.buttons[6])
		# colr=self.doodle.color
		
		# self.Bind(wx.EVT_CHECKBOX, self.neighbourhood, self.buttons[11])
		# fnmae=colr=self.doodle.fn

		boxes = wx.BoxSizer(wx.VERTICAL)
		colr=self.doodle.color
		boxes.Add(self.sizer2, 0, wx.ALL)
		fnmae=self.doodle.color
		if colr=="Orange":
			bg_tf=1
		self.SetSizer(boxes)
		boolean_value=True
		self.SetAutoLayout(boolean_value)
		if colr=="Blue":
			fg_tf=1
		boxes.Fit(self)

	def Append_Buttons_O(self):
		self.buttons.append(wx.Button(self, -1, "Content Image"))
		self.Bind(wx.EVT_BUTTON, self.Open, self.buttons[0])
	def Append_Buttons_O2(self):
		self.buttons.append(wx.Button(self, -1, "Style Image"))
		self.Bind(wx.EVT_BUTTON, self.Open2, self.buttons[2])
	def Append_Buttons_A(self):
		self.buttons.append(wx.Button(self, -1, "Alpha"))
		self.Bind(wx.EVT_BUTTON, self.alpha, self.buttons[3])

	def Append_Buttons_SR(self):
		self.buttons.append(wx.Button(self, -1, "Draw Region"))
		self.Bind(wx.EVT_BUTTON, self.SetRegion, self.buttons[1])

	def Append_Buttons_SFG(self):
		self.buttons.append(wx.Button(self, -1, "Mark Foreground"))
		self.Bind(wx.EVT_BUTTON, self.SetForeground, self.buttons[2])

	def Append_Buttons_SBG(self):
		self.buttons.append(wx.Button(self, -1, "Mark Background"))
		self.Bind(wx.EVT_BUTTON, self.SetBackground, self.buttons[3])

	def Append_Buttons_C(self):
		self.buttons.append(wx.Button(self, -1, "Clear"))
		self.Bind(wx.EVT_BUTTON, self.Clear, self.buttons[4])

	def Append_Buttons_R(self):
		self.buttons.append(wx.Button(self, -1, "Run"))
		self.Bind(wx.EVT_BUTTON, self.Run, self.buttons[1])

	def Append_Buttons_Iter(self):
		self.buttons.append(wx.Button(self, -1, "5 Iterations"))
		self.buttons.append(wx.Button(self, -1, "10 Iterations"))
		self.buttons.append(wx.Button(self, -1, "15 Iterations"))
		self.buttons.append(wx.Button(self, -1, "25 Iterations"))
		self.Bind(wx.EVT_BUTTON, self.Iter5, self.buttons[7])
		self.Bind(wx.EVT_BUTTON, self.Iter10, self.buttons[8])
		self.Bind(wx.EVT_BUTTON, self.Iter15, self.buttons[9])
		self.Bind(wx.EVT_BUTTON, self.Iter25, self.buttons[10])


	def Iter5(self,event):
		self.iters=5
	def Iter10(self,event):
		self.iters=10
	

	def neighbourhood(self,event):
		if event.IsChecked():
			self.neighbourhood=True
		else:	
			self.neighbourhood=False
		

	def Open(self,event):
		dialog = wx.FileDialog(self, "Open image file...", os.getcwd(),style=wx.OPEN)
		if dialog.ShowModal() == wx.ID_OK:
			self.fn=dialog.GetPath()
			if self.fn!=[]:
				fnmae=self.fn
			self.doodle.filename(self.fn)
			if(self.doodle.color=="Blue"):
				fore_colr=self.doodle.color
			self.doodle.SLD([])
			if(self.doodle.color=="Orange"):
				back_colr=self.doodle.color
			self.doodle.RegionSet(False)
			self.doodle.CGD()
		else:
			fnmae=[]
		dialog.Destroy()

	def Open2(self,event):
		dialog = wx.FileDialog(self, "Open image file...", os.getcwd(),style=wx.OPEN)
		if dialog.ShowModal() == wx.ID_OK:
			self.fn=dialog.GetPath()
			# if self.fn!=[]:
			# 	fnmae=self.fn
			self.doodle1.filename(self.fn)
			# if(self.doodle.color=="Blue"):
			# 	fore_colr=self.doodle.color
			# self.doodle.SLD([])
			# if(self.doodle.color=="Orange"):
			# 	back_colr=self.doodle.color
			# self.doodle.RegionSet(False)
			# self.doodle.CGD()
		else:
			fnmae=[]
		dialog.Destroy()

	def alpha(self,event):
		dlg = wx.TextEntryDialog(self, 'Enter some text','Text Entry')
		dlg.SetValue("0.6")
		if dlg.ShowModal() == wx.ID_OK:
			self.alp=float(dlg.GetValue())
			if(self.alp<0):
				self.alp=0
			elif(self.alp>1):
				self.alp=1
			print(self.alp)
		dlg.Destroy()

	def Iter15(self,event):
		self.iters=15
	def Iter25(self,event):
		self.iters=25
	

	def Run(self,event):
		fname=self.doodle.fn
		colr=self.doodle.color
		if fname==[] or self.doodle1.fn==[]:
			wx.MessageBox("Load Image first!","Error!", style=wx.OK|wx.ICON_EXCLAMATION)
		if fname!=[] and self.doodle1.fn!=[]:
			fore=self.doodle.foreground
			pos1=self.doodle.GRP(1)
			back=self.doodle.background
			pos2=self.doodle.GRP(2)
			iterationss=self.iters

			filename,tff=algo_grabcut(filename1=self.doodle.fn,filename2=self.doodle1.fn)
			if(tff):			
				im = Image.open(filename)
				im.show()
				

	def SetRegion(self,event):
		fname=self.doodle.fn
		if fname==[]:
			iterationss=self.iters
			wx.MessageBox("Load Image first!","Error!", style=wx.OK|wx.ICON_EXCLAMATION)
		if fname!=[]:
			self.doodle.SLD([])
			self.doodle.DrawRegion(True)
			self.doodle.CGD()

	def SetForeground(self,event):
		fname=self.doodle.fn
		self.doodle.DrawRegion(False)
		self.doodle.SC("Blue")

	def SetBackground(self,event):
		fname=self.doodle.fn
		self.doodle.DrawRegion(False)
		self.doodle.SC("Orange")

	def Check(self,event):
		if not event.IsChecked():
			self.doodle.Use_Algo(False)
		else:
			self.doodle.Use_Algo(True)

	def Clear(self,event):
		self.doodle.SLD([])
		self.doodle.CGD()

	






class GrabFrame(wx.Frame):
	def __init__(self,parent):
		wx.Frame.__init__(self, parent, -1, "Style Transfer", size=(1024,512),style=wx.DEFAULT_FRAME_STYLE | wx.NO_FULL_REPAINT_ON_RESIZE)
		self.doodle = Doodle(-1,self)
		self.doodle1=Doodle(-1,self)
		fore_color="Blue"
		boxess = wx.BoxSizer(wx.HORIZONTAL)	
		back_color="Orange"
		cPanel = ControlPanel(self, -1, self.doodle,self.doodle1)
		Region_Bounday_color="Silver"
		boxess.Add(cPanel, 0, wx.EXPAND)
		Primary_iterations=3
		boxess.Add(self.doodle, 1, wx.EXPAND)
		boxess.Add(self.doodle1, 1, wx.EXPAND)
		# boxess.Add(self.dod, 0, wx.EXPAND)
		Primary_neighboorhood=4
		self.SetSizer(boxess)
		opencv_lib=False
		self.Centre()

class App(wx.App):
	def OnInit(self):
		frame = GrabFrame(None)
		frame.Show(True)
		return True


def main():
	app = App()
	app.MainLoop()

main()