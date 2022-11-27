from PyQt5.QtWidgets import QLabel,QWidget
from PyQt5.QtCore import Qt,QRect,QPoint
from PyQt5.QtGui import QPainter,QPen,QImage,QWheelEvent,QMouseEvent,QContextMenuEvent

class Label(QLabel):
  x0=0
  y0=0
  x1=0
  y1=0
  x_move=0
  y_move=0
  ZoomValue=1.0
  old_pos=None
  Pressed=False
  drawtext=False
#   //是否显示字符
  drawtext = False
	# //显示的字符
  disptext=''
	# //显示字符位置row
  row_disptext=0
	# //显示字符位置col;
  col_disptext=0
  open_mouse_flag=False
  select_roi_flag=False
  draw_roi_flag=False
  clear_flag=False
  rect = QRect()
  Image=QImage()
  def __init__(self,parent=None):
     super(Label,self).__init__(parent)
     load=self.Image.load('./pics/egg_chd.jpg')
     self.setMinimumWidth(32)
     self.setMinimumHeight(32)
     self.disp()
  def clear_draw_text(self):
    self.disptext=''
    self.row_disptext=-1
    self.col_disptext=-1
    self.drawtext=False
  def disp(self):
        self.clear_draw_text()
        size=self.Image.byteCount()
        if(size==0):
          return
        painter = QPainter(self)
        width=min(self.Image.width(),self.width()) 
        height=(width*1.0)/(self.Image.width()*1.0/self.Image.height())
        height=min(height,self.height()) 
        width=height*1.0*(self.Image.width()*1.0/self.Image.height())
        painter.translate(self.width() / 2 + self.x_move, self.height() / 2 + self.y_move);     # int X_move , Y_move ; x y 轴平移的距离
        painter.scale(self.ZoomValue,self.ZoomValue)
        pecRect=QRect(-width / 2, -height / 2, width, height)
        painter.drawImage(pecRect,self.Image)

  def OnZoomInImage(self):
     self.ZoomValue +=0.05 
     self.update()
  def OnZoomOutImage(self):
      self.ZoomValue -=0.05 
      self.ZoomValue=max(self.ZoomValue,0.05)
      self.update()
  def OnPresetImage(self):
      self.ZoomValue=1.0
      self.x_move=0
      self.y_move=0
  def wheelEvent(self,event:QWheelEvent):
      value=event.angleDelta().y()
      if(value>0):
         self.OnZoomInImage()#放大
      else:
         self.OnZoomOutImage()#缩小

  #按下鼠标
  def mousePressEvent(self, event:QMouseEvent):
    self.Pressed=True
    self.old_pos=event.pos()
    if self.open_mouse_flag is True:
      self.select_roi_flag=True
      self.x0=event.x()
      self.y0=event.y()

  #释放鼠标
  def mouseReleaseEvent(self, event:QMouseEvent):
    self.select_roi_flag=False
    self.Pressed=False
    self.setCursor(Qt.ArrowCursor)

  #移动鼠标
  def mouseMoveEvent(self, event:QMouseEvent):
    if(self.Pressed==False):
       return QWidget.mouseMoveEvent(event)
    self.setCursor(Qt.SizeAllCursor)
    pos=event.pos()
    xPtInterval=pos.x()-self.old_pos.x()
    yPtInterval=pos.y()-self.old_pos.y()
    self.x_move+=xPtInterval
    self.y_move+=yPtInterval
    self.old_pos=pos
    self.update()
    if self.select_roi_flag is True:
      self.x1=event.x()
      self.y1=event.y()
      if self.draw_roi_flag is True:
        self.update()
 #鼠标右键
  def contextMenuEvent(self,event:QContextMenuEvent):
     pos=event.pos()
     pos=self.mapToGlobal(pos)

     
  #绘制事件
  def paintEvent(self,event):
    # super().paintEvent(event)
    
    # painter.setPen(QPen(Qt.red, 5, Qt.SolidLine))
    # if self.clear_flag is True:
    #   self.x0=0
    #   self.y0=0
    #   self.x1=0
    #   self.y1=0
    # self.rect = QRect(self.x0, self.y0, abs(self.x1 - self.x0), abs(self.y1 - self.y0))
    # painter.drawRect(self.rect)
    # self.update() 
    size=self.Image.byteCount()
    if(size==0):
      return
    painter = QPainter(self)
    width=min(self.Image.width(),self.width()) 
    height=(width*1.0)/(self.Image.width()*1.0/self.Image.height())
    height=min(height,self.height()) 
    width=height*1.0*(self.Image.width()*1.0/self.Image.height())
    painter.translate(self.width() / 2 + self.x_move, self.height() / 2 + self.y_move);     # int X_move , Y_move ; x y 轴平移的距离
    painter.scale(self.ZoomValue,self.ZoomValue)
    pecRect=QRect(-width / 2, -height / 2, width, height)
    painter.drawImage(pecRect,self.Image)
    if self.drawtext:
      if len(self.disptext)==0:
        return
      self.row_disptext=max(self.row_disptext,1)
      self.col_disptext=max(self.col_disptext,1)
      painter.drawText(self.row_disptext,self.col_disptext,self.disptext)

