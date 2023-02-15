import tkinter as tk
from tkinter import ttk
from tkinter import *
from PIL import Image
import tkinter.filedialog
from tkinter.tix import Tk, Control, ComboBox  #升级的组合控件包
from tkinter.messagebox import showinfo, showwarning, showerror #各种类型的提示框
from PIL import Image,ImageTk
import operator
from numpy.lib.shape_base import column_stack
from skimage import *
from skimage.feature import greycomatrix, greycoprops, texture
import os
import numpy as np 
from scipy.spatial.distance import pdist
from selenium import webdriver 

from vigor_select import vigor_select

grd_path=[]
grd_label=[]
sat_list=[]
grd_index_dict={}
sat_index_dict={}
index_to_name={}
city = "NewYork"
data_root = "E:/dataset/data_vigor/"
label_root = "splits"
idx = 0
test_sat_list_fname = os.path.join(data_root, label_root, city, 'satellite_list.txt')
with open(test_sat_list_fname, 'r') as file:
    for line in file.readlines():
        sat_list.append(os.path.join(data_root, city, 'satellite', line.replace('\n', '')))
        sat_index_dict[line.replace('\n', '')] = idx
        index_to_name[idx]=line.replace('\n', '')
        idx += 1

with open(os.path.join(data_root, "splits", city, 'same_area_balanced_test.txt'),'r') as file:
    idx=0
    for line in file.readlines():
        data = np.array(line.split(' '))
        label = []
        for i in [1, 4, 7, 10]:
            label.append(sat_index_dict[data[i]])
        label = np.array(label).astype(np.int32)
        grd_label.append(label)
        grd_path.append(data[0])
        grd_index_dict[data[0]]=idx
        idx+=1
gpath=data_root + "NewYork/panorama/" +grd_path[0]

def center_window(root, width, height):
  screenwidth = root.winfo_screenwidth()
  screenheight = root.winfo_screenheight()
  size = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
  root.geometry(size)
  root.update()

root = Tk()
root.title("基于内容的图像检索")
root.geometry("800x600")
center_window(root, 900, 600)
root.resizable(width=False, height=False)
# root.iconbitmap(default = './index/logo.ico')

comvalue=tkinter.StringVar()#窗体自带的文本，新建一个值  
comboxlist=ttk.Combobox(root,textvariable=comvalue) #初始化

# 欧氏距离函数
def get_euclidean_dist(a,b):
    a=np.array(a)
    b=np.array(b)
    res=np.sqrt(np.sum(np.power((a - b), 2)))
    return res
# 加权距离函数
def get_weighted_dist(a,b):
    X=np.vstack([a,b])
    res=pdist(X, 'sqeuclidean')
    return res[0]
# 切比雪夫距离
def get_intersection_dist(a,b):
    X=np.vstack([a,b])
    res=pdist(X, 'canberra')
    return res[0]
dist_name='欧氏距离'
def select_dist(*agr):
    global dist_name
    dist_name=comboxlist.get()
def get_dist(a,b):
    if(dist_name=='欧氏距离'):
        return get_euclidean_dist(a,b)
    else: 
        if(dist_name=='加权距离'):
            return get_weighted_dist(a,b)
        else:
            return get_intersection_dist(a,b)

class item:
    def __init__(self):
        self.path=""
        self.value=0

canvas = Canvas(root,height=800, width=500,scrollregion=(0,0,100,3100))
bar=Scrollbar(root)

def show_imgs(imgs,value):
    for i in range(len(imgs)):
        canvas.create_image(110+i%3*160, 70+i//3*180, image=imgs[i])
        canvas.create_text(110+i%3*160, 70+i//3*180+90,text="实际距离: "+str(value[i])[:str(value[i]).find('.')]+'m')

    bar.pack(side=RIGHT,fill=tk.Y)
    bar.config(command=canvas.yview)
    canvas.config(yscrollcommand=bar.set)
    canvas.bind("<MouseWheel>",processWheel)

    canvas.pack(side=RIGHT,fill=BOTH)
    root.mainloop()

def select():
    path=tkinter.filedialog.askopenfilename()
    if path!='':
        global gpath
        gpath=path
        img = Image.open(path)
        img = img.resize((250, 250),Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(img)
        label=tk.Label(root,image=photo)
        label.place(x=30,y=320)
        root.mainloop()

def processWheel(event):
        a= int(-(event.delta)/60)
        canvas.yview_scroll(a,'units')

def geo_select():
    canvas.delete("all")

    imgs=[]
    name=gpath[gpath.rfind("/")+1:]
    sat_list=vigor_select(index=grd_index_dict[name])
    for i in sat_list:
        sat_path=data_root+"NewYork/satellite/"+index_to_name[i]
        img = Image.open(sat_path)
        img = img.resize((150, 150))
        imgs.append(ImageTk.PhotoImage(img))
    
    value=[]
    LL=np.array(name.split(','))
    Lat_A,Lng_A=LL[1].astype(np.float32),LL[2].astype(np.float32)
    for i in sat_list:
        name=index_to_name[i]
        LL=np.array(name.replace(".png","").split('_'))
        Lat_B,Lng_B=LL[1].astype(np.float32),LL[2].astype(np.float32)
        value.append(gps2distance(Lat_A,Lng_A,Lat_B,Lng_B))
    show_imgs(imgs,value)

# compute the distance between two locations [Lat_A,Lng_A], [Lat_B,Lng_B]
def gps2distance(Lat_A,Lng_A,Lat_B,Lng_B):
    # https://en.wikipedia.org/wiki/Great-circle_distance
    lat_A = Lat_A * np.pi/180.
    lat_B = Lat_B * np.pi/180.
    lng_A = Lng_A * np.pi/180.
    lng_B = Lng_B * np.pi/180.
    R = 6371004.
    C = np.sin(lat_A)*np.sin(lat_B) + np.cos(lat_A)*np.cos(lat_B)*np.cos(lng_A-lng_B)
    distance = R*np.arccos(C)
    return distance


def help():
    pass
    # chrome_driver="F:/anaconda3/Lib/site-packages/selenium/webdriver/chrome/chromedriver.exe"
    # driver = webdriver.Chrome(executable_path=chrome_driver)
    # driver.get("file://F:/大学学习/大三下/综合实训/CBIR/index/main.html")

def init_window(self):
    self.master.title("跨视图地理定位系统VIGOR")
    self.pack(fill=BOTH,expand=1)

    menu=Menu(self.master)
    self.master.configure(menu=menu)
    menu.add_cascade(label="文件(F)",command=select)
    menu.add_cascade(label="复位",command=reset)
    menu.add_cascade(label="帮助(H)",command=help)

def reset():
    tk.messagebox.showinfo(title = '提示',message='系统已重置')
    global gpath
    gpath=data_root + "NewYork/panorama/" +grd_path[0]
    global app
    app=Window(root)

def init():
    
    ybar=tk.Scrollbar(root,orient='vertical')
    tree = ttk.Treeview(root,height=12,selectmode='browse',yscrollcommand=ybar.set)
    parent=[]
    for name in grd_path:
        parent.append(data_root + "NewYork/panorama/" + name)

    parent.reverse()
    for i in range(len(parent)):
        tree.insert("",0,parent[i],text=(parent[i][parent[i].rfind("/")+1:]),values=parent[i])
        # son[i].reverse()
        # for j in son[i]:
        #     tree.insert(p,0,j,text=(j[j.rfind('/')+1:len(j)]),values=j)
    def selectTree(event):
        for item in tree.selection():
            item_text = tree.item(item, "values")
            global gpath
            gpath=item_text[0]

            if(gpath.find('.')>=0):
                img = Image.open(gpath)
                img = img.resize((350, 250),Image.ANTIALIAS)
                photo = ImageTk.PhotoImage(img)
                label=tk.Label(root,image=photo)
                label.place(x=30,y=320)

                root.mainloop()
    
    # 选中行
    tree.bind('<<TreeviewSelect>>', selectTree)
    tree.place(x=30,y=15,width=350,height=250)
    ybar['command']=tree.yview
    ybar.place(x=380,y=15,height=250)

    comboxlist=ttk.Combobox(root,textvariable=comvalue) #初始化  
    comboxlist["values"]=("VIGOR","SAFA")  
    comboxlist.current(0)  #选择第一个  
    comboxlist.bind("<<ComboboxSelected>>",select_dist)  #绑定事件,(下拉列表框被选中时，绑定go()函数)  
    comboxlist.place(x=30,y=280)

    bnt=tk.Button(root, text ="定位",activebackground='red', width=15,command=geo_select)
    bnt.place(x=230,y=275)

    img = Image.open(gpath,'r')
    img = img.resize((350, 250),Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(img)
    label=tk.Label(root,image=photo)
    label.place(x=30,y=320)

    geo_select()

    root.mainloop()

class Window(Frame):
    def __init__(self,master=None):
        Frame.__init__(self,master)
        self.master=master
        init_window(self)
        init()

app=Window(root)
root.mainloop()
