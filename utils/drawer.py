from tkinter import *
from PIL import ImageDraw as D

import math
class DashedImageDraw(D.ImageDraw):

    def thick_line(self, xy, direction, fill=None, width=0):

        if xy[0] != xy[1]:
            self.line(xy, fill=fill, width=width)
        else:
            x1, y1 = xy[0]
            dx1, dy1 = direction[0]
            dx2, dy2 = direction[1]
            if dy2 - dy1 < 0:
                x1 -= 1
            if dx2 - dx1 < 0:
                y1 -= 1
            if dy2 - dy1 != 0:
                if dx2 - dx1 != 0:
                    k = - (dx2 - dx1) / (dy2 - dy1)
                    a = 1 / math.sqrt(1 + k ** 2)
                    b = (width * a - 1) / 2
                else:
                    k = 0
                    b = (width - 1) / 2
                x3 = x1 - math.floor(b)
                y3 = y1 - int(k * b)
                x4 = x1 + math.ceil(b)
                y4 = y1 + int(k * b)
            else:
                x3 = x1
                y3 = y1 - math.floor((width - 1) / 2)
                x4 = x1
                y4 = y1 + math.ceil((width - 1) / 2)
            self.line([(x3, y3), (x4, y4)], fill=fill, width=1)
        return

    def dashed_line(self, xy, dash=(2, 2), fill=None, width=0):
        for i in range(len(xy) - 1):
            x1, y1 = xy[i]
            x2, y2 = xy[i + 1]
            x_length = x2 - x1
            y_length = y2 - y1
            length = math.sqrt(x_length ** 2 + y_length ** 2)
            dash_enabled = True
            postion = 0
            while postion <= length:
                for dash_step in dash:
                    if postion > length:
                        break
                    if dash_enabled:
                        start = postion / length
                        end = min((postion + dash_step - 1) / length, 1)
                        self.thick_line([(round(x1 + start * x_length),
                                          round(y1 + start * y_length)),
                                         (round(x1 + end * x_length),
                                          round(y1 + end * y_length))],
                                        xy, fill, width)
                    dash_enabled = not dash_enabled
                    postion += dash_step
        return

    def dashed_rectangle(self, xy, dash=(2, 2), outline=None, width=0):
        x1, y1 = xy[0]
        x2, y2 = xy[1]
        halfwidth1 = math.floor((width - 1) / 2)
        halfwidth2 = math.ceil((width - 1) / 2)
        min_dash_gap = min(dash[1::2])
        end_change1 = halfwidth1 + min_dash_gap + 1
        end_change2 = halfwidth2 + min_dash_gap + 1
        odd_width_change = (width - 1) % 2
        self.dashed_line([(x1 - halfwidth1, y1), (x2 - end_change1, y1)],
                         dash, outline, width)
        self.dashed_line([(x2, y1 - halfwidth1), (x2, y2 - end_change1)],
                         dash, outline, width)
        self.dashed_line([(x2 + halfwidth2, y2 + odd_width_change),
                          (x1 + end_change2, y2 + odd_width_change)],
                         dash, outline, width)
        self.dashed_line([(x1 + odd_width_change, y2 + halfwidth2),
                          (x1 + odd_width_change, y1 + end_change2)],
                         dash, outline, width)
        return

class RectangleDrawer:
    def __init__(self, master):
        self.master = master
        width, height = 512, 512
        self.canvas = Canvas(self.master, bg='#F0FFF0', width=width, height=height)
        self.canvas.pack()

        self.rectangles = []
        self.colors = ['blue', 'red', 'purple', 'orange', 'green', 'yellow', 'black']

        self.canvas.bind("<Button-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.start_x = None
        self.start_y = None
        self.cur_rect = None
        self.master.update()
        width = self.master.winfo_width()
        height = self.master.winfo_height()
        x = (self.master.winfo_screenwidth() // 2) - (width // 2)
        y = (self.master.winfo_screenheight() // 2) - (height // 2)
        self.master.geometry('{}x{}+{}+{}'.format(width, height, x, y))


    def on_button_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.cur_rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline=self.colors[len(self.rectangles)%len(self.colors)], width=5, dash=(4, 4))

    def on_move_press(self, event):
        cur_x, cur_y = (event.x, event.y)
        self.canvas.coords(self.cur_rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event):
        cur_x, cur_y = (event.x, event.y)
        self.rectangles.append([self.start_x, self.start_y, cur_x, cur_y])
        self.cur_rect = None

    def get_rectangles(self):
        return self.rectangles


def draw_rectangle():
    root = Tk()
    root.title("Rectangle Drawer")

    drawer = RectangleDrawer(root)

    def on_enter_press(event):
        root.quit()

    root.bind('<Return>', on_enter_press)

    root.mainloop()
    rectangles = drawer.get_rectangles()

    new_rects = []
    for r in rectangles:
        new_rects.extend(r)

    return new_rects

if __name__ == '__main__':
    root = Tk()
    root.title("Rectangle Drawer")

    drawer = RectangleDrawer(root)

    def on_enter_press(event):
        root.quit()

    root.bind('<Return>', on_enter_press)

    root.mainloop()
    rectangles = drawer.get_rectangles()

    string = '['
    for r in rectangles:
        string += '['
        for n in r:
            string += str(n)
            string += ','
        string = string[:-1]
        string += '],'
    string = string[:-1]
    string += ']'
    print("Rectangles:", string)