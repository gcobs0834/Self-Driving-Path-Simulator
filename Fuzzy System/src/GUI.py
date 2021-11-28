import numpy as np
from tkinter import *
import tkinter as tk
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from tkinter.filedialog import askopenfilename
from Car import Car
from Map import Map, Edge, Recorder
from gui_utils import add_text, add_button
from fuzz import Fuzzifier, Rules
from enum import Enum
from time import sleep


class State(Enum):
    PLAYING = 0
    CRASH = 1
    FINISH = 2


class GUI(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.root = master
        self.grid()
        self.data = self.load_data()
        self.car, self.edge = self.init_components()
        self.state = State.PLAYING
        self.recorder = Recorder()
        self.recorder.add(self.car)
        self.create_widgets()
        self.clean_fig()
        self.draw_road(self.edge.finish_area, self.edge.road_edges)
        self.draw_car(self.car.loc(), self.car.car['degree'], self.car.car['radius'])

    def load_data(self):
        case_file_path = './maps/case01.txt'
        d = Map(case_file_path)
        return d.get()

    def init_components(self):
        c = Car(self.data['start_point'], self.data['start_degree'],3)
        c.update_sensor(self.data['road_edges'])
        r = Edge(self.data['finish_area'], self.data['road_edges'])
        return c, r

    def create_widgets(self):
        self.winfo_toplevel().title("CI HW1")

        # 地圖與道路
        self.road_fig = Figure(figsize=(6, 6), dpi=100)
        
        self.road_canvas = FigureCanvasTkAgg(
            self.road_fig, self)
        self.road_canvas.draw()
        self.road_canvas.get_tk_widget().grid(row=0, column=0, columnspan=3)

         # current State
        _, self.st = add_text(self,
                              1, "Status:", self.state)
        # car location, sensor dist, car steering wheel
        _, self.loc = add_text(self, 2, "Location:", self.car.loc())
        _, self.l_point = add_text(self,
                              3, "Front Left Dist:", self.car.sensor_dist['l_point'])
        _, self.c_point = add_text(self,
                             4, "Front Dist:", self.car.sensor_dist['c_point'])
        _, self.r_point = add_text(self,
                              5, "Front Right Dist:", self.car.sensor_dist['r_point'])
        _, self.cd = add_text(self,
                              6, "Car Degree:", self.car.car['degree'])
        _, self.swd = add_text(self,
                               7, "Car Steering Wheel Degree:", self.car.car['steering_wheel_degree'])
        # update car
        _, self.next = add_button(self,
                                  8, "Start Playing", "Run", self.run)


    def turn_steering_wheel(self, degree):
        self.car.turn_wheel(degree)

    def run(self):
        while self.state == State.PLAYING:
            self.update()
            sleep(0.02)

    def update(self):
        self.update_state()
        self.update_car()
        self.recorder.add(self.car)

    def update_state(self):
        if self.edge.is_crash(self.car):
            self.state = State.CRASH
        elif self.edge.is_finish(self.car):
            self.state = State.FINISH
            self.recorder.to_file()
            
        self.st["text"] = self.state

    def update_car(self):
        l_point, c_point, r_point = self.car.update_sensor(
            self.data['road_edges'])
        l_point = Fuzzifier.l_point(l_point)
        c_point = Fuzzifier.c_point(c_point)
        r_point = Fuzzifier.r_point(r_point)
        self.turn_steering_wheel(Rules.apply(l_point, c_point, r_point))

        self.car.kinematic_next()
        self.loc["text"] = self.car.loc()
        self.cd["text"] = self.car.car['degree']
        self.swd["text"] = self.car.car['steering_wheel_degree']
        self.clean_fig()
        self.draw_road(self.edge.finish_area, self.edge.road_edges)
        self.draw_car(self.car.loc(), self.car.car['degree'], self.car.car['radius'])
        self.draw_route()
        self.road_canvas.draw()

    def clean_fig(self):
        # init_fig
        self.road_fig.clf()
        self.road_fig.ax = self.road_fig.add_subplot(111)
        self.road_fig.ax.set_title('CI HW1')
        self.road_fig.ax.set_aspect(1)
        self.road_fig.ax.set_xlim([-20, 60])
        self.road_fig.ax.set_ylim([-10, 60])

    def draw_road(self, finish_area, road_edges):
        # draw edge
        for i in range(len(road_edges)-1):
            self.road_fig.ax.text(road_edges[i][0], road_edges[i][1], '({},{})'.format(
                road_edges[i][0], road_edges[i][1]))
            self.road_fig.ax.plot([road_edges[i][0], road_edges[i+1][0]], [
                                  road_edges[i][1], road_edges[i+1][1]], 'k')
        # finish_area
        a, b = finish_area[0]
        c, d = finish_area[1]
        self.road_fig.ax.plot([a, c], [b, b], 'r')
        self.road_fig.ax.plot([c, c], [b, d], 'r')
        self.road_fig.ax.plot([c, a], [d, d], 'r')
        self.road_fig.ax.plot([a, a], [d, b], 'r')

    def draw_car(self, loc, car_degree, radius):
        # draw car
        self.road_fig.ax.plot(loc[0], loc[1], '.b')
        circle = plt.Circle(loc, radius, color='b', fill=False)
        self.road_fig.ax.add_artist(circle)
        # sensors
        self.l_point["text"], self.c_point["text"], self.r_point["text"] = self.car.update_sensor(
            self.data['road_edges'])
        for s in self.car.sensor_point:
            self.road_fig.ax.plot(
                [loc[0], self.car.sensor_point[s][0]],
                [loc[1], self.car.sensor_point[s][1]], 'y')
            self.road_fig.ax.plot(
                self.car.sensor_point[s][0], self.car.sensor_point[s][1], '.b')

    def draw_route(self):
        records = self.recorder.get()
        for r in records:
            self.road_fig.ax.plot(int(float(r[0])+0.0001), int(float(r[1])+0.0001), '.y')


if __name__ == "__main__":
    root = tk.Tk()
    app = GUI(root)
    root.mainloop()