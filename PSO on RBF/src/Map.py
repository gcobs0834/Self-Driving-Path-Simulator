
from numpy import sin,arcsin,cos,pi
import math

class Map():
    def __init__(self, file_path):
        self.file_path = file_path
        self.start_point = []
        self.start_degree = 0
        self.finish_area = [] ##[[x1,y1],[x2,y2]]
        self.road_edges = []
        self.load_data()

    def load_data(self):
        with open(self.file_path) as f:
            data = f.readlines()
        for i in range(len(data)):
            points = list(map(lambda x: int(x),
                            data[i].replace('\n','').split(',')))
            if i==0:
                self.start_point = points[:-1]
                self.start_degree = points[-1]
            elif i==1 or i ==2:
                self.finish_area.append(points)
            else:
                self.road_edges.append(points)
    def get(self):
        return {
            'start_point': self.start_point,
            'start_degree': self.start_degree,
            'finish_area': self.finish_area,
            'road_edges': self.road_edges
        }

class Edge():
    def __init__(self,finish_area,road_edges):
        self.finish_area = finish_area
        self.road_edges =road_edges
        

    def is_finish(self, car):
        point = car.loc()
        if point[0] < self.finish_area[1][0]\
            and point[0] > self.finish_area[0][0]\
            and point[1] > self.finish_area[1][1]\
            and point[1] < self.finish_area[0][1]:
            return True
        return False   
    
    def is_crash(self, car):
        for i in self.dist(car, self.road_edges):
            if i +0.01<= car.car['radius']:
                return True
        return False
    def edge(self,car):
        detect_range = 100
        u_point = [int(car.car['x'] + cos(pi*90/180)*detect_range),
                int(car.car['y'] + sin(pi*90/180)*detect_range)]
        d_point = [int(car.car['x'] + cos(pi*270/180)*detect_range),
                int(car.car['y'] + sin(pi*270/180)*detect_range)]
        l_point = [int(car.car['x'] + cos(pi*180/180)*detect_range),
                int(car.car['y'] + sin(pi*180/180)*detect_range)]
        r_point = [int(car.car['x'] + cos(pi*0/180)*detect_range),
                int(car.car['x'] + sin(pi*0/180)*detect_range)]
        return u_point,d_point,l_point,r_point
    
    def dist(self,car,edges): ##return dist for every side
        u_point,d_point,l_point,r_point = self.edge(car)
        Edges = ([edges[i],edges[i+1]] for i in range(len(edges)-1))
        car_lines = {
            "up":[car.loc(), u_point],
            "down":[car.loc(), d_point],
            "left":[car.loc(), l_point],
            "right":[car.loc(), r_point]
        }
        def line_intersection(line1, line2): 
            xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0]) 
            ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) #Typo was here 

            def det(a, b): 
             return a[0] * b[1] - a[1] * b[0] 

            div = det(xdiff, ydiff) 
            if div == 0: 
                return None

            d = (det(*line1), det(*line2)) 
            x = round(det(d, xdiff)/div) 
            y = round(det(d, ydiff)/div) 
            ##判斷是否在兩線段中間
            if x >= min(line1[0][0],line1[1][0])-1 and x<=max(line1[0][0],line1[1][0])+1\
                    and x >= min(line2[0][0],line2[1][0])-1 and x<=max(line2[0][0],line2[1][0])+1\
                    and y >= min(line2[0][1],line2[1][1])-1 and x<=max(line2[0][1],line2[1][1])+1\
                    and y >= min(line1[0][1],line1[1][1])-1 and x<=max(line1[0][1],line1[1][1])+1\
                :
                return [x, y]
            else:
                return None
        # print line_intersection((A, B), (C, D)) #
        def dist(a,b):
            return math.sqrt((a[0] - b[0])**2+ (a[1] - b[1])**2)

        side_dist = {}
        side_point = {}
        for side in car_lines:
            side_point[side] = [0,0]
            side_dist[side] = 10000
            for l in Edges:
                point = line_intersection(l, car_lines[side])
                if point is not None and dist(point, car.loc()) < side_dist[side]:
                    side_point[side].append(point)
        return [side_dist["up"],side_dist["down"],side_dist["left"],side_dist["right"]]


class Recorder():
    def __init__(self):
        self.records = []

    def to_file(self):
        with open("./outputs/train4D.txt", "w") as f4d, open("./outputs/train6D.txt", "w") as f6d:
            for r in self.records:
                f4d.write(" ".join(r[2:]) + "\n")
                f6d.write(" ".join(r) + "\n")
        #print("Records Wrote to file.")

    def add(self, car):
        r = list(map(lambda x: str(x), [car.loc()[0], car.loc()[1], car.sensor_dist['c_point'],
                                        car.sensor_dist['r_point'], car.sensor_dist['l_point'], car.car['steering_wheel_degree']]))
        self.records.append(r)
        #print("Records Added {}".format(self.records[-1]))

    def get(self):
        return self.records

    def clean(self):
        self.records = []



if __name__ =='__main__':
    loads = Map('./maps/case01.txt')
    print(loads.get())
    