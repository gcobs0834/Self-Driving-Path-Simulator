import math
from numpy import cos, sin, pi, arcsin




class Car():
    
    def __init__(self, start_point, start_degree, car_radius):
        self.car = {
            'radius': car_radius,
            'x':start_point[0],
            'y':start_point[1],
            'degree': start_degree,
            'steering_wheel_degree': 0
        }  # [x,y,Θ]

        self.sensor_dist = {
            'l_point':0, #left distance
            'c_point':0, #center distance
            'r_point':0, #right distance
        }
        self.sensor_point = {
            'l_point':start_point,
            'c_point':start_point,
            'r_point':start_point
        }

    def kinematic_next(self):
        self.car['x'] = (self.car['x'] + 
                        cos(pi*(self.car['degree'] + self.car['steering_wheel_degree'])/180) + 
                        sin(pi*(self.car['degree'])/180)*sin(pi*(self.car['steering_wheel_degree'])/180))
        self.car['y'] = (self.car['y'] + 
                        sin(pi*(self.car['degree'] + self.car['steering_wheel_degree'])/180) - 
                        sin(pi*(self.car['steering_wheel_degree'])/180)*cos(pi*(self.car['degree'])/180))
        self.car['degree'] = (self.car['degree'] -
                            180*arcsin(
                                sin(pi*(self.car['steering_wheel_degree'])/180))/pi)
    
    def loc(self):
        return [self.car['x'],self.car['y']]

    def sensor(self):
        left_degree = self.car['degree'] + 45
        right_degree = self.car['degree'] - 45
        detect_range = 100
        ##detect point renew
        l_x = int(self.car['x']+ cos(pi* left_degree/180)*detect_range) ##force to int
        l_y = int(self.car['y']+ sin(pi* left_degree/180)*detect_range) ##force to int
        c_x = int(self.car['x']+ cos(pi*self.car['degree']/180)*detect_range) #middle dect_point
        c_y = int(self.car['y']+ sin(pi*self.car['degree']/180)*detect_range) #middle dect_point
        r_x = int(self.car['x']+ cos(pi*right_degree/180)*detect_range) #right dect_point
        r_y = int(self.car['y']+ sin(pi*right_degree/180)*detect_range) #right dect_point
        return [l_x,l_y], [c_x,c_y], [r_x,r_y]
    
    def update_sensor(self, road_edges):
        l_point, c_point, r_point = self.sensor()
        road_lines = [[road_edges[i], road_edges[i+1]]
                      for i in range(len(road_edges)-1)]
        s_lines = {
            "l_point": [self.loc(), l_point],
            "c_point": [self.loc(), c_point],
            "r_point": [self.loc(), r_point]
        }
        s_p = {"l_point": [], "c_point": [], "r_point": []}
        for sensor in s_lines:
            self.sensor_dist[sensor] = 0
            self.sensor_point[sensor] = [0, 0]
            self.sensor_dist[sensor] = 100000
            for l in road_lines:
                point = self.line_intersection(l, s_lines[sensor])
                if sensor == "c_point":
                    print('l:{} sensor:{}'.format(l, s_lines[sensor])) 
                if point is not None:
                    s_p[sensor].append(point)
            for point in s_p[sensor]:
                if self.dist(self.loc(), point) < self.sensor_dist[sensor]:
                    self.sensor_dist[sensor] = self.dist(self.loc(), point)
                    self.sensor_point[sensor] = point
            print('a:{} b:{} c:{} '.format(self.sensor_dist['l_point'], self.sensor_dist['c_point'], self.sensor_dist['r_point']))
        return self.sensor_dist['l_point'], self.sensor_dist['c_point'], self.sensor_dist['r_point']

  
    def turn_wheel(self,turn_degree):
        self.car['steering_wheel_degree'] = turn_degree

    def line_intersection(self,l1, l2):
        xdiff = (l1[0][0] - l1[1][0], l2[0][0] - l2[1][0])
        ydiff = (l1[0][1] - l1[1][1], l2[0][1] - l2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]
        div = det(xdiff, ydiff)
        if div == 0:
            return None
        d = (det(*l1), det(*l2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        x = round(x)
        y = round(y)
        if x >= min(l1[0][0], l1[1][0])-1 and x <= max(l1[0][0], l1[1][0])+1 \
                and y >= min(l1[0][1], l1[1][1])-1 and y <= max(l1[0][1], l1[1][1])+1 \
                and x >= min(l2[0][0], l2[1][0])-1 and x <= max(l2[0][0], l2[1][0])+1 \
                and y >= min(l2[0][1], l2[1][1])-1 and y <= max(l2[0][1], l2[1][1])+1:
            return [x, y]
        else:
            return None
    def dist(self,a,b):
            return math.sqrt((b[0]-a[0])**2+(b[1]-a[1])**2)



