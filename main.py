import numpy as np
import tkinter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Triangle:
    def __init__(self, p1, p2, p3):
        self.points = [p1, p2, p3]
        self.circumcenter = self.calculate_circumcenter()

    def calculate_circumcenter(self):
        p1, p2, p3 = self.points
        d = 2 * (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))
        ux = ((p1.x**2 + p1.y**2) * (p2.y - p3.y) + (p2.x**2 + p2.y**2) * (p3.y - p1.y) + (p3.x**2 + p3.y**2) * (p1.y - p2.y)) / d
        uy = ((p1.x**2 + p1.y**2) * (p3.x - p2.x) + (p2.x**2 + p2.y**2) * (p1.x - p3.x) + (p3.x**2 + p3.y**2) * (p2.x - p1.x)) / d
        return Point(ux, uy)

def bowyer_watson(points):
    super_triangle = create_super_triangle(points)
    triangulation = [super_triangle]

    for point in points:
        bad_triangles = []
        for triangle in triangulation:
            if is_inside_circumcircle(triangle, point):
                bad_triangles.append(triangle)

        polygon = []
        for triangle in bad_triangles:
            for p in triangle.points:
                if is_edge_shared(p, bad_triangles):
                    polygon.append(p)

        for triangle in bad_triangles:
            triangulation.remove(triangle)

        for edge in polygon_edges(polygon):
            triangulation.append(Triangle(edge.p1, edge.p2, point))

    for triangle in triangulation:
        if any(point in super_triangle.points for point in triangle.points):
            triangulation.remove(triangle)

    return triangulation

def create_super_triangle(points):
    min_x = min(point.x for point in points)
    min_y = min(point.y for point in points)
    max_x = max(point.x for point in points)
    max_y = max(point.y for point in points)

    dx = max_x - min_x
    dy = max_y - min_y
    dmax = max(dx, dy)
    mid_x = (min_x + max_x) / 2
    mid_y = (min_y + max_y) / 2

    p1 = Point(mid_x - 20 * dmax, mid_y - dmax)
    p2 = Point(mid_x, mid_y + 20 * dmax)
    p3 = Point(mid_x + 20 * dmax, mid_y - dmax)
    return Triangle(p1, p2, p3)

def is_inside_circumcircle(triangle, point):
    circumcenter = triangle.circumcenter
    radius = np.sqrt((circumcenter.x - triangle.points[0].x)**2 + (circumcenter.y - triangle.points[0].y)**2)
    distance = np.sqrt((circumcenter.x - point.x)**2 + (circumcenter.y - point.y)**2)
    return distance <= radius

def is_edge_shared(edge, triangles):
    count = 0
    for triangle in triangles:
        if edge in triangle.points:
            count += 1
    return count == 2

def polygon_edges(polygon):
    edges = []
    for i in range(len(polygon)):
        edges.append(Edge(polygon[i], polygon[(i + 1) % len(polygon)]))
    return edges

class Edge:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

# Generowanie losowych punktów
#np.random.seed(1)
points = [Point(np.random.rand(), np.random.rand()) for _ in range(20)]

# Triangulacja Bowyera-Watsona
triangulation = bowyer_watson(points)

root = tkinter.Tk()
root.wm_title("Projekt GO")


# Wyświetlanie wynikowych trójkątów
plt.triplot([point.x for point in points], [point.y for point in points], [[point.x, point.y] for point in triangulation])
plt.plot([point.x for point in points], [point.y for point in points], 'o')

fig, ax = plt.gcf(), plt.gca()
# plt.show()


def on_key_press(event):
    print("You pressed {}".format(event.key))


def _quit():
    root.quit()
    root.destroy()

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

def plot():
    try:
        n_points = int(points_entry.get())
        if(n_points<3):
            raise ValueError
        ax.clear()
        points = [Point(np.random.rand(), np.random.rand()) for _ in range(n_points)]
        triangulation = bowyer_watson(points)
        plt.triplot([point.x for point in points], [point.y for point in points],
                    [[point.x, point.y] for point in triangulation])
        plt.plot([point.x for point in points], [point.y for point in points], 'o')
        canvas.draw()
    except ValueError:
        answer.config(text="Enter a valid integer equal or greater than 3.")



tkinter.Button(master=root, text="Draw", command=plot).pack(pady=10)

tkinter.Label(text="Number of points: ").pack(pady=10)
points_entry = tkinter.Entry(root)
points_entry.pack(pady=10)
answer = tkinter.Label(text='')
answer.pack(pady=10)

button_quit = tkinter.Button(master=root, text='Quit', command=_quit)
button_quit.pack(side=tkinter.BOTTOM)
tkinter.mainloop()
