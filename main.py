import numpy as np
import os
import tkinter
import delaunay as D
from tkinter import (
    filedialog as fd,
)
from PIL import Image, ImageOps
from tkinter.messagebox import showinfo
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import cv2

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

def constrained_bowyer_watson(points, boundary_points):
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
    min_x = min(point.x for point in points) + 1
    min_y = min(point.y for point in points) + 1
    max_x = max(point.x for point in points) + 1
    max_y = max(point.y for point in points) + 1

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

# QUAD TREE
class Rect:
    def __init__(self, cx, cy, w, h):
        self.cx, self.cy = cx, cy
        self.w, self.h = w, h
        self.west_edge, self.east_edge = cx - w/2, cx + w/2
        self.north_edge, self.south_edge = cy - h/2, cy + h/2

    def contains(self, point):
        """Is point (a Point object or (x,y) tuple) inside this Rect?"""
        try:
            point_x, point_y = point.x, point.y
        except AttributeError:
            point_x, point_y = point

        return (self.west_edge <= point_x < self.east_edge and
                self.north_edge <= point_y < self.south_edge)

    def intersects(self, other):
        """Does Rect object other interesect this Rect?"""
        return not (other.west_edge > self.east_edge or
                    other.east_edge < self.west_edge or
                    other.north_edge > self.south_edge or
                    other.south_edge < self.north_edge)

    def draw(self, ax, c='gray', lw=1, **kwargs):
        x1, y1 = self.west_edge, self.north_edge
        x2, y2 = self.east_edge, self.south_edge
        ax.plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1], c=c, lw=lw, **kwargs)


class QuadTree:
    def __init__(self, boundary, max_points=4, depth=0):
        """Initialize this node of the quadtree.

        boundary is a Rect object defining the region from which points are
        placed into this node; max_points is the maximum number of points the
        node can hold before it must divide (branch into four more nodes);
        depth keeps track of how deep into the quadtree this node lies.

        """

        self.boundary = boundary
        self.max_points = max_points
        self.points = []
        self.depth = depth
        # A flag to indicate whether this node has divided (branched) or not.
        self.divided = False

    def divide(self):
        """Divide (branch) this node by spawning four children nodes."""

        cx, cy = self.boundary.cx, self.boundary.cy
        w, h = self.boundary.w / 2, self.boundary.h / 2
        # The boundaries of the four children nodes are "northwest",
        # "northeast", "southeast" and "southwest" quadrants within the
        # boundary of the current node.
        self.nw = QuadTree(Rect(cx - w/2, cy - h/2, w, h),
                                    self.max_points, self.depth + 1)
        self.ne = QuadTree(Rect(cx + w/2, cy - h/2, w, h),
                                    self.max_points, self.depth + 1)
        self.se = QuadTree(Rect(cx + w/2, cy + h/2, w, h),
                                    self.max_points, self.depth + 1)
        self.sw = QuadTree(Rect(cx - w/2, cy + h/2, w, h),
                                    self.max_points, self.depth + 1)
        self.divided = True

    def insert(self, point):
        """Try to insert Point point into this QuadTree."""

        if not self.boundary.contains(point):
            # The point does not lie inside boundary: bail.
            return False
        if len(self.points) < self.max_points:
            # There's room for our point without dividing the QuadTree.
            self.points.append(point)
            return True

        # No room: divide if necessary, then try the sub-quads.
        if not self.divided:
            self.divide()

        return (self.ne.insert(point) or
                self.nw.insert(point) or
                self.se.insert(point) or
                self.sw.insert(point))

    def query(self, boundary, found_points):
        """Find the points in the quadtree that lie within boundary."""

        if not self.boundary.intersects(boundary):
            # If the domain of this node does not intersect the search
            # region, we don't need to look in it for points.
            return False

        # Search this node's points to see if they lie within boundary ...
        for point in self.points:
            if boundary.contains(point):
                found_points.append(point)
        # ... and if this node has children, search them too.
        if self.divided:
            self.nw.query(boundary, found_points)
            self.ne.query(boundary, found_points)
            self.se.query(boundary, found_points)
            self.sw.query(boundary, found_points)
        return found_points

    def draw(self, ax):
        """Draw a representation of the quadtree on Matplotlib Axes ax."""

        self.boundary.draw(ax)
        if self.divided:
            self.nw.draw(ax)
            self.ne.draw(ax)
            self.se.draw(ax)
            self.sw.draw(ax)


def getContourPoints(image: str, scale: float):
    img = plt.imread(image, 0)
    # print(img)
    # print(img.shape)
    # plt.imshow(Image.open("C:\\Users\\juziu\\Desktop\\STUDIA\\Inżynieria Obliczeniowa 4. semestr\\Geometria Obliczeniowa\\ProjektGO\\ksztalty.jpg"))
    # plt.show()

    img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    try:
        height, width, channel = img.shape
    except:
        try:
            height, width = img.shape
        except:
            print("image not suitable")
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # threshold and invert so hexagon is white on black background
    thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
    thresh = 255 - thresh

    # get contours
    result = np.zeros_like(img)
    contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    # cv2.imshow("thresh", thresh)

    cntr = np.concatenate(contours, axis=0)
    # cv2.drawContours(result, [cntr], 0, (255, 255, 255), 1)
    # cv2.drawContours(result, contours[0], 0, (255, 255, 255), 1)
    # cv2.imshow("result", result)
    return (cntr, width, height, contours)


# Generowanie losowych punktów
#np.random.seed(1)
points = [Point(np.random.rand(), np.random.rand()) for _ in range(20)]
# Triangulacja Bowyera-Watsona
triangulation = bowyer_watson(points)
print(triangulation)
DT = D.Delaunay_Triangulation()
for p in points:
    DT.AddPoint(D.Point(p.x, p.y))
XS, YS, TS = DT.export()
# print(XS)
# print(YS)
# print(TS)
# print([point.x for point in points])
# print([point.y for point in points])
# print(len(triangulation))
root = tkinter.Tk()
root.wm_title("Projekt GO")


# Wyświetlanie wynikowych trójkątów
# plt.triplot([point.x for point in points], [point.y for point in points], [[point.x, point.y] for point in triangulation])
plt.triplot([point.x for point in points], [point.y for point in points], [[point.x, point.y] for point in triangulation])
plt.plot([point.x for point in points], [point.y for point in points], 'o')
contours = None
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

def quadTree():
    global ax, points, contours
    try:
        coeff = float(scale_coeff.get())
        if 0>coeff or coeff>1:
            raise ValueError
        ax.clear()

        image, width, height, contours = getContourPoints(filename, coeff)
        x, y = image[:, 0, 0], np.subtract(height, image[:, 0, 1])
        y = image[:, 0, 1]
        coords = np.transpose([x, y])

        boundary = Rect(width / 2, height / 2, width, height)

        points = [Point(*coord) for coord in coords]

        qtree = QuadTree(boundary, 1)
        for point in points:
            qtree.insert(point)

        pltImage = Image.open('ksztalty.jpg')
        rsize = pltImage.resize((np.array(pltImage.size) * 1.0).astype(int))
        grayimg = ImageOps.grayscale(rsize)

        #ig, ax = plt.subplots(figsize=(10, 9))
        # plt.imshow(grayimg, cmap='gray')
        ax.scatter(x, y, s=0.3, color='red', alpha=1.0)
        qtree.draw(ax)

        canvas.draw()
    except ValueError:
        answer.config(text="Enter a valid value between 0 and 1.")

triang = None
def delaunay():
    global points, triangulation, contours, triang
    try:
        coeff = float(scale_coeff.get())
        if 0>coeff or coeff>1:
            raise ValueError
        ax.clear()
        print(filename)
        # KONTURY
        image, width, height, contours = getContourPoints(filename, coeff)
        x, y = image[:, 0, 0], np.subtract(height, image[:, 0, 1])
        y = image[:, 0, 1]
        coords = np.transpose([x, y])
        points = [Point(*coord) for coord in coords]
        coords = np.transpose((x,y))

        # ROZMIESZCZENIE PUNKTOW
        image = plt.imread(filename)
        image = cv2.resize(image, (0,0), fx=coeff, fy=coeff, interpolation=cv2.INTER_AREA)

        ret, image = cv2.threshold(image, 100, 100, cv2.THRESH_BINARY)

        scatter = np.where(image ==0)
        x,y = scatter[1][0::3], scatter[0][0::3]
        coords = np.transpose((x,y))
        #plt.scatter(x, y, color='yellow', s=0.2)
        # KONIEC

        print(len(contours))
        # plt.scatter(np.transpose(contours[0])[0], np.transpose(contours[0])[1], color='black', s=0.2)
        canvas.draw()
        # DELAUNAY !!!
        # w kazdym ksztalcie robimy triangulacje

        triang = []
        for contour in contours:
            set_of_points = []

            cv2.drawContours(image, contour, -1, (0, 255, 0), 1)
            # plt.imshow(image)
            for coord in coords:
                pt = tuple([int(round(coord[0])), int(round(coord[1]))])
                if cv2.pointPolygonTest(contour, pt, False) >= 0:

                    set_of_points.append(pt)

            con = np.transpose(contour)
            plt.plot(con[0], con[1], color='r', linewidth=2)
            points = [Point(x,y) for x, y in set_of_points]
            triangulation = bowyer_watson(points)
            plt.triplot([point.x for point in points], [point.y for point in points], [[point.x, point.y] for point in triangulation])
            plt.plot([point.x for point in points], [point.y for point in points], 'o')

            DT = D.Delaunay_Triangulation()
            for p in points:
                DT.AddPoint(D.Point(p.x, p.y))
            XS, YS, TS = DT.export()
            triang.append(TS)

            points = np.transpose(set_of_points)
            # print(points)
            plt.scatter(points[0], points[1], color='red', s=0.2)

        canvas.draw()
    except ValueError:
        answer.config(text="Enter a valid value between 0 and 1.")

def constrained_delaunay():
    global triangulation, points, contours, triang
    try:
        coeff = float(scale_coeff.get())
        if 0>coeff or coeff>1:
            raise ValueError
        ax.clear()
        print(filename)
        # KONTURY
        image, width, height, contours = getContourPoints(filename, coeff)
        x, y = image[:, 0, 0], np.subtract(height, image[:, 0, 1])
        y = image[:, 0, 1]
        coords = np.transpose([x, y])
        points = [Point(*coord) for coord in coords]
        coords = np.transpose((x,y))
        # print(coords)

        # ax.scatter(x, y, s=0.3, color='red', alpha=1.0)

        # ROZMIESZCZENIE PUNKTOW
        image = plt.imread(filename)
        image = cv2.resize(image, (0,0), fx=coeff, fy=coeff, interpolation=cv2.INTER_AREA)

        ret, image = cv2.threshold(image, 100, 100, cv2.THRESH_BINARY)

        scatter = np.where(image ==0)
        x,y = scatter[1][0::3], scatter[0][0::3]
        coords = np.transpose((x,y))
        #plt.scatter(x, y, color='yellow', s=0.2)
        # KONIEC

        print(len(contours))
        # plt.scatter(np.transpose(contours[0])[0], np.transpose(contours[0])[1], color='black', s=0.2)
        canvas.draw()
        # DELAUNAY !!!
        # w kazdym ksztalcie robimy triangulacje

        triang=[]
        for contour in contours:
            set_of_points = []
            #c = np.concatenate(contour, axis=0)
            #c = np.append(c, c[0])
            # c = np.append(contour, contour[0])
            # max_x = np.max(np.transpose(contour)[0])
            # min_x = np.min(np.transpose(contour)[0])
            # max_y = np.max(np.transpose(contour)[1])
            # min_y = np.min(np.transpose(contour)[1])
            # cv2.fillPoly(image, contour, color=(0, 0, 255))
            cv2.drawContours(image, contour, -1, (0, 255, 0), 1)
            # plt.imshow(image)
            for coord in coords:
                pt = tuple([int(round(coord[0])), int(round(coord[1]))])
                # print(coord)
                # print(f'contour: {np.concatenate(contour, axis=0)}')
                # print(cv2.pointPolygonTest(contour, pt, False))
                if cv2.pointPolygonTest(contour, pt, False) >= 0:

                    set_of_points.append(pt)
                # if min_x<=pt[0]<=max_x and min_y<=pt[1]<=max_y:
                #     set_of_points.append(pt)
            con = np.transpose(contour)
            plt.plot(con[0], con[1], color='r', linewidth=2)
            points = [Point(x,y) for x, y in set_of_points]
            concat = np.concatenate(np.concatenate(contours, axis=0), axis=0)
            # print(concat)


            triangulation = constrained_bowyer_watson(points, concat)
            plt.triplot([point.x for point in points], [point.y for point in points], [[point.x, point.y] for point in triangulation])
            #plt.plot([point.x for point in points], [point.y for point in points], 'o')

            # DT = D.Delaunay_Triangulation()
            # for p in points:
            #     DT.AddPoint(D.Point(p.x, p.y))
            # XS, YS, TS = DT.export()
            # triang.append(TS)

            points = np.transpose(set_of_points)

            # print(points)
            #plt.scatter(points[0], points[1], color='red', s=0.2)
            plt.scatter(np.transpose(concat)[0], np.transpose(concat)[1], color='b', s=1.0)

        # points = [Point(np.random.rand(), np.random.rand()) for _ in range(n_points)]
        # triangulation = bowyer_watson(points)
        # plt.triplot([point.x for point in points], [point.y for point in points],
        #             [[point.x, point.y] for point in triangulation])
        # plt.plot([point.x for point in points], [point.y for point in points], 'o')
        canvas.draw()
    except ValueError:
        answer.config(text="Enter a valid value between 0 and 1.")

def delaunay_boundary():
    global ax, points, triangulation, contours, triang
    try:
        coeff = float(scale_coeff.get())
        if 0>coeff or coeff>1:
            raise ValueError
        ax.clear()
        print(filename)
        # KONTURY
        image, width, height, contours = getContourPoints(filename, coeff)
        x, y = image[:, 0, 0], np.subtract(height, image[:, 0, 1])
        y = image[:, 0, 1]
        coords = np.transpose([x, y])
        points = [Point(*coord) for coord in coords]
        coords = np.transpose((x,y))
        # print(coords)

        # ax.scatter(x, y, s=0.3, color='red', alpha=1.0)

        # ROZMIESZCZENIE PUNKTOW
        image = plt.imread(filename)
        image = cv2.resize(image, (0,0), fx=coeff, fy=coeff, interpolation=cv2.INTER_AREA)

        ret, image = cv2.threshold(image, 100, 100, cv2.THRESH_BINARY)

        scatter = np.where(image ==0)
        x,y = scatter[1][0::3], scatter[0][0::3]
        coords = np.transpose((x,y))
        #plt.scatter(x, y, color='yellow', s=0.2)
        # KONIEC

        print(len(contours))
        # plt.scatter(np.transpose(contours[0])[0], np.transpose(contours[0])[1], color='black', s=0.2)
        canvas.draw()
        # DELAUNAY !!!
        # w kazdym ksztalcie robimy triangulacje

        triang=[]
        for contour in contours:
            # set_of_points = []
            #c = np.concatenate(contour, axis=0)
            #c = np.append(c, c[0])
            # c = np.append(contour, contour[0])
            # max_x = np.max(np.transpose(contour)[0])
            # min_x = np.min(np.transpose(contour)[0])
            # max_y = np.max(np.transpose(contour)[1])
            # min_y = np.min(np.transpose(contour)[1])
            # cv2.fillPoly(image, contour, color=(0, 0, 255))
            # cv2.drawContours(image, contour, -1, (0, 255, 0), 1)
            # # plt.imshow(image)
            # for coord in coords:
            #     pt = tuple([int(round(coord[0])), int(round(coord[1]))])
            #     # print(coord)
            #     # print(f'contour: {np.concatenate(contour, axis=0)}')
            #     # print(cv2.pointPolygonTest(contour, pt, False))
            #     if cv2.pointPolygonTest(contour, pt, False) >= 0:
            #
            #         set_of_points.append(pt)
            #     # if min_x<=pt[0]<=max_x and min_y<=pt[1]<=max_y:
            #     #     set_of_points.append(pt)
            contour = contour[:,0]
            con = np.transpose(contour)
            #print(contour)
            print(len(contour))
            plt.plot(con[0], con[1], color='r', linewidth=2)
            points = [Point(x,y) for x, y in contour]
            triangulation = bowyer_watson(points)
            plt.triplot([point.x for point in points], [point.y for point in points], [[point.x, point.y] for point in triangulation])
            plt.plot([point.x for point in points], [point.y for point in points], 'o')

            DT = D.Delaunay_Triangulation()
            for p in points:
                DT.AddPoint(D.Point(p.x, p.y))
            XS, YS, TS = DT.export()
            triang.append(TS)

            points = np.transpose(contour)

            # print(points)
            plt.scatter(points[0], points[1], color='red', s=0.2)

        # points = [Point(np.random.rand(), np.random.rand()) for _ in range(n_points)]
        # triangulation = bowyer_watson(points)
        # plt.triplot([point.x for point in points], [point.y for point in points],
        #             [[point.x, point.y] for point in triangulation])
        # plt.plot([point.x for point in points], [point.y for point in points], 'o')
        canvas.draw()
    except ValueError:
        answer.config(text="Enter a valid value between 0 and 1.")


#OGOLNIE
#todo implementacja kodu quadTree (z poprzedniego pliku)
#todo opcja ładowania siatki niestrukturalnej (delaunay) albo siatki strukturalnej (quadTree) na przycisk
#DELAUNAY
#todo ustalanie współczynnika z jaką gęstością są generowane punkty w "wydzieleniach"
#todo TRZY OPCJE:
#todo 1. podział na prostokąty gdzie są wydzielenia i triangulacja tylko tych punktów
#todo 2. sprawdzanie przy uploadowanych obrazach czy kolor się zgadza w danym miejscu
#todo 3. sprawdzić wykorzystany kod już w folderze "Projekt".

#KONIEC

def save_file():
    current_dir = os.getcwd()
    file = open(f"{current_dir}\plate_mesh.dat", "w")
    for i in range(len(contours)):
        contour = contours[i][:, 0]
        triangulation = triang[i]
        points = [Point(x, y) for x, y in contour]
        nb_nodes = len(points)
        nb_elements = len(triangulation)
        # print(np.transpose(points))

        file.write("{} {}\n".format(nb_nodes, nb_elements))
        for i, node in enumerate(points):
            file.write("{} {} {}\n".format(i, node.x, node.y))
        for j, elem in enumerate(triangulation):
            file.write("{} {} {} {}\n".format(j, elem[0], elem[1], elem[2]))
    file.close()

def save_quad_file():
    current_dir = os.getcwd()
    file = open(f"{current_dir}\plate_quad_mesh.dat", "w")
    for i in range(len(contours)):
        contour = contours[i][:, 0]
        points = [Point(x, y) for x, y in contour]
        nb_nodes = len(points)
        # print(np.transpose(points))

        file.write("{}\n".format(nb_nodes))
        for i, node in enumerate(points):
            file.write("{} {} {}\n".format(i, node.x, node.y))
    file.close()

tkinter.Label(text="Coeffient: ").pack(pady=10)
scale_coeff = tkinter.Entry(root)
scale_coeff.pack(pady=10)
answer = tkinter.Label(text='')
answer.pack(pady=10)

tkinter.Button(master=root, text="Draw Delaunay", command=delaunay).pack(pady=10)
tkinter.Button(master=root, text="Draw Delaunay only boundaries", command=delaunay_boundary).pack(pady=10)
tkinter.Button(master=root, text="Draw Constrained Delaunay", command=constrained_delaunay).pack(padx=10)
tkinter.Button(master=root, text="Draw QuadTree", command=quadTree).pack(padx=10)

filename = None
def select_file():
    global filename
    filetypes = (
        ('Jpeg', '*.jpg'),
        ('All files', '*.*')
    )

    filename = fd.askopenfilename(
        title='Open a file',
        initialdir='.',
        filetypes=filetypes
    )

    # showinfo(
    #     title='Selected file',
    #     message=filename
    # )

button_quit = tkinter.Button(master=root, text='Quit', command=_quit)
button_open_file = tkinter.Button(master=root, text='Upload file', command=select_file)
button_quit.pack(side=tkinter.BOTTOM)
button_open_file.pack(expand=True)
tkinter.Button(master=root, text="Save Delaunay file", command=save_file).pack(pady=10)
tkinter.Button(master=root, text="Save QuadTree file", command=save_quad_file).pack(pady=10)
tkinter.mainloop()
