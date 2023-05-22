import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageOps
import numpy as np
import cv2


class Point:
    def __init__(self, x, y, payload=None):
        self.x, self.y = x, y


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
    # read image
    img = cv2.imread(image)
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    height, width, channel = img.shape
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # threshold and invert so hexagon is white on black background
    thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
    #ret, image = cv2.threshold(img, 0, 100, cv2.THRESH_BINARY)
    print(thresh)
    thresh = 255 - thresh

    # get contours
    result = np.zeros_like(img)
    contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    #cv2.imshow("thresh", thresh)
    cntr = np.concatenate(contours, axis=0)
    cv2.drawContours(result, [cntr], 0, (255, 255, 255), 1)
    #cv2.imshow("result", result)
    return (cntr, width, height)

    # show thresh and contour
    #cv2.imshow("thresh", thresh)
    #cv2.imshow("result", result)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

#image, width, height = getContourPoints('points.jpg')
#image, width, height = getContourPoints('bg.jpg', 1.0)
image, width, height = getContourPoints('ksztalty.jpg', 1.0)
print(width, height)
boundary = Rect(width/2, height/2, width, height)

x, y = image[:,0,0], np.subtract(height, image[:,0,1])#,  image[:][0][1]
y = image[:,0,1]

coords = np.transpose([x, y])
points = [Point(*coord) for coord in coords]

qtree = QuadTree(boundary, 1)
for point in points:
    qtree.insert(point)

#pltImage = Image.open('bg.jpg')
pltImage = Image.open('ksztalty.jpg')
rsize = pltImage.resize((np.array(pltImage.size)*1.0).astype(int))
grayimg = ImageOps.grayscale(rsize)

fig, ax = plt.subplots(figsize=(10,9))
plt.imshow(grayimg, cmap='gray')
ax.scatter(x, y, s=0.3, color='red', alpha=0.0)
qtree.draw(ax)
plt.show()
