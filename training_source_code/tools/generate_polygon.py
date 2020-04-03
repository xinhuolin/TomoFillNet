import math, random
from PIL import Image, ImageDraw
from tqdm import tqdm


def generatePolygon(ctrX, ctrY, aveRadius, irregularity=0.5, spikeyness=0.5, numVerts=10):
    '''Start with the centre of the polygon at ctrX, ctrY, 
    then creates the polygon by sampling points on a circle around the centre. 
    Randon noise is added by varying the angular spacing between sequential points,
    and by varying the radial distance of each point from the centre.

    Params:
    ctrX, ctrY - coordinates of the "centre" of the polygon
    aveRadius - in px, the average radius of this polygon, this roughly controls how large the polygon is, really only useful for order of magnitude.
    irregularity - [0,1] indicating how much variance there is in the angular spacing of vertices. [0,1] will map to [0, 2pi/numberOfVerts]
    spikeyness - [0,1] indicating how much variance there is in each vertex from the circle of radius aveRadius. [0,1] will map to [0, aveRadius]
    numVerts - self-explanatory

    Returns a list of vertices, in CCW order.
    '''

    irregularity = clip(irregularity, 0, 1) * 2 * math.pi / numVerts
    spikeyness = clip(spikeyness, 0, 1) * aveRadius

    # generate n angle steps
    angleSteps = []
    lower = (2 * math.pi / numVerts) - irregularity
    upper = (2 * math.pi / numVerts) + irregularity
    sum = 0
    for i in range(numVerts):
        tmp = random.uniform(lower, upper)
        angleSteps.append(tmp)
        sum = sum + tmp

    # normalize the steps so that point 0 and point n+1 are the same
    k = sum / (2 * math.pi)
    for i in range(numVerts):
        angleSteps[i] = angleSteps[i] / k

    # now generate the points
    points = []
    angle = random.uniform(0, 2 * math.pi)
    for i in range(numVerts):
        r_i = clip(random.gauss(aveRadius, spikeyness), 0, 2 * aveRadius)
        x = ctrX + r_i * math.cos(angle)
        y = ctrY + r_i * math.sin(angle)
        points.append((int(x), int(y)))

        angle = angle + angleSteps[i]

    return points


def clip(x, min, max):
    if (min > max):
        return x
    elif (x < min):
        return min
    elif (x > max):
        return max
    else:
        return x


def shake(num, range=(0.9, 1.1)):
    num = num* ( random.random()* (range[1] - range[0]) + range[0])
    return num


def generateShape():
    verts = generatePolygon(ctrX=shake(128), ctrY=shake(128), aveRadius=shake(random.randint(20, 50)),
                            irregularity=shake(random.random() * 0.5), spikeyness=shake(random.random() * 0.4),
                            numVerts=random.randint(3, 60))
    verts_2 = generatePolygon(ctrX=shake(128), ctrY=shake(128), aveRadius=shake(random.randint(20, 30)),
                              irregularity=random.random() * 0.5, spikeyness=shake(0.1), numVerts=random.randint(3, 100))
    verts_3 = generatePolygon(ctrX=shake(128), ctrY=shake(128), aveRadius=random.randint(15, 25),
                              irregularity=random.random() *shake(0.5), spikeyness=random.random() * shake(0.9),
                              numVerts=random.randint(3, 20))

    im = Image.new('L', (256, 256), 0)
    draw = ImageDraw.Draw(im)
    draw.polygon(verts, fill=255)
    if random.random() > 0.5:
        draw.polygon(verts_2, fill=255)
    if random.random() > 0.5:
        draw.polygon(verts_3, fill=0)

    return im


def generate_dataset(root="../new_data_inner_inpaint/original_img", num=1000):
    for i in tqdm(range(num)):
        im = generateShape()
        im.save("%s/%04d.png" % (root, i))


if __name__ == '__main__':
    im = generateShape()
    im.show()
    generate_dataset()

# now you can save the image (im), or do whatever else you want with it.
