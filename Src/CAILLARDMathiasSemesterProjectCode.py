#CAILLARD Mathias.
#Coded used for the project entitled "Modelling of roads and road interchanges"

""" In this file, you'll find the Python code of the different tools that I build throughout the project."""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib import lines
import math
import random as rd
import scipy.stats


def circleTreePoints() :
    """ This function draw a circle that goes throught 3 points whose coordinates are given by x and y"""

    #Random sample of points
    x = np.array([3,5,6])
    y = np.array([3,5,12])

    [x1,x2,x3] = x
    [y1,y2,y3] = y

    """ The equation of the cirle is given by the following equation :

    det([[x**2,y**2,x,y,1],
         [x1**2,y1**2,x1,y1,1],
         [x2**2,y2**2,x2,y2,1],
         [x3**2,y3**2,x3,y3,1]]) = 0

    which is tantamount to the equation :

    a(x**2 + y**2) + b*x + c*y + d = 0
         """
    a = x1*y2 - x1*y3 - y1*x2 + y1*x3 + x2*y3 - y2*x3

    b = (x1**2)*y3 + (y1**2)*y3 - y2*(x1**2) - (y1**2)*y2 + y1*(x2**2) + y1*(y2**2) - y1*(x3**2) - y1*(y3**2) - y3*(x2**2) - y3*(y2**2) + (x3**2)*y2 + y2*(y3**2)

    c = - (x1**2)*x3 - (y1**2)*x3 + x2*(x1**2) + (y1**2)*x2 - x1*(x2**2) - x1*(y2**2) + x1*(x3**2) + x1*(y3**2) + x3*(x2**2) + x3*(y2**2) - (x3**2)*x2 - x2*(y3**2)

    d = y2*x3*(x1**2 + y1**2) - x2*y3*(x1**2 + y1**2) + x1*y3*(x2**2 + y2**2) - x1*y2*(x3**2 + y3**2) + y1*x2*(x3**2 + y3**2) - y1*x3*(x2**2 + y2**2)

    # (xc,yc) : coordinate of the center of the circle
    xc = -b/(2*a)
    yc = -c/(2*a)

    # RR : the square of the radius
    RR = (b**2 + c**2)/((2*a)**2) - (d/a)
    R = math.sqrt(RR)


    circle = plt.Circle((xc,yc),R,fill=False)

    """considering z<i> whose affix is the point <i> i.e. (x<i>,y<i>). theta<i> is the principal complex angle of z<i> considering the affine space centered on the affix of (xc + i*yc) """


    if (x1-xc) >= 0 :
        theta1 = math.atan((y1-yc)/(x1-xc))
    else :
        theta1 = math.atan((y1-yc)/(x1-xc)) + math.pi

    if (x3-xc) >= 0 :
        theta2 = math.atan((y3-yc)/(x3-xc))
    else :
        theta2 = math.atan((y3-yc)/(x3-xc)) + math.pi

    arc_circle = patches.Arc((xc,yc),2*R,2*R,0,theta1*180/math.pi,theta2*180/math.pi, color = "red",linewidth = 3)

    """gca : get the current axis"""
    ax = plt.gca()
    ax.scatter(x,y)
    ax.set_aspect('equal', 'box')
    ax.add_patch(circle)
    ax.add_patch(arc_circle)
    ax.set_title("drawing a circle that goes through 3 points")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    plt.show()


def circleMultiplePoints(points) :

    """ points : a list. each element (point) is a couple (x,y) that represents the coordinate of the point.

    This function displays points, and draws circles that goes throught them, three by three.

    Example : circleMultiplePoints(pointsGenerator(0.5,10,50,lambda x : math.cos(x) + math.sin(x)**2,0.2, False))"""

    xs = []
    ys = []
    for point in points :
        xs.append(point[0])
        ys.append(point[1])

    """ A list. LinkPointsCirlces[i] is a list that contains the index of circles that goes throught the point points[i] """
    LinkPointsCircles = [[] for k in range(len(points))]



    Circles = []
    Arc_circles = []

    i = 0

    while i <= len(points)-3 :
        [x1,x2,x3] = [xs[i],xs[i+1],xs[i+2]]
        [y1,y2,y3] = [ys[i],ys[i+1],ys[i+2]]
        Circles.append(calculateCircleAndArc(x1,x2,x3,y1,y2,y3)[0])
        Arc_circles.append(calculateCircleAndArc(x1,x2,x3,y1,y2,y3)[1])


        LinkPointsCircles[i].append(i//2)
        LinkPointsCircles[i+1].append(i//2)
        LinkPointsCircles[i+2].append(i//2)

        i +=2 #Circles have at least one point that is "owned" by another circle





    X = []
    Y = []
    for i in range(len(points)) :
        for circleIndex in LinkPointsCircles[i] :
            X.append(i)
            Y.append(1/(Circles[circleIndex].get_radius()))


    figure, axis = plt.subplots(1,2)
    axis[0].scatter(xs,ys)
    axis[0].autoscale()
    axis[0].set_aspect('equal', 'box')
    axis[0].set_title("Arc of circles passing through points")

    for circle in Circles :
        axis[0].add_patch(circle)

    for arc_circle in Arc_circles :
        axis[0].add_patch(arc_circle)


    axis[1].plot(X,Y)
    axis[1].set_title("Curvature of circles")
    axis[1].set_xlabel("number of the points")
    axis[1].set_ylabel("(1/Radius)")

    plt.grid()
    plt.show()




def circleMultiplePointsReturn(points) :

    """ points : a list. each element (point) is a couple (x,y) that represents the coordinate of the point.

    This function return arc of circles ,  that goes throught them points, three by three."""

    xs = []
    ys = []
    for point in points :
        xs.append(point[0])
        ys.append(point[1])

    """ A list. LinkPointsCirlces[i] is a list that contains the index of circles that goes throught the point points[i] """
    LinkPointsCircles = [[] for k in range(len(points))]



    Circles = []
    Arc_circles = []

    i = 0

    while i <= len(points)-3 :
        [x1,x2,x3] = [xs[i],xs[i+1],xs[i+2]]
        [y1,y2,y3] = [ys[i],ys[i+1],ys[i+2]]
        Circles.append(calculateCircleAndArc(x1,x2,x3,y1,y2,y3)[0])
        Arc_circles.append(calculateCircleAndArc(x1,x2,x3,y1,y2,y3)[1])


        LinkPointsCircles[i].append(i//2)
        LinkPointsCircles[i+1].append(i//2)
        LinkPointsCircles[i+2].append(i//2)

        i +=2 #Circles have at least one point that is "owned" by another circle

    return [Circles, Arc_circles]





def calculateCircleAndArc(x1,x2,x3,y1,y2,y3) :
    """ return : a circle that goes throught the 3 points (x1,y1), (x2,y2) and (x3,y3)"""

    a = x1*y2 - x1*y3 - y1*x2 + y1*x3 + x2*y3 - y2*x3

    b = (x1**2)*y3 + (y1**2)*y3 - y2*(x1**2) - (y1**2)*y2 + y1*(x2**2) + y1*(y2**2) - y1*(x3**2) - y1*(y3**2) - y3*(x2**2) - y3*(y2**2) + (x3**2)*y2 + y2*(y3**2)

    c = - (x1**2)*x3 - (y1**2)*x3 + x2*(x1**2) + (y1**2)*x2 - x1*(x2**2) - x1*(y2**2) + x1*(x3**2) + x1*(y3**2) + x3*(x2**2) + x3*(y2**2) - (x3**2)*x2 - x2*(y3**2)

    d = y2*x3*(x1**2 + y1**2) - x2*y3*(x1**2 + y1**2) + x1*y3*(x2**2 + y2**2) - x1*y2*(x3**2 + y3**2) + y1*x2*(x3**2 + y3**2) - y1*x3*(x2**2 + y2**2)

    xc = -b/(2*a)
    yc = -c/(2*a)
    RR = (b**2 + c**2)/((2*a)**2) - (d/a)
    R = math.sqrt(RR)

    circle = plt.Circle((xc,yc),R,fill=False,alpha = 0.3)



    """considering z<i> whose affix is the point <i> i.e. (x<i>,y<i>). theta<i> is the principal complex angle of z<i> considering the affine space centered on the affix of (xc + i*yc) """

    if (x1-xc) >= 0 :
        theta1 = math.atan((y1-yc)/(x1-xc))
    else :
        theta1 = math.atan((y1-yc)/(x1-xc)) + math.pi

    if (x2-xc) >= 0 :
        theta2 = math.atan((y2-yc)/(x2-xc))
    else :
        theta2 = math.atan((y2-yc)/(x2-xc)) + math.pi

    if (x3-xc) >= 0 :
        theta3 = math.atan((y3-yc)/(x3-xc))
    else :
        theta3 = math.atan((y3-yc)/(x3-xc)) + math.pi


    """The arc of circle that connect the point i and i+2 must go through the point i+1"""
    if theta3 < theta2 and theta2 < theta1 :
        (theta1,theta3) = (theta3, theta1)


    arc_circle = patches.Arc((xc,yc),2*R,2*R,0,theta1*180/math.pi,theta3*180/math.pi, color = "red",linewidth = 3)


    return [circle, arc_circle]



def pointsGenerator(xMin,xMax,numberOfPoints,function,noise,show) :
    """ Return : a list of points. A point is a list [x,y], where x and y represents the coordinates

    example : pointsGenerator(-4,7,30,lambda x : math.sin(x),0.4, True)"""


    xs = np.linspace(xMin,xMax,numberOfPoints)
    f = function
    ys = [f(x) for x in xs]
    res = []
    #adding noise
    for i in range(len(ys)) :
        ys[i] = ys[i] + (2*rd.random()-1)*(noise)

    for i in range(len(ys)) :
        res.append([xs[i],ys[i]])

    if show :
        plt.scatter(xs,ys)
        plt.show()
    return res


def regressionLines(points, thresholdEnd,limrvalue,thresholdCurve) :
    threshold_init = thresholdEnd
    """
    idea to detect straight lines : I take the first point, and the second one. While the line regression between the first and the second point is good enough, I take the next point for the "second point". if the while is "too short", then it wasn't a line, but a curve. Then the "last point" become the "first", we we do it again until the very last point.

    example : regressionLines(pointsGenerator(0.5,10,50,lambda x : math.cos(x) + math.sin(x)**2,0.1, False),3,0.95,5)
    """
    axis = plt.gca()
    xs = []
    ys = []
    for point in points :
        xs.append(point[0])
        ys.append(point[1])

    thresholdStart = 0
    res_reg = []


    xs_init = xs[thresholdStart:thresholdEnd]
    ys_init = ys[thresholdStart:thresholdEnd]
    xs_init = np.array(xs_init)

    res = scipy.stats.linregress(xs_init, ys_init)


    while thresholdEnd < len(xs) :

        while ((res.rvalue**2 > limrvalue) and (thresholdEnd < len(xs))):

            print("thresholdStart : " + str(thresholdStart))
            print("thresholdEnd : " + str(thresholdEnd))

            xs_init = xs[thresholdStart:thresholdEnd]
            ys_init = ys[thresholdStart:thresholdEnd]

            res = scipy.stats.linregress(xs_init, ys_init)
            thresholdEnd += 1
            print("rvalue**2 : " + str(res.rvalue**2)+ "\n")


        if (thresholdEnd - thresholdStart) >= thresholdCurve : #if it's a line
            xs_init = np.array(xs_init)
            plt.plot(xs_init, res.intercept + res.slope*xs_init, 'r', label='fitted line')
        else :

            print("courbe!")
            print("thresholdStart : " + str(thresholdStart))
            print("thresholdEnd : " + str(thresholdEnd) + "\n")
            """
            pointsCurve = []
            k = thresholdStart
            while k < thresholdEnd :
                pointsCurve.append([xs[k],ys[k]])
                k+=1

            Circles = circleMultiplePointsReturn(pointsCurve)[0]
            Arc_circles = circleMultiplePointsReturn(pointsCurve)[1]
            for circle in Circles :
                axis.add_patch(circle)

            for arc_circle in Arc_circles :
                axis.add_patch(arc_circle)
                """







        thresholdStart = thresholdEnd - 2
        thresholdEnd += threshold_init


        xs_init = xs[thresholdStart:thresholdEnd]
        ys_init = ys[thresholdStart:thresholdEnd]
        res = scipy.stats.linregress(xs_init, ys_init)



    plt.title("Algorithm trying to find straight lines")
    axis.set_aspect('equal', 'box')
    plt.legend()
    plt.scatter(xs,ys,alpha=0.4)
    plt.show()


def roadConstructor(show, Rsigned, xs, ys, signList) :
    """ A road is a 3-uple of three elements. the first one is a line of straight roads. The second one is a list of curvatures that connect those straight roads. The last one corresponds to the list of angle used to define the arc of circles. A road must starts with a straight line, and ends with a straight line.

    Show : Bool. indicate whether of not a plot should be printed
    Rsigned : list of signed radiuses. Positive --> outer tangent. Negative --> inner tangent. Except for the last one that doesn't have a tangent associated. Instead, the sign of the last one determine which couple of tangent (inner-outer) we should select : botten or upper.

    xs : list of x-coordinates of circles
    ys : list of y-coordinates of circles
    roadConstructor(True, [1,-1.5,1,-2.2,2,-3,1.6], [-3,2,5,8,0,10,15], [0,4,-2,3,15,20,9], [1,-1,-1,1,1,-1])

     """
    R = []
    for signedRadius in Rsigned :
        R.append(abs(signedRadius))

    #data (input)
    points = []

    #sign determine which side of outer-inner pair we choose
    if Rsigned[-1] > 0 :
        sign = 1
    else :
        sign = -1



    #for storing tangent points
    tangentPointsL1 = []
    tangentPointsL2 = []

    #for storing arc of circles and tangents
    arc_circles = []
    arc_circles_theta = []
    tangents = []


    #creating points
    for i in range(len(ys)) :
        points.append([xs[i],ys[i]])


    #coordinates of points (center of circles)
    indexForPoints = 0
    while indexForPoints < len(points) - 1:
        xL2 = points[indexForPoints + 1][0]
        xL1 = points[indexForPoints][0]
        yL2 = points[indexForPoints + 1][1]
        yL1 = points[indexForPoints][1]

        #Radius
        R1 = R[indexForPoints]
        R2 = R[indexForPoints + 1]

        #vector L1L2
        xL1L2 = xL2 - xL1
        yL1L2 = yL2 - yL1
        vectorL1L2 = np.array([[xL1L2], [yL1L2]])
        normVectorL1L2 = np.linalg.norm(vectorL1L2)

        #length of the tangent
        if Rsigned[indexForPoints] > 0 :
            d = math.sqrt( normVectorL1L2**2 - (R2 - R1)**2)
        else :
            d = math.sqrt( normVectorL1L2**2 - (R2 + R1)**2)

        #angle used to build the tangent
        sign = signList[indexForPoints]

        if Rsigned[indexForPoints] > 0 :
            alpha = sign*np.arcsin((R2 - R1)/normVectorL1L2)
        else :
            alpha = sign*np.arcsin((R2 + R1)/normVectorL1L2)


        #rotation matrix based on the angle alpha
        rotationAlpha = np.array([[np.cos(alpha),-np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])

        #Creation of the intermediate point A
        vectorOL1 = np.array([[xL1],[yL1]])
        vectorOA = vectorOL1 + (d / normVectorL1L2) * np.matmul(rotationAlpha,vectorL1L2)
        xA = vectorOA[0][0]
        yA = vectorOA[1][0]

        #creation of the point B to build tangent (second circle L2)
        vectorOL2 = np.array([[xL2],[yL2]])
        vectorL2A = np.array([[xA - xL2], [yA - yL2]])

        if Rsigned[indexForPoints] > 0 :
            vectorOB = vectorOL2 + (R2 / (R2 - R1)) * vectorL2A
        else :
            vectorOB = vectorOL2 + (R2 / (R2 + R1)) * vectorL2A

        xB = vectorOB[0][0]
        yB = vectorOB[1][0]
        tangentPointsL2.append([xB,yB])


        #creation of the point C to build tangent (first circle L1)
        if Rsigned[indexForPoints] > 0 :
            vectorOC = vectorOL1 + (R1 / (R2 - R1)) * vectorL2A
        else :
            vectorOC = vectorOL1 - (R1 / (R2 + R1)) * vectorL2A

        xC = vectorOC[0][0]
        yC = vectorOC[1][0]
        tangentPointsL1.append([xC,yC])


        #creating tangents
        tangents.append(lines.Line2D([xC,xB],[yC,yB]))

        indexForPoints += 1


    #creation of arc of circles
    indexForPoints = 1
    while indexForPoints < len(points) - 1 :

        xc = points[indexForPoints][0]
        yc = points[indexForPoints][1]
        [xL1,yL1] = tangentPointsL1[indexForPoints]
        [xL2,yL2] = tangentPointsL2[indexForPoints-1]
        radius = R[indexForPoints]


        if (xL2-xc) >= 0 :
            theta2 = math.atan((yL2-yc)/(xL2-xc))
        else :
            theta2 = math.atan((yL2-yc)/(xL2-xc)) + math.pi

        if (xL1-xc) >= 0 :
            theta1 = math.atan((yL1-yc)/(xL1-xc))
        else :
            theta1 = math.atan((yL1-yc)/(xL1-xc)) + math.pi

        #inner-inner tangents : we have to switch angles
        if (Rsigned[indexForPoints-1] < 0 and signList[indexForPoints-1] < 0) or (Rsigned[indexForPoints-1] > 0 and signList[indexForPoints-1] < 0):
            theta1, theta2 = theta2, theta1


        arc_circle = patches.Arc((xc,yc),2*radius,2*radius,0,theta1*180/math.pi,theta2*180/math.pi, color = "red",linewidth = 3)
        arc_circles.append(arc_circle)
        arc_circles_theta.append([theta1,theta2])
        indexForPoints += 1


    if show :

        #plotting and showing the results
        ax = plt.gca()
        ax.set_aspect('equal', 'box')


        #plotting points
        plt.scatter(xs,ys)

        #plotting circles around points
        i = 0
        while i < len(points) :
            xc = points[i][0]
            yc = points[i][1]
            r = R[i]
            circle = plt.Circle((xc,yc),r,fill=False,alpha = 0.3)
            ax.add_patch(circle)
            i +=1

        #plotting tangent point B on second circle L2
        for pointB in tangentPointsL2 :
            plt.scatter(pointB[0],pointB[1])

        #plotting tangent point C on first circle L1
        for pointC in tangentPointsL1 :
            plt.scatter(pointC[0],pointC[1])


        #plotting the tangent
        for tangent in tangents :
            ax.add_line(tangent)

        #plotting arc of circles
        for arc_circle in arc_circles :
            ax.add_patch(arc_circle)

        #show the plot
        plt.title("Artificial road")
        plt.show()


    #returning a road
    return(arc_circles, tangents, arc_circles_theta)



def sampleRoad(road) :


    """ A road is a 3-uple of two elements. the first one is a line of straight roads. The second one is a list of curvatures that connect those straight roads. The last one corresponds to the list of angle used to define the arc of circles.  This function must returns a list of points corresponding to the sampling process"""

    num = 30

    lines_ = road[1]
    curvatures = road[0]
    thetas = road[2]

    print(thetas)


    samplesLines = [[] for i in range(len(lines_))]
    samplesCurvatures = [[] for i in range(len(curvatures))]



    #sampling lines
    indexLine = 0
    while indexLine < len(lines_) :
        line = lines_[indexLine]
        dataLine = line.get_data()

        xstart = dataLine[0][0]
        xend = dataLine[0][1]
        ystart = dataLine[1][0]
        yend = dataLine[1][1]

        xsampleLine = np.linspace(xstart,xend,num)
        ysampleLine = np.linspace(ystart,yend,num)

        indexSample = 0
        while indexSample < len(xsampleLine) :
            samplesLines[indexLine].append([xsampleLine[indexSample],ysampleLine[indexSample]])
            indexSample += 1
        indexLine += 1

    #sampling curvatures
    indexCurvature = 0
    while indexCurvature < len(curvatures) :
        curvature = curvatures[indexCurvature]
        radius = (curvature.get_width()) / 2
        (xc,yc) = curvature.get_center()

        theta1 = thetas[indexCurvature][0]
        theta2 = thetas[indexCurvature][1]
        print(theta1, theta2)
        angles = np.linspace(min(theta1,theta2),max(theta1,theta2),num)
        for angle in angles :
            samplesCurvatures[indexCurvature].append([radius*np.cos(angle)+xc,radius*np.sin(angle)+yc])

        indexCurvature += 1



    #creating sample
    sample = []
    indexSample = 0
    while indexSample < len(lines_)-1 :
        for point in samplesLines[indexSample] :
            sample.append(point)
        for point in samplesCurvatures[indexSample] :
            sample.append(point)
        indexSample += 1
    for point in samplesLines[indexSample] :
        sample.append(point)

    indexSample = 0
    xSample = []
    ySample = []
    while indexSample < len(sample) :
        xSample.append(sample[indexSample][0])
        ySample.append(sample[indexSample][1])
        indexSample += 1

    #plotting sample
    plt.scatter(xSample,ySample)
    plt.title("Sample road")
    ax = plt.gca()
    ax.set_aspect('equal', 'box')

    plt.show()

    return(sample)



def roadConstructor2(startingPoint, startDistance, alpha, listAngles, listDistances, radius,showCircle = True) :
    """ Points are lists of two elements. Angles are float between  -pi and pi. distances are positive floats. alpha is the starting angle. PointWithPrime = either the first one, either point after the turn. PointWithoutPrime : before the turn radius : radius of circles.

    example :
    roadConstructor2([0,0],2,math.pi/3,[np.pi/4, np.pi/3, -np.pi/6, -5*np.pi/6, np.pi*7/9],[2,5,4,6,9],2,False)


    """

    #Useful matrix
    RotationPositive90 = np.array([[np.cos(np.pi/2),-np.sin(np.pi/2)], [np.sin(np.pi/2), np.cos(np.pi/2)]])

    RotationNegative90 = np.array([[np.cos(-np.pi/2),-np.sin(-np.pi/2)], [np.sin(-np.pi/2), np.cos(-np.pi/2)]])

    #Configuring
    ax = plt.gca()
    ax.set_aspect('equal', 'box')

    points = []
    """ points[0] : starting point.
        points[i] : point. for i >= 1 and i <= len(listAngles)
        points[len(listAngles+1] : ending point
    """

    arcOfCircles = []
    """ arcOfCircles[i] : arcOfCircle between  point<i+2> and poin<i+2>Prime.
    i >= 0 and i <= len(listAngles) - 1
    """
    circles = []
    """ a list of list of two circle.
    Circles[i] : [circleC1,circleC2] list of circles for the points i+2.
    i >= 0 and i <= len(listAngles) - 1"""

    lines_ = []

    arcOfCircles_theta = []

    #Creating P1prime (first point)
    P1prime = startingPoint
    points.append(P1prime)



    #Creating P2 (second point)
    rotationAlpha = np.array([[np.cos(alpha),-np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])

    x1prime = P1prime[0]
    y1prime = P1prime[1]

    vectorOP1prime = np.array([[x1prime],[y1prime]])

    l1 = startDistance
    l1primeprime = l1*np.sin(alpha)

    l1prime = np.sqrt(l1**2 - l1primeprime**2)

    vectorP1primeP1primeprime = np.array([[x1prime + l1prime],[y1prime]])

    vectorP1primeP2 = np.matmul(rotationAlpha, vectorP1primeP1primeprime)

    vectorOP2 = vectorP1primeP2 + vectorOP1prime
    [x2,y2] = [vectorOP2[0][0],vectorOP2[1][0]]
    P2 = [x2, y2]
    points.append(P2)


    lines_.append(lines.Line2D([x1prime,x2],[y1prime,y2]))


    indexAngle = 0
    while indexAngle < len(listAngles) :


        #generating circle for turn
        P1prime = points[len(points)-2]
        P2 = points[len(points)-1]

        xP1prime, yP1prime = P1prime[0],P1prime[1]

        xP2, yP2 = P2[0], P2[1]

        vectorOP2 = np.array([[xP2],[yP2]])

        vectorP1primeP2 = np.array([[xP2-xP1prime],[yP2-yP1prime]])

        normVectorP1primeP2 = np.linalg.norm(vectorP1primeP2)




        #generating C1

        vectorP2C1 = (radius / normVectorP1primeP2) * np.matmul(RotationPositive90, vectorP1primeP2)

        vectorOC1 = vectorOP2 + vectorP2C1
        [xC1, yC1] = [vectorOC1[0][0],vectorOC1[1][0]]

        #generating C2

        vectorP2C2 = (radius / normVectorP1primeP2) * np.matmul(RotationNegative90, vectorP1primeP2)

        vectorOC2 = vectorOP2 + vectorP2C2
        [xC2, yC2] = [vectorOC2[0][0],vectorOC2[1][0]]

        #generating circle C1
        circleC1 = patches.Circle((xC1,yC1), radius, fill=False, alpha = 0.3)


        #generating circle C2
        circleC2 = patches.Circle((xC2,yC2), radius, fill=False, alpha = 0.3)



        circles.append([circleC1,circleC2])


        #generating P2prime
        angle = listAngles[indexAngle]

        RotationAngle = np.array([[np.cos(angle),-np.sin(angle)], [np.sin(angle), np.cos(angle)]])

        if angle >= 0 :
            circleTurn = circles[indexAngle][0]
        else :
            circleTurn = circles[indexAngle][1]

        (xC,yC) = circleTurn.get_center()


        if circleTurn == circles[indexAngle][0] :
            vectorP2C = vectorP2C1
        else :
            vectorP2C = vectorP2C2

        vectorCP2prime = np.matmul(RotationAngle, (-1)*vectorP2C)

        vectorP2P2prime = vectorP2C + vectorCP2prime
        vectorOP2prime = vectorOP2 + vectorP2P2prime

        [xP2prime, yP2prime] = [vectorOP2prime[0][0],vectorOP2prime[1][0]]
        P2prime = [xP2prime,yP2prime]
        points.append(P2prime)

        #generating arc of circle
        if (xP2-xC) >= 0 :
            theta2 = math.atan((yP2-yC)/(xP2-xC))
        else :
            theta2 = math.atan((yP2-yC)/(xP2-xC)) + math.pi

        if (xP2prime-xC) >= 0 :
            theta1 = math.atan((yP2prime-yC)/(xP2prime-xC))
        else :
            theta1 = math.atan((yP2prime-yC)/(xP2prime-xC)) + math.pi

        if circleTurn == circles[indexAngle][0] :
            theta1, theta2 = theta2, theta1

        arcOfCircles_theta.append([theta1,theta2])
        arc_circle = patches.Arc((xC,yC),2*radius,2*radius,0,theta1*180/math.pi ,theta2*180/math.pi, color = "red",linewidth = 3)

        arcOfCircles.append(arc_circle)
        ax.add_patch(arc_circle)




        #Creation of P3
        rotationalAngle = np.array([[np.cos(angle),-np.sin(angle)], [np.sin(angle), np.cos(angle)]])

        d = listDistances[indexAngle]

        vectorP2primeP3 = d / (normVectorP1primeP2) * np.matmul(rotationalAngle, vectorP1primeP2)

        vectorOP3 = vectorOP2prime + vectorP2primeP3
        [xP3,yP3] = [vectorOP3[0][0], vectorOP3[1][0]]
        points.append([xP3,yP3])


        #Creation of straight line between P2prime and P3
        P3 = points[len(points)-1]
        P2prime = points[len(points)-2]
        xP3, yP3 = P3[0], P3[1]
        xP2prime, yP2prime = P2prime[0], P2prime[1]
        lines_.append(lines.Line2D([xP2prime,xP3],[yP2prime,yP3]))

        indexAngle += 1




     #display
    xs = []
    ys = []
    for point in points :
        xs.append(point[0])
        ys.append(point[1])

    for line in lines_ :
        ax.add_line(line)

    if showCircle :
        for coupleCircle in circles :
            for circle in coupleCircle :
                ax.add_patch(circle)

    plt.scatter(xs,ys)
    plt.show()

    #returning a road
    return(arcOfCircles, lines_, arcOfCircles_theta)













