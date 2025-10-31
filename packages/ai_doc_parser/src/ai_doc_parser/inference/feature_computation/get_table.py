"""@author: dshah"""

import itertools
import math
from collections import defaultdict

import fitz


def truncate(x):
    return math.floor(x)


def debugPDFout(bboxList, page):
    outpdf = fitz.open()
    outpage = outpdf.new_page(width=page.rect.width, height=page.rect.height)
    shape = outpage.new_shape()
    for bbox in bboxList:
        x0 = bbox[0]
        y0 = bbox[1]
        x1 = bbox[2]
        y1 = bbox[3]
        top_left = fitz.Point(x0, y0)
        bottom_right = fitz.Point(x1, y1)

        shape.draw_rect(fitz.Rect(top_left, bottom_right))
        shape.finish()
        shape.commit()
    # all paths processed - commit the shape to its page

    outpdf.save("debug-page-01.pdf")


def debugPDFout2(rectList, page):
    outpdf = fitz.open()
    outpage = outpdf.new_page(width=page.rect.width, height=page.rect.height)
    shape = outpage.new_shape()
    for r in rectList:
        shape.draw_rect(r)
        shape.finish(
            fill=[1.0, 0.0, 0.0],  # fill color
            color=[1.0, 1.0, 1.0],  # line color
            even_odd=False,  # control color of overlaps
            width=1.5,  # line width
            stroke_opacity=1.0,  # same value for both
            fill_opacity=0.25,  # opacity parameters
        )
    shape.commit()
    # all paths processed - commit the shape to its page

    outpdf.save("debug-rects--page-" + str(page.number) + ".pdf")


# New Approach
from itertools import product


def mergeIntervals(arr):
    # Sorting based on the increasing order
    # of the start intervals
    arr.sort(key=lambda x: x[0])

    # Stores index of last element
    # in output array (modified arr[])
    index = 0

    # Traverse all input Intervals starting from
    # second interval
    for i in range(1, len(arr)):
        # If this is not first Interval and overlaps
        # with the previous one, Merge previous and
        # current Intervals
        if arr[index][1] >= arr[i][0]:
            arr[index][1] = max(arr[index][1], arr[i][1])
        else:
            index = index + 1
            arr[index] = arr[i]

    #    print("The Merged Intervals are :", end=" ")
    #    for i in range(index+1):
    #        print(arr[i], end=" ")

    return arr[: index + 1]


def removeDuplicates(lst):
    return list(set([i for i in lst]))


def removeSmallLines(lines):
    proper_lines = []
    for l in lines:
        x0 = l[0]
        y0 = l[1]
        x1 = l[2]
        y1 = l[3]

        deltaX = abs(x1 - x0)
        deltaY = abs(y1 - y0)

        length = deltaX * deltaX + deltaY * deltaY
        if length >= 10:
            proper_lines.append(l)

    return proper_lines


def get_table_location(page: fitz.Page, debug_table: bool = False) -> List[fitz.Rect]:
    """
    Get the location of tables in page
    by finding horizontal lines with same length

    Parameters
    ----------
    page: page object of pdf

    Returns
    -------
    table_rects: rectangles that contain tables
    """

    # make a list of horizontal lines
    # each line is represented by y and length
    hor_lines = []
    ver_lines = []
    hor_lines2 = []
    ver_lines2 = []
    paths = page.get_drawings()
    for p in paths:
        for item in p["items"]:
            if item[0] == "l":  # this is a line item
                p1 = item[1]  # start point
                p2 = item[2]  # stop point

                if round(abs(p1.y - p2.y)) <= 0.25:  # line horizontal?
                    hor_lines.append(
                        (round(p1.y + 0.5), round(p2.x - p1.x + 0.5))
                    )  # potential table delimiter

                    hor_lines2.append(
                        (
                            round(p1.x + 0.5),
                            round(p1.y + 0.5),
                            round(p2.x + 0.5),
                            round(p2.y + 0.5),
                        )
                    )
                if round(abs(p1.x - p2.x)) <= 0.25:
                    ver_lines.append(
                        (
                            round(p1.x + 0.5),
                            round(p1.y + 0.5),
                            round(p2.x + 0.5),
                            round(p2.y + 0.5),
                        )
                    )
                ver_lines2.append(
                    (
                        round(p1.x + 0.5),
                        round(p1.y + 0.5),
                        round(p2.x + 0.5),
                        round(p2.y + 0.5),
                    )
                )
            if item[0] == "re":  # this is a rectangle item
                r = item[1]

    hor_lines_dict = defaultdict(list)
    for li in hor_lines2:
        hor_lines_dict[li[1]].append([li[0], li[2]])

    # find whether table exists by number of lines with same length > 3
    table_rects = []
    # sort the list for ensuring the correct group by same keys
    hor_lines.sort(key=lambda x: x[1])

    # getting the top-left point and bottom-right point of table
    for k, g in itertools.groupby(hor_lines, key=lambda x: x[1]):
        g = list(g)
        if len(g) >= 3:  # number of lines of table will always >= 3
            g.sort(key=lambda x: x[0])  # sort by y value
            top_left = fitz.Point(0, g[0][0])
            bottom_right = fitz.Point(page.rect.width, g[-1][0])
            table_rects.append(fitz.Rect(top_left, bottom_right))

    if len(table_rects) >= 1:
        # create tempList
        ver_lines.sort(key=lambda x: x[1])
        temp_v_list = []
        for t in ver_lines:
            if round(t[3]) > round(t[1]):
                temp_v_list.append([round(t[1]), round(t[3])])
            else:
                temp_v_list.append([round(t[3]), round(t[1])])

        if len(temp_v_list) == 0:
            return table_rects

        tables_vertical_extents = mergeIntervals(temp_v_list)

        hor_table_extents = defaultdict()
        for k, v in hor_lines_dict.items():
            hor_table_extents[k] = mergeIntervals(v)

        if tables_vertical_extents != None:
            new_table_rects = []
            for r in tables_vertical_extents:
                try:
                    min_y = r[0]
                    max_y = r[1]
                    top_left = fitz.Point(hor_table_extents[min_y][0][0], min_y)
                    bottom_right = fitz.Point(hor_table_extents[max_y][0][1], max_y)
                    new_table_rects.append(fitz.Rect(top_left, bottom_right))
                except KeyError:
                    continue
            if new_table_rects != None:
                return new_table_rects
            else:
                return table_rects
    return table_rects


def get_table_locationNew(page: fitz.Page, debugTable=False):
    """
    Get the location of tables in page
    by finding horizontal lines with same length

    Parameters
    ----------
    page: page object of pdf

    Returns
    -------
    table_rects: rectangles that contain tables
    """

    # make a list of horizontal lines
    # each line is represented by y and length
    hor_lines = []
    ver_lines = []
    hor_lines2 = []
    #    ver_lines2 = []
    #    min_x = 10000000
    #    max_x = -1
    paths = page.get_drawings()
    # print(paths)
    for p in paths:
        for item in p["items"]:
            if item[0] == "l":  # this is a line item
                p1 = item[1]  # start point
                p2 = item[2]  # stop point

                #                if debugTable:
                #                    print(p1, p2)
                if round(abs(p1.y - p2.y)) <= 0.25:  # line horizontal?
                    hor_lines.append(
                        (
                            round(p1.y + 0.5),
                            round(p2.x - p1.x + 0.5),
                            round(p1.x),
                            round(p2.x),
                        )
                    )  # potential table delimiter
                    #                    minx = min(p1.x, p2.x)
                    #                    if minx < min_x:
                    #                        min_x = minx
                    #                    maxx = max(p1.x, p2.x)
                    #                    if maxx > max_x:
                    #                        max_x = maxx

                    # hor_lines2.append((round(p1.x+0.5), round(p1.y+0.5), round(p2.x+0.5), round(p2.y+0.5)))
                    hor_lines2.append(
                        (
                            truncate(p1.x),
                            truncate(p1.y),
                            truncate(p2.x),
                            truncate(p2.y + 0.5),
                        )
                    )
                if round(abs(p1.x - p2.x)) <= 0.25:
                    # ver_lines.append((round(p1.x+0.5), round(p1.y+0.5), round(p2.x+0.5), round(p2.y+0.5)))
                    ver_lines.append(
                        (
                            truncate(p1.x + 0.5),
                            truncate(p1.y + 0.5),
                            truncate(p2.x + 0.5),
                            truncate(p2.y + 0.5),
                        )
                    )
            #                ver_lines2.append((round(p1.x+0.5), round(p1.y+0.5), round(p2.x+0.5), round(p2.y+0.5)))
            if item[0] == "re":  # this is a rectangle item
                r = item[1]
                # if r.height <= 1 and r.width <= 1:
                #    continue
                # ir = item[1].irect
                p1 = fitz.Point(r[0], r[1])
                p2 = fitz.Point(r[2], r[3])

                # if r.height <= 1:
                hor_lines.append(
                    (
                        round(p1.y + 0.5),
                        round(p2.x - p1.x + 0.5),
                        round(p1.x),
                        round(p2.x),
                    )
                )  # potential table delimiter
                # hor_lines2.append((truncate(p1.x), truncate(p1.y), truncate(p2.x), truncate(p2.y)))
                if r.height >= 1:
                    hor_lines2.append(
                        (truncate(p1.x), truncate(p1.y), truncate(p2.x), truncate(p1.y))
                    )
                    hor_lines2.append(
                        (truncate(p1.x), truncate(p2.y), truncate(p2.x), truncate(p2.y))
                    )
                else:
                    hor_lines2.append(
                        (
                            truncate(p1.x),
                            truncate((p1.y + p2.y) * 0.5),
                            truncate(p2.x),
                            truncate((p1.y + p2.y) * 0.5),
                        )
                    )

                if r.width >= 1:
                    ver_lines.append(
                        (truncate(p1.x), truncate(p1.y), truncate(p1.x), truncate(p2.y))
                    )
                    ver_lines.append(
                        (truncate(p2.x), truncate(p1.y), truncate(p2.x), truncate(p2.y))
                    )
                else:
                    ver_lines.append(
                        (
                            truncate((p1.x + p2.x) * 0.5),
                            truncate(p1.y),
                            truncate((p1.x + p2.x) * 0.5),
                            truncate(p2.y),
                        )
                    )

    hor_lines2 = removeDuplicates(hor_lines2)
    ver_lines = removeDuplicates(ver_lines)

    blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)["blocks"]
    max_lineheight = 0
    for b in blocks:
        for l in b["lines"]:
            bbox = fitz.Rect(l["bbox"])
            if bbox.height > max_lineheight:
                max_lineheight = bbox.height
    # we now have the max lineheight on this page
    hor_lines2_updated = []
    for li in hor_lines2:
        p1x = li[0]
        p1y = li[1]
        p2x = li[2]
        p2y = li[3]

        rect = fitz.Rect(
            p1x - 4, p1y - max_lineheight - 4, p2x + 4, p2y + 4
        )  # the rectangle "above" a drawn line
        text = page.get_textbox(rect)
        if len(text.strip()) == 0:
            hor_lines2_updated.append(li)
        # print(f"Underlined: '{text}'.")

    if len(hor_lines2_updated) > 0:
        hor_lines2 = hor_lines2_updated

    hor_lines2 = removeSmallLines(hor_lines2)
    ver_lines = removeSmallLines(ver_lines)

    if len(hor_lines2) <= 1 and len(ver_lines) <= 1:
        return []

    hor_lines_dict = defaultdict(list)
    for li in hor_lines2:
        if li[0] < li[2]:
            hor_lines_dict[li[1]].append([li[0], li[2]])
        else:
            hor_lines_dict[li[1]].append([li[2], li[0]])

    ver_lines_dict = defaultdict(list)
    for li in ver_lines:
        if li[1] < li[3]:
            ver_lines_dict[li[0]].append([li[1], li[3]])
        else:
            ver_lines_dict[li[0]].append([li[3], li[1]])
    # new_hor_lines = []
    # for k,v in hor_lines_dict.items():
    #    new_hor_lines.append((k,round(v)))
    # find whether table exists by number of lines with same length > 3
    table_rects = []
    new_table_rects = []
    # sort the list for ensuring the correct group by same keys
    hor_lines.sort(key=lambda x: x[1])

    # getting the top-left point and bottom-right point of table
    for k, g in itertools.groupby(hor_lines, key=lambda x: x[1]):
        g = list(g)
        if len(g) >= 3:  # number of lines of table will always >= 3
            g.sort(key=lambda x: x[0])  # sort by y value
            # top_left = fitz.Point(0, g[0][0])
            top_left = fitz.Point(g[0][2], g[0][0])
            # bottom_right = fitz.Point(page.rect.width, g[-1][0])
            bottom_right = fitz.Point(g[-1][3], g[-1][0])
            table_rects.append(fitz.Rect(top_left, bottom_right))

    if len(table_rects) >= 1 or len(ver_lines_dict) > 0:
        # create tempList
        #        ver_lines.sort(key=lambda x: x[1])
        #        tempVList = []
        #        for t in ver_lines:
        #            if round(t[3]) > round(t[1]):
        #                tempVList.append([round(t[1]), round(t[3])])
        #            else:
        #                tempVList.append([round(t[3]), round(t[1])])

        #        if len(tempVList) == 0:
        #            return table_rects

        #        tablesVerticalExtents = mergeIntervals(tempVList)

        horTableExtents = defaultdict()
        for k, v in hor_lines_dict.items():
            s = mergeIntervals(v)
            if len(s) > 0:
                horTableExtents[k] = s

        sorted(horTableExtents.keys())
        verTableExtents = defaultdict()
        for k, v in ver_lines_dict.items():
            s = mergeIntervals(v)
            if len(s) > 0:
                verTableExtents[k] = s

        sorted(verTableExtents.keys())

        def getIndex(l: list, x: int, e=-1) -> int:
            try:
                idx = l.index(x)
            except ValueError:
                idx = e
            return idx

        def closest(lst, K):
            return lst[min(range(len(lst)), key=lambda i: abs(lst[i] - K))]

        def getClosestValue(l, val):
            return min(range(len(l)), key=lambda i: abs(l[i] - val))

        def getClosestIndex(l, val):
            z = sorted(enumerate(l), key=lambda x: abs(x[1] - val))
            return min(enumerate(l), key=lambda x: abs(x[1] - val))

        def getBoundingBox(line):
            min_x, min_y = float("inf"), float("inf")
            max_x, max_y = float("-inf"), float("-inf")

            for i, l in enumerate(line):
                if i % 2 == 0:
                    if line[i] < min_x:
                        min_x = line[i]
                    if line[i] > max_x:
                        max_x = line[i]
                else:
                    if line[i] < min_y:
                        min_y = line[i]
                    if line[i] > max_y:
                        max_y = line[i]

            return (min_x, max_x, min_y, max_y)

        # sort the horizontal lines by y-coordinate
        hor_lines2.sort(key=lambda x: x[1])
        # sort the vertical lines by x-coordinate
        ver_lines.sort(key=lambda x: x[0])

        boxList = []

        def add2BBoxList(bbox):
            found = True
            if len(boxList) == 0:
                boxList.append(bbox)
            for idx, box in enumerate(boxList):
                x0 = box[0]
                y0 = box[1]
                x1 = box[2]
                y1 = box[3]
                boxW = abs(x1 - x0)
                boxH = abs(y1 - y0)
                bboxW = abs(bbox[0] - bbox[2])
                bboxH = abs(bbox[3] - bbox[1])

                if abs(bbox[0] - x0) <= 2 and abs(bbox[1] - y0) <= 2:
                    found = False
                    if (boxW - bboxW) <= 2 or (boxH - bboxH) <= 2:
                        boxList[idx] = bbox
                        break
                    else:
                        break
            if found:
                boxList.append(bbox)

        def getTouchingVertLine(hline):
            hor_x0 = hline[0]
            hor_y0 = hline[1]
            hor_x1 = hline[2]
            hor_y1 = hline[3]  # should be same as hor_y0

            for v_idx, vline in enumerate(ver_lines):
                ver_x0 = vline[0]
                ver_y0 = vline[1]
                vline[2]
                ver_y1 = vline[3]

                if abs(ver_x0 - hor_x0) <= 2 or abs(ver_x0 - hor_x1) <= 2:
                    if abs(ver_y0 - hor_y0) <= 2 or abs(ver_y0 - hor_y1) <= 2:
                        return v_idx, vline
                    elif abs(ver_y1 - hor_y0) <= 2 or abs(ver_y1 - hor_y1) <= 2:
                        return v_idx, vline

            return -1, None

        #        for y in yLevels:
        #            min_x_list = horTableExtents.get(y)
        #            if min_x_list != None:
        #                print(y)
        #                for _, xv in enumerate(min_x_list):
        #                    min_x = xv[0]
        #                    max_x = xv[1]
        #                    min_y = getClosestIndex(xLevels, min_x)[1]
        #                    max_y = getClosestIndex(xLevels, max_x)[1]
        #                    bbox = (min_x, min_y, max_x, max_y)
        #                    add2BBoxList(bbox)
        #                    print(_, min_x, min_y, max_x, max_y)

        hlines_consumed = []
        vlines_consumed = []
        i1 = 0
        i2 = 0
        for h_idx, hline in enumerate(hor_lines2):
            # print(h_idx)
            min_x = hline[0]
            max_x = hline[2]
            min_y = hline[1]
            max_y = hline[3]
            # bbox = (hmin_x, hmin_y, hmax_x, hmax_y)
            # bbox = getBoundingBox(hline)
            # add2BBoxList(bbox)

            # get touching Vertical line
            v_idx, vline = getTouchingVertLine(hline)
            if vline == None:
                i2 += 1
                continue
            vline[0]
            vline[2]
            vmin_y = vline[1]
            vmax_y = vline[3]

            # if vmin_x < min_x:
            #    min_x = vmin_x
            if vmin_y < min_y:
                min_y = vmin_y
            # if vmax_x > max_x:
            #    max_x = vmax_x
            if vmax_y > max_y:
                max_y = vmax_y

            bbox = (min_x, min_y, max_x, max_y)
            add2BBoxList(bbox)
            hlines_consumed.append(h_idx)
            vlines_consumed.append(v_idx)
            i1 += 1
            # print(_, min_x, min_y, max_x, max_y)

        boxList2 = []
        for bbox in boxList:
            x1 = bbox[0]
            y1 = bbox[1]
            x2 = bbox[2]
            y2 = bbox[3]

            if abs(x2 - x1) < 1 or abs(y2 - y1) < 1:
                # boxList2.append(bbox)
                continue
            else:
                boxList2.append(bbox)

        close_dist = 2

        # use product, more concise
        def should_merge(box1, box2):
            a = (box1[0], box1[1]), (box1[2], box1[3])
            b = (box2[0], box2[1]), (box2[2], box2[3])

            # print('box1:')
            # print(box1)
            # print('box2:')
            # print(box2)
            if any(
                abs(a_v - b_v) <= close_dist
                for i in range(2)
                for a_v, b_v in product(a[i], b[i])
            ):
                return True, [
                    min(*a[0], *b[0]),
                    min(*a[1], *b[1]),
                    max(*a[0], *b[0]),
                    max(*a[1], *b[1]),
                ]

            return False, None

        # Computer a Matrix similarity of distances of the text and object
        def calc_dist(box1, box2):
            # text: ymin, xmin, ymax, xmax
            # obj: ymin, xmin, ymax, xmax
            bbox[0]
            bbox[1]
            bbox[2]
            bbox[3]

            box1_xmin = box1[0]
            box1_ymin = box1[1]
            box1_xmax = box1[2]
            box1_ymax = box1[3]

            box2_xmin = box2[0]
            box2_ymin = box2[1]
            box2_xmax = box2[2]
            box2_ymax = box2[3]

            x_dist = min(
                abs(box1_xmin - box2_xmin),
                abs(box1_xmin - box2_xmax),
                abs(box1_xmax - box2_xmin),
                abs(box1_xmax - box2_xmax),
            )
            y_dist = min(
                abs(box1_ymin - box2_ymin),
                abs(box1_ymin - box2_ymax),
                abs(box1_ymax - box2_ymin),
                abs(box1_ymax - box2_ymax),
            )

            # print('x_dist: ' + str(x_dist))
            # print('y_dist: ' + str(y_dist))
            dist = x_dist + y_dist
            #
            # std::max(rectA.left, rectB.left) < std::min(rectA.right, rectB.right) && std::max(rectA.top, rectB.top) < std::min(rectA.bottom, rectB.bottom)
            #
            checkHorz = max(box1_xmin, box2_xmin) - min(box1_xmax, box2_xmax)
            checkVert = max(box1_ymin, box2_ymin) - min(box1_ymax, box2_ymax)
            if abs(checkHorz) <= 1 and abs(checkVert) <= 1:
                dist = 0
            return dist

        def should_merge2(box1, box2):
            # print('box1: '+ str(box1) )
            # print('box2: '+ str(box2))
            dist = calc_dist(box1, box2)
            # print('dist:' +str(dist))

            if dist <= close_dist:
                return True, [
                    min(box1[0], box2[0]),
                    min(box1[1], box2[1]),
                    max(box1[2], box2[2]),
                    max(box1[3], box2[3]),
                ]

            return False, None

        # Merge BBoxes
        for i, box1 in enumerate(boxList2):
            for j, box2 in enumerate(boxList2):
                if i == j or not box1 or not box2:
                    continue
                is_merge, new_box = should_merge2(box1, box2)

                if is_merge:
                    # print('merged bbox: ' + str(new_box))
                    boxList2[i] = None
                    boxList2[j] = new_box
                    break

        boxList2 = [b for b in boxList2 if b]
        # print(boxes)
        # sort based on area
        boxList2.sort(key=lambda x: abs(x[2] - x[0]) * abs(x[3] - x[1]), reverse=True)
        # Merge BBoxes

        def pointInRect(bbox, x, y):
            x1 = bbox[0]
            y1 = bbox[1]
            x2 = bbox[2]
            y2 = bbox[3]

            #            if (x1 - x) <=2  and (x - x2) <= 2:
            #                if (y1 - y) <=2  and (y - y2 <=2):
            #                    return True
            #            return False
            if (x - x1) >= 0 and (x2 - x) >= 0:
                if (y - y1) >= 0 and (y2 - y) >= 0:
                    return True
            return False

        def ifInside(box1, box2):
            box1[0]
            box1[1]
            box1[2]
            box1[3]

            if pointInRect(box2, box1[0], box1[1]) and pointInRect(
                box2, box1[2], box1[3]
            ):
                return True, [box2[0], box2[1], box2[2], box2[3]]
            # elif (pointInRect(box2, box1[0], box1[0]) and pointInRect(box2, box1[0], box1[0])):
            #    return True, [box1[0], box1[1], box1[2], box1[3]]
            else:
                return False, None

        for i, box1 in enumerate(boxList2):
            for j, box2 in enumerate(boxList2):
                if i == j or not box1 or not box2:
                    continue
                is_merge, new_box = ifInside(box1, box2)

                if is_merge:
                    # print('merged bbox: ' + str(new_box))
                    boxList2[i] = None
                    # boxList2[j] = new_box
                    break

        boxList2 = [b for b in boxList2 if b]
        if debugTable:
            debugPDFout(boxList2, page)

        hlines_unused = [
            x for i, x in enumerate(hor_lines2) if not i in hlines_consumed
        ]
        vlines_unused = [x for i, x in enumerate(ver_lines) if not i in vlines_consumed]

        nHorzLines = len(hlines_unused)
        nVertLines = len(vlines_unused)

        i = 0
        while i < nHorzLines:
            x1 = hlines_unused[i][0]
            x2 = hlines_unused[i][2]
            y1 = hlines_unused[i][1]
            y2 = hlines_unused[i][3]

            foundBbox = False
            for bbox in boxList2:
                if pointInRect(bbox, x1, y1) and pointInRect(bbox, x2, y2):
                    foundBbox = True
                    break

            if foundBbox:
                hlines_unused.pop(i)
                nHorzLines = nHorzLines - 1
            else:
                i += 1

        i = 0
        while i < nVertLines:
            x1 = vlines_unused[i][0]
            x2 = vlines_unused[i][2]
            y1 = vlines_unused[i][1]
            y2 = vlines_unused[i][3]

            foundBbox = False
            for bbox in boxList2:
                if pointInRect(bbox, x1, y1) and pointInRect(bbox, x2, y2):
                    foundBbox = True
                    break

            if foundBbox:
                vlines_unused.pop(i)
                nVertLines = nVertLines - 1
            else:
                i += 1

        hlines_left = []
        for i, hline in enumerate(hlines_unused):
            x1 = hline[0]
            x2 = hline[2]
            y1 = hline[1]
            y2 = hline[3]
            hlines_left.append((y1, x2 - x1, i))

        hlines_left.sort(key=lambda x: x[1])

        # getting the top-left point and bottom-right point of table
        for k, g in itertools.groupby(hlines_left, key=lambda x: x[1]):
            g = list(g)
            if len(g) >= 3:  # number of lines of table will always >= 3
                g.sort(key=lambda x: x[0])  # sort by y value
                top_left = fitz.Point(hlines_unused[g[0][2]][0], g[0][0])
                bottom_right = fitz.Point(hlines_unused[g[-1][2]][2], g[-1][0])
                new_table_rects.append(fitz.Rect(top_left, bottom_right))

        #        if len(new_table_rects) == 0 and len(table_rects) > 0:
        #            new_table_rects.extend(table_rects)

        new_table_rects2 = new_table_rects
        new_table_rects2.extend(table_rects)

        for i, r1 in enumerate(new_table_rects2):
            for j, r2 in enumerate(new_table_rects2):
                if i == j or not r1 or not r2:
                    continue

                # box1[0], box1[1], box1[2], box1[3] = r1.x0, r1.y0, r1.x1, r1.y1
                box1 = (r1.x0, r1.y0, r1.x1, r1.y1)
                # box2[0], box2[1], box2[2], box2[3] = r2.x0, r2.y0, r2.x1, r2.y1
                box2 = (r2.x0, r2.y0, r2.x1, r2.y1)

                is_merge, new_box = ifInside(box1, box2)

                if is_merge:
                    # print('merged rect: ' + str(new_box))
                    new_table_rects2[i] = None
                    break

        new_table_rects = [r for r in new_table_rects2 if r]

        if len(boxList2) > 0:
            for bbox in boxList:
                x1 = bbox[0]
                y1 = bbox[1]
                x2 = bbox[2]
                y2 = bbox[3]

                if (
                    abs(x2 - x1) <= 5 or abs(y2 - y1) <= 5
                ):  # too thin rectangle -> unlikely to contain text!
                    continue

                top_left = fitz.Point(bbox[0], bbox[1])
                bottom_right = fitz.Point(bbox[2], bbox[3])
                new_table_rects.append(fitz.Rect(top_left, bottom_right))

    if new_table_rects != None:
        if debugTable and len(new_table_rects) > 0:
            debugPDFout2(new_table_rects, page)
        return new_table_rects
    else:
        return table_rects

    return table_rects


def is_table_content(
    table_coordinate: List[Tuple[float, float, float, float]],
    bx0: float,
    bx1: float,
    by0: float,
    by1: float,
) -> bool:
    # check if a line coordinate is inside a table coordinate
    bx0, bx1, by0, by1 = map(float, (bx0, bx1, by0, by1))
    for table_coor in table_coordinate:
        tx0, ty0, tx1, ty1 = map(float, table_coor)
        if (tx0 < bx0 < tx1 or tx0 < bx1 < tx1) and (
            ty0 < by0 < ty1 or ty0 < by1 < ty1
        ):
            return True
    return False
