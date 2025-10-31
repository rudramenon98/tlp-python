import traceback

import pandas as pd
from more_itertools import split_when


def get_y_center(b):
    up, down = b[0], b[2]
    return (up + down) / 2


def not_vertically_overlapping(b1, b2):
    up1, down1 = b1[0], b1[2]
    up2, down2 = b2[0], b2[2]
    return down1 < up2 or (down1 - up2) < (up2 - up1)


def groupbyrow(boxes):
    sorted_boxes = sorted(boxes, key=get_y_center)
    return list(split_when(sorted_boxes, not_vertically_overlapping))


def get_x_center(b):
    left, right = b[0], b[2]
    return (left + right) / 2


def not_horz_overlapping(b1, b2):
    left1, right1 = b1[0], b1[2]
    left2, right2 = b2[0], b2[2]
    return right1 < left2 or (right1 - left2) < (left2 - left1)


def groupbycolumn(boxes):
    sorted_boxes = sorted(boxes, key=get_x_center)
    return list(split_when(sorted_boxes, not_horz_overlapping))


def order_blocks_impl(boxes):

    rows = groupbycolumn(boxes)

    idx = 0

    new_block_ids = dict()
    lone_wolf_blocks = []

    for i, row in enumerate(rows):
        # row2 = sorted(row, key=get_y_center)
        if len(row) == 1:
            lone_wolf_blocks.append(row[0])
            idx += 1
            curr_block_id = row[0][-1]
            new_block_ids[curr_block_id] = idx
            continue
        row2 = sorted(row, key=lambda x: -x[1])
        # row2 = row
        for box in reversed(row2):
            idx += 1
            bbox = []
            # bbox.extend(rect)
            bbox.append(box[0])
            bbox.append(box[1])
            bbox.append(box[2])
            bbox.append(box[3])
            # label = str(box[-1])+'--'+str(idx)
            # draw_labelled_box(bbox, label, colours[i])
            curr_block_id = box[-1]
            new_block_ids[curr_block_id] = idx

    return new_block_ids, lone_wolf_blocks


def estimate_number_of_columns(boxes, width):

    if len(boxes) == 0:
        return 1

    column_bounry_h = width / 2

    n_left_column = 0
    n_right_column = 0
    n_crossing = 0
    n_others = 0
    total_text_len = 0
    for box in boxes:
        box_text_len = box[-2]
        total_text_len += box_text_len

        if box[0] < column_bounry_h <= box[2]:
            n_crossing += box_text_len
        elif box[2] < column_bounry_h:
            n_left_column += box_text_len
        elif box[0] >= column_bounry_h:
            n_right_column += box_text_len
        else:
            n_others += box_text_len

    # crossing_ratio = n_crossing / len(boxes)
    if total_text_len > 0:
        crossing_ratio = n_crossing / total_text_len

        if crossing_ratio > 0.80:
            return 1
        else:
            return 2
    else:
        return 1


def order_blocks(df):

    groups = df.groupby('PageNumber')  # group page-wise

    list(df.columns)

    df2 = pd.DataFrame()

    for group in groups:  # page-wise iteration

        boxes = []
        block_ids_set = set()
        # get the block data for each page
        each_file_json = group[1]

        # sort from top to bottom in the page
        # each_file_json.sort_values(by = 'line_y0', inplace = True)
        for i in range(each_file_json.shape[0]):

            currRow = each_file_json.iloc[i]

            #
            if pd.notna(currRow['ExtractedClass']):
                #                each_file_json.iloc[i]['new_block_order'] = -1
                continue

            # get the block extents
            bx0 = currRow['block_x0']
            bx1 = currRow['block_x1']
            by0 = currRow['block_y0']
            by1 = currRow['block_y1']

            # get the intial block id
            block_id = currRow['ID']

            if block_id not in block_ids_set:
                block_ids_set.add(block_id)
            else:
                continue

            text_len = len(str(each_file_json.iloc[i]['text']))
            boxes.append([bx0, by0, bx1, by1, text_len, block_id])

        ncols = estimate_number_of_columns(boxes, each_file_json.iloc[0]['page_width'])
        each_file_json['ncols'] = ncols

        if ncols != 2:
            each_file_json['new_block_order'] = each_file_json['ID']
            df2 = pd.concat([df2, each_file_json])
            continue

        # order the blocks
        new_block_ids, lone_wolf_blocks = order_blocks_impl(boxes)

        try:

            def get_new_block_id(x, page_no):
                if page_no != x['PageNumber']:
                    # print('wrong page')
                    return -1
                else:
                    pass
                    # print('correct page')
                y = new_block_ids.get(x['ID'], -1)
                if y != -1:
                    return y
                else:
                    # print('Key not found =  ' + str(x['Block_ID']))
                    return x['ID']

            # df['new_block_order'] = df.apply(get_new_block_id, page_no = each_file_json.iloc[0]['Page_No'], axis=1)
            each_file_json['new_block_order'] = each_file_json.apply(
                get_new_block_id, page_no=each_file_json.iloc[0]['PageNumber'], axis=1
            )
            # df2.append(each_file_json)
            df2 = pd.concat([df2, each_file_json])
        except Exception:
            traceback.print_exc()

    df = df2
    return df
