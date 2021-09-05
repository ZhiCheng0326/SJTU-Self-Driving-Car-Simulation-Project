import numpy as np
import cv2
from services.detection import detection
from services.svm import SVM
from services.utils import imwrite

# if you want print some log when your program is running,
# just append a string to this variable
log = []
def stop_fn(state, left_speed=0, right_speed=0):
    log.append("stopping, l:%.2f, r:%.2f" % (left_speed, right_speed))
    return left_speed, right_speed

def turn_right_fn(state, left_speed, lr_ratio = 0.2):
    right_speed = lr_ratio * left_speed
    log.append("turning right, l:%.2f, r:%.2f" % (left_speed, right_speed))
    return left_speed, right_speed

def turn_left_fn(state, right_speed, lr_ratio = 0.2):
    left_speed = lr_ratio * right_speed
    log.append("turning left, l:%.2f, r:%.2f" % (left_speed, right_speed))
    return left_speed, right_speed

def straight_fn(state):
    left_speed = right_speed = -0.2
    log.append("going straight, l:%.2f, r:%.2f" % (left_speed, right_speed))
    return left_speed, right_speed

def get_lanes(state, lines):
    diff_neighbor = lines[:-1] - lines[1:]
    id = np.where(abs(diff_neighbor[:,0])>100)[0]
    if id.size == 0:
        #only one lane detected
        left_lane= np.average(lines, axis=0).astype(int)
        right_lane = None
        set_state(state, 'parking_state', 1)
    else:
        id = id[0]
        left_lane = np.average(lines[:id+1, :], axis=0).astype(int)
        right_lane = np.average(lines[id+2:, :], axis=0).astype(int)
        set_state(state, 'parking_state', 2)

    return left_lane, right_lane

def find_first_white_lane():
    #looking for parking
    left_speed = -0.5
    right_speed = -0.8
    log.append("finding carpark, l:%.2f, r:%.2f" % (left_speed, right_speed))
    return left_speed, right_speed

def find_second_white_lane(view2, state, left_lane):
    #find second white lane after first lane detected
    h, w, d = view2.shape #h=480, w=640, d=3

    cx = min(w, left_lane[2]+200)
    cy = h*0.75

    cx = int(cx)
    cy = int(cy)
    cv2.circle(view2, (cx, cy), 4, (0,0,255), -1)
    imwrite(str(state.get()['state']) + '-dot.jpg', view2)

    err = cx - w/2
    log.append("err: %d" % err)
    kp = 0.001 #0.01

    if err == 0:
        return straight_fn(state)
    elif err > 0:
        right_speed = -0.5
        lr_ratio=(400-abs(err))*kp
        return turn_left_fn(state, right_speed, lr_ratio) #right_speed=0.01
    elif err < 0:
        # turn left
        left_speed = -0.5
        lr_ratio=(400-abs(err))*kp
        return turn_right_fn(state, left_speed, lr_ratio) #left_speed=0.01

def align_with_yellow_line(view1, state):
    hsv = cv2.cvtColor(view1, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([23,41,133])
    upper_yellow = np.array([40,255,255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow) #binarize yellow color

    edges = cv2.Canny(mask_yellow, 80, 120)
    edges[80:,:] = 0 # avoid seeing the frt part of the car

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, maxLineGap=200)
    if lines is not None:
        lines = np.squeeze(lines, axis=1)
        slope = (lines[:,3]-lines[:,1])/ (lines[:,2]-lines[:,0])
        slope = np.average(slope)
        log.append("slope: "+ str(slope))
        # print(slope)

        if slope > 0.01:
            return turn_left_fn(state, -0.2, lr_ratio=0.5)

        elif slope < -0.01:
            return turn_right_fn(state, -0.2, lr_ratio=0.5)

        elif slope < 0.01 and slope > -0.01:
            return straight_fn(state)
    else:
        set_state(state, 'parking_state', 3)
        return stop_fn(state)

def reverse_till_no_yellow(view2, state):
    hsv = cv2.cvtColor(view2, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([23,41,133])
    upper_yellow = np.array([40,255,255])

    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow) #binarize yellow color
    mask_yellow[:340,:] = 0
    imwrite(str(state.get()['state']) + '-noyello.jpg', mask_yellow)
    cnts = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    if cnts:
        return straight_fn(state)
    else:
        return stop_fn(state)

def parking(view1, view2, state, left_lane=None, right_lane=None):
    log.append("parking_state: " + str(state.get()['parking_state']))

    if state.get()['parking_state'] == 0:
        left_speed, right_speed = find_first_white_lane()

    elif state.get()['parking_state'] == 1:
        left_speed, right_speed = find_second_white_lane(view2, state, left_lane)

    elif state.get()['parking_state'] == 2:
        left_speed, right_speed = align_with_yellow_line(view1, state)

    elif state.get()['parking_state'] == 3:
        left_speed, right_speed = reverse_till_no_yellow(view2, state)

    return left_speed, right_speed

def detect_white_line(view, state):
    """
    parking_state{
        0: No lane detected
        1: Left lane detected
        2: Left and right Lane detected
        3: Aligned with view1 yellow line
    }
    """
    hsv = cv2.cvtColor(view, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 210])
    upper_white = np.array([255, 30, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white) #binarize white color

    h, w, d = view.shape #h=480, w=640, d=3
    mask_white[0:350, 0:w] = 0 # show only bottom part #350

    if state.get()['parking_state'] == 0:
        mask_white[:, 580:w] = 0

    imwrite(str(state.get()['state']) + '-mask.jpg', mask_white)
    lines = cv2.HoughLinesP(mask_white, 1, np.pi/180, 60, maxLineGap=200)

    if lines is not None:
        lines = np.squeeze(lines, axis=1)
        lines = lines[lines[:,0].argsort()]
        slope = (lines[:,3]-lines[:,1])/ (lines[:,2]-lines[:,0])
        lines = lines[abs(slope)>0.2]

    # draw Hough lines
    left_lane = right_lane = None
    if lines is not None:
      left_lane, right_lane = get_lanes(state, lines)

      if left_lane is not None:
          cv2.line(view, (left_lane[0], left_lane[1]), (left_lane[2], left_lane[3]), (0, 255, 0), 2)
      if right_lane is not None:
          cv2.line(view, (right_lane[0], right_lane[1]), (right_lane[2], right_lane[3]), (0, 255, 0), 2)

    return left_lane, right_lane

def set_state(state, key, value):
    state.get()[key] = value
    state.set(state.get())

def image_to_speed(view1, view2, state):
    """This is the function where you should write your code to
    control your car.

    You need to calculate your car wheels' speed based on the views.

    Whenever you need to print something, use log.append().

    Args:
        view1 (ndarray): The left-bottom view,
                          it is grayscale, 1 * 120 * 160
        view2 (ndarray): The right-bottom view,
                          it is colorful, 3 * 120 * 160
        state: your car's state, initially None, you can
               use it by state.set(value) and state.get().
               It will persist during continuous calls of
               image_to_speed. It will not be reset to None
               once you have set it.

    Returns:
        (left, right): your car wheels' speed
    """
    state_dict = {'state':1, 'id_num':None, 'parking_state':0, 'current_speed':0.5}

    if state.get() is None:
        state.set(state_dict)
    else:
        state.get()['state'] += 1
        state.set(state.get())

    state_str = str(state.get()['state'])
    id_num = state.get()['id_num']

    log.append("#" + state_str)
    imwrite(state_str + '-1.jpg', view1)
    imwrite(state_str + '-2.jpg', view2)

    if view2 is not None:
        if id_num != 33:
            sign_classes = {
                14: 'Stop',
                33: 'Turn right',
                34: 'Turn left',
                35: 'Straight'
            }
            svm = SVM()
            detector = detection()
            im = view2
            rect = detector.ensemble(im)
            if rect:
                xmin, ymin, xmax, ymax = rect
                if xmax < 600:
                    roi = im[ymin:ymax, xmin:xmax, :]
                    id_num = svm.predict(roi, "hog")
                    sign_flag = 1
                    set_state(state, 'id_num', id_num)
            log.append("id:" + str(id_num))

    if id_num == 33:
        # sign found
        if state.get()['parking_state'] == 0 or state.get()['parking_state'] == 1:
            left_lane, right_lane = detect_white_line(view2, state)
            left_speed, right_speed = parking(view1, view2, state, left_lane, right_lane)
        else:
            left_speed, right_speed = parking(view1, view2, state)

    else:
        # sign not found
        left_speed = right_speed = 1.0

    return left_speed, right_speed
