import numpy as np
import cv2
from services.detection import detection
from services.svm import SVM
from services.utils import imwrite

# if you want print some log when your program is running,
# just append a string to this variable
log = []
color_dist = {
            'Red': {'Lower': np.array([175,50,20]), 'Upper': np.array([180, 255, 255])},
            'Yellow': {'Lower': np.array([23,41,133]), 'Upper': np.array([40,255,255])},
            'Green': {'Lower': np.array([35, 43, 35]), 'Upper': np.array([90, 255, 255])},
              }

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

def straight_fn(state, initial_speed):
    if state.get()['current_speed'] < 0.9:
        initial_speed += 0.1
        set_state(state, 'current_speed', initial_speed)
    left_speed = right_speed = state.get()['current_speed']
    log.append("going straight, l:%.2f, r:%.2f" % (left_speed, right_speed))
    return left_speed, right_speed

def follow_lane(view, mask, state):
    h, w= mask.shape #h=120, w=160
    M = cv2.moments(mask)
    cx = int(M['m10']/M['m00'])
    cx = int((108) / 2 + cx)
    cy = int(M['m01']/M['m00'])
    cv2.circle(view, (cx, cy), 4, (0,0,255), -1)
    imwrite(str(state.get()['state']) + '-dot.jpg', view)

    err = cx - w/2
    log.append("err: %d" % err)

    if abs(err) < 20: #20
        kp = 0.02
    else:
        kp = 0.01

    if state.get()['current_speed'] < 0.5:
        set_state(state, 'current_speed', 0.5)

    if err == 0:
        return straight_fn(state, state.get()['current_speed'])
    elif err > 0:
        # turn right
        left_speed = 1*state.get()['current_speed']
        lr_ratio=(50-abs(err))*kp
        return turn_right_fn(state, left_speed, lr_ratio) #right_speed=0.01
    elif err < 0:
        # turn left
        right_speed = 1*state.get()['current_speed']
        lr_ratio=(50-abs(err))*kp
        return turn_left_fn(state, right_speed, lr_ratio) #left_speed=0.01

def detect_yellow_line(view, state):

    hsv = cv2.cvtColor(view, cv2.COLOR_BGR2HSV)
    mask_yellow = cv2.inRange(hsv, color_dist['Yellow']['Lower'], color_dist['Yellow']['Upper']) #binarize yellow color

    mask = mask_yellow
    h, w, d = view.shape #h=120, w=160, d=3

    mask[0:40, 0:w] = 0
    mask[60:, 20:w] = 0

    # imwrite(str(state.get()['state']) + '-mask_white.jpg', mask_white)
    if state.get()['id_num'] == 35:
        mask[:, 128:w] = 0              # hide right yellow line

    elif state.get()['id_num'] == 34:
        mask[:, 128:w] = 0              # hide right yellow line

    M = cv2.moments(mask)
    if M['m00'] > 0:
        return True, mask
    else:
        return False, None ##if yellow path end

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
        color_id: 0=red, 1=yellow, 2=green, 3=has_check_light

    Returns:
        (left, right): your car wheels' speed
    """
    state_dict = {'state':1, 'id_num':None, 'current_speed':0.5}


    if state.get() is None:
        state.set(state_dict)
    else:
        state.get()['state'] += 1
        state.set(state.get())

    state_str = str(state.get()['state'])
    id_num = state.get()['id_num']

    imwrite(state_str + '-1.jpg', view1)
    imwrite(state_str + '-2.jpg', view2)

    if view1 is not None:
      has_yellow, mask = detect_yellow_line(view1, state)

    if view2 is not None:
        log.append("id:" + str(id_num))
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
            roi = im[ymin:ymax, xmin:xmax, :]
            imwrite(state_str + '-roi.jpg', roi)
            if xmax < 600:
                roi = im[ymin:ymax, xmin:xmax, :]
                id_num = svm.predict(roi, "hog")
                set_state(state, 'id_num', id_num)
                sign_flag = 1

    if not has_yellow:
        if id_num == 14:
            left_speed, right_speed = stop_fn(state)

        elif id_num == 33:
            left_speed, right_speed = turn_right_fn(state, 0.6, 0.4) #0.6, 0.2

        elif id_num == 34:
            left_speed, right_speed = turn_left_fn(state, 0.6, 0.5) #0.6, 0.3

        elif id_num == 35:
            left_speed, right_speed = straight_fn(state, 1.0)

    else:
        # follow line
        left_speed, right_speed = follow_lane(view1, mask, state)

    set_state(state, 'current_speed', max(left_speed, right_speed))
    return left_speed, right_speed
