
import cv2
import os
import copy
import numpy as np

from src import model
from src import util
from src.body import Body
from src.hand import Hand

# Set the model type and paths
model_type = 'body25'  # 'coco' also supported
if model_type == 'body25':
    model_path = './model/pose_iter_584000.caffemodel.pt'
else:
    model_path = './model/body_pose_model.pth'

body_estimation = Body(model_path, model_type)
hand_estimation = Hand('model/hand_pose_model.pth')

# Set the input and output folder paths
input_folder = 'F:\\VITON-HD-main\\VITON-HD-main2\\datasets\\test\\image'
output_folder = 'F:\\VITON-HD-main\\VITON-HD-main2\\datasets\\test\\openpose-img'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process each image in the input folder
for image_name in os.listdir(input_folder):
    # Construct the full path to the image
    test_image_path = os.path.join(input_folder, image_name)

    # Read the image
    oriImg = cv2.imread(test_image_path)  # B,G,R order

    # Skip files that are not images
    if oriImg is None:
        continue

    # Create a black canvas with the same dimensions as the original image
    canvas = np.zeros_like(oriImg)

    # Detect body pose
    candidate, subset = body_estimation(oriImg)
    canvas = util.draw_bodypose(canvas, candidate, subset, model_type)

    # Detect hands
    hands_list = util.handDetect(candidate, subset, oriImg)
    all_hand_peaks = []
    for x, y, w, is_left in hands_list:
        peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
        peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
        peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)
        all_hand_peaks.append(peaks)

    canvas = util.draw_handpose(canvas, all_hand_peaks)
    canvas = cv2.resize(canvas, (768, 1024))
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGRA2RGB)

    # Save the result
    result_name = f'{image_name.split(".")[0]}_rendered.png'
    result_path = os.path.join(output_folder, result_name)
    cv2.imwrite(result_path, canvas)

    print(f"Processed and saved: {result_path}")


# ##############################################################
#
# import cv2
# import copy
# import numpy as np
#
# from src import model
# from src import util
# from src.body import Body
# from src.hand import Hand
#
# model_type = 'body25'  # 'coco'
# if model_type == 'body25':
#     model_path = './model/pose_iter_584000.caffemodel.pt'
# else:
#     model_path = './model/body_pose_model.pth'
#
# body_estimation = Body(model_path, model_type)
# hand_estimation = Hand('model/hand_pose_model.pth')
#
# test_image_path = '08909_00.jpg'
# oriImg = cv2.imread(test_image_path)  # B,G,R order
#
# # Create a black canvas with the same dimensions as the original image
# canvas = np.zeros_like(oriImg)
#
# # Detect body pose
# candidate, subset = body_estimation(oriImg)
# canvas = util.draw_bodypose(canvas, candidate, subset, model_type)
#
# # Detect hands
# hands_list = util.handDetect(candidate, subset, oriImg)
# all_hand_peaks = []
# for x, y, w, is_left in hands_list:
#     peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
#     peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
#     peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)
#     all_hand_peaks.append(peaks)
#
# canvas = util.draw_handpose(canvas, all_hand_peaks)
# canvas = cv2.resize(canvas, (768, 1024))
# canvas = cv2.cvtColor(canvas, cv2.COLOR_BGRA2RGB)
# # Save the result
# result_name = test_image_path.split('.')[0] + '_' + 'rendered.png'
# cv2.imwrite(result_name, canvas)
###########################################################################