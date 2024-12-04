import cv2
import numpy as np


input_image_path = "images/FinalPic.jpg"
image = cv2.imread(input_image_path)
original_image = image.copy()

dot_size = 10

height, width, _ = image.shape

new_image = np.zeros_like(image)

for y in range(0, height, dot_size):
    for x in range(0, width, dot_size):
        end_y = min(y + dot_size, height)
        end_x = min(x + dot_size, width)

        block = image[y:end_y, x:end_x]

        average_color = block.mean(axis=(0, 1)).astype(int)

        new_image[y:end_y, x:end_x] = average_color

is_lightened = False
is_grayscale = False
is_color_extraction_mode = False
is_edge_mode = False
extracted_color_text = ""

cv2.namedWindow('Dot Image')

def on_trackbar(threshold):
    global display_image
    if is_edge_mode:
        blurred = cv2.GaussianBlur(new_image, (5, 5), 0)
        edges = cv2.Canny(blurred, threshold, threshold * 2)
        display_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cv2.putText(display_image, "Edge Mode", (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('Dot Image', display_image)

initial_threshold = 50
max_threshold = 150

cv2.createTrackbar('Canny th', 'Dot Image', initial_threshold, max_threshold, on_trackbar)

def on_mouse_click(event, x, y, flags, param):
    global is_color_extraction_mode, extracted_color_text
    if is_color_extraction_mode and event == cv2.EVENT_LBUTTONDOWN:  # 마우스 왼쪽 버튼이 눌렸을 때
        b, g, r = original_image[y, x]
        extracted_color_text = f"Color extraction mode: R:{r} G:{g} B:{b}"

cv2.setMouseCallback('Dot Image', on_mouse_click)

while True:
    display_image = new_image.copy()

    key = cv2.waitKey(1) & 0xFF

    if key == ord('1'):
        is_lightened = True
        is_grayscale = False
        is_color_extraction_mode = False
        is_edge_mode = False

    if key == ord('0'):
        is_lightened = False
        is_grayscale = False
        is_color_extraction_mode = False
        is_edge_mode = False

    if key == ord('2'):
        is_grayscale = True
        is_lightened = False
        is_color_extraction_mode = False
        is_edge_mode = False

    if key == ord('3'):
        is_color_extraction_mode = True
        is_lightened = False
        is_grayscale = False
        is_edge_mode = False

    if key == ord('4'):
        is_edge_mode = True
        is_lightened = False
        is_grayscale = False
        is_color_extraction_mode = False
        on_trackbar(cv2.getTrackbarPos('Canny th', 'Dot Image'))

    if is_lightened:
        display_image = cv2.addWeighted(new_image, 0.5, np.full_like(new_image, 255), 0.5, 0)
        cv2.putText(display_image, "Blur Mode", (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    if is_grayscale:
        display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2GRAY)
        display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)
        cv2.putText(display_image, "Grayscale Mode", (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    if is_color_extraction_mode and extracted_color_text:
        cv2.putText(display_image, extracted_color_text, (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    if is_edge_mode:
        on_trackbar(cv2.getTrackbarPos('Canny th', 'Dot Image'))

    if key == ord('5'):
        cv2.imwrite('output_image.jpg', display_image)
        print("Current mode image saved as 'output_image.jpg'")

    cv2.imshow('Dot Image', display_image)

    if key == 27:
        break

cv2.destroyAllWindows()
