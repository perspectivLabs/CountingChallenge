import cv2

def detect_screw_nuts(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image , (3000 , 3000))
    gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
    binary_image = cv2.adaptiveThreshold(gray, 255 , cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 255, 8 )
    edges = cv2.Canny(binary_image, 60 , 150)
    all_contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    output = image.copy()

    nut_contours = []
    screw_contours = []

    for contour in all_contours:
        epsilon = 0.031 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) > 6 and cv2.isContourConvex(approx):
            nut_contours.append(contour)
        elif 1:
            screw_contours.append(contour)
    
    filtered_nut_contours = []
    for contour in nut_contours:
        area = cv2.contourArea(contour)
        if area <= 3000 and area >= 60  :
            filtered_nut_contours.append(contour)

    cv2.drawContours(output, filtered_nut_contours , -1, (255, 0, 0), 3)

    filtered_screw_contours = []
    for contour in screw_contours:
        area = cv2.contourArea(contour)
        if area <= 100000 and area >= 1400  :
            filtered_screw_contours.append(contour)

    cv2.drawContours(output, filtered_screw_contours , -1, (0, 255, 0), 3)
    cv2.imwrite(image_path.split(".")[0] + "_detected.jpg" , output)

    screws = len(filtered_screw_contours) - len(filtered_nut_contours)
    if screws < 0:
        screws = 0
    nuts = len(filtered_nut_contours)
    return screws , nuts

