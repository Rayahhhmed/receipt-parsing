from PIL import Image
import cv2
import re
import pytesseract as pt
from imutils.perspective import four_point_transform
import imutils
import word_embeddings_ngram as wem

FILE_PATH = "data/"
BANNED_WORDS = ["store", "phone", "total", "'date", "time", "subtotal", \
                "tax", "change", "cash", "credit", "debit", "card", "amount", \
                "paid", "due", "balance", "abn", "gst", "tax", "taxes", "total", \
                "amount", "subtotal", "net", "receipt", "change", "cash", "credit", "items", "item"]

### Helpers ####################
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def noise_removal(image):
    gray = grayscale(image)
    blurred = cv2.GaussianBlur(gray, (5, 5,), 0)
    import numpy as np
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return blurred

def thick_font(image):
    import numpy as np
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return image


# Modified from stack overflow 
def getSkewAngle(cvImage):
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = cvImage.copy()
    blur = cv2.GaussianBlur(newImage, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=2)

    # Find all contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    for c in contours:
        rect = cv2.boundingRect(c)
        x,y,w,h = rect
        cv2.rectangle(newImage, (x, y),(x + w,y + h),(0, 255, 0), 2)

    largestContour = contours[0]
    minAreaRect = cv2.minAreaRect(largestContour)
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle


def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage

def deskew(cvImage):
    angle = getSkewAngle(cvImage)
    return rotateImage(cvImage, -1.0 * angle)

def strip_nums(s):
    return ''.join([i for i in s if not i.isdigit()])

def preprocess_image(img) :
    no_noise = noise_removal(img)
    return deskew(no_noise)

def create_contours(blurred):
    cnts = cv2.findContours(blurred.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    receipt_contour = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            receipt_contour = approx
            break
    if receipt_contour is None or len(receipt_contour) < 4:
        raise Exception(("Could not find receipt outline. "
            "Try debugging your edge detection and contour steps."))
    return receipt_contour


def post_process(raw_text):
    res = ""
    # post processing
    cleaned_text = re.sub(' +', ' ', raw_text)
    
    for row in cleaned_text.split("\n"):
        if re.search(r'(\$+)', row) is None and bool(re.search(r'\d', row)) == True:
            res += strip_nums(row) + "\n"
    print(res)
    res = [q.strip().lower() for q in res.strip().split("\n") if len(q) >= 3 and all(w not in q.lower() for w in BANNED_WORDS)]
   
    product_names = []
    for names in res:
        temp = names.split(" ")
        temp = [item_name for item_name in temp if len(item_name) > 2]
        product_names.append(" ".join(temp))
    return product_names

##########################################

    
def get_item_names_from_receipt(file_name: str):
    image_file = FILE_PATH + file_name
    orig = cv2.imread(image_file)
    if orig is None:
        raise Exception((f"File {image_file} not found. Make sure you specified the correct path"))    
    image = orig.copy()
    image = preprocess_image(image)
    image = orig.copy()
    image = imutils.resize(image, width=500)
    ratio = orig.shape[1] / float(image.shape[1])

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5,), 0)
    
    receipt_contour = create_contours(blurred)
    
    receipt = four_point_transform(orig, receipt_contour.reshape(4, 2) * ratio)
    # cv2.imwrite("temp/fixed_ocr.jpg", imutils.resize(receipt, width=500))
    raw_text = pt.image_to_string(
        cv2.cvtColor(receipt, cv2.COLOR_BGR2RGB),
        config="--psm 4")
    product_names = post_process(raw_text)
    # print(raw_text)
    print(product_names)
    
    if len(product_names) == 0:
        raise Exception("No product names found. Try taking a better picture of the receipt please!")

    mapped_words = []
    for i in range(len(product_names)):
        print("------------------")
        
        print(f"{i+1}. {product_names[i]}")
        print(wem.predict_closest_word(product_names[i]))
        mapped_words.append(wem.predict_closest_word(product_names[i]))
        print("------------------")
    
    print(mapped_words)







get_item_names_from_receipt("sus2.jpg")
