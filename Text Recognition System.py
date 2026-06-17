import cv2
import pytesseract

def contours_text(orig, img, contours):
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Drawing a rectangle on the copied image 
        rect = cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.imshow('cnt', rect)
        cv2.waitKey()
        # Cropping the text block for giving input to OCR 
        cropped = orig[y:y + h, x:x + w]
        # Apply OCR on the cropped image 
        config = ('-l eng --oem 1 --psm 3')
        text = pytesseract.image_to_string(cropped, config=config)
        print(text)

# Read image
im = cv2.imread('./testimg.jpg')

# Convert the image to gray scale
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# Perform some preprocessing steps to detect contours (optional and may vary based on image)
# For example, using thresholding
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# Detect contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Call the function to detect and recognize text
contours_text(im, gray, contours)

# Close all windows
cv2.destroyAllWindows()
