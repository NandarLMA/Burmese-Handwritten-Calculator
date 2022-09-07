import cv2 as cv
import numpy as np
from scipy.ndimage import interpolation as inter
from tensorflow.keras.models import load_model

model = load_model('model_new.h5')

def drawBoundingBox(img):
    coord_x, coord_xm, coord_y, coord_ym = [],[],[],[]
    dilated = cv.dilate(img, None, iterations = 1)
    cnts,h = cv.findContours(dilated, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    
    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    for i,c in enumerate(cnts):

        x,y,w,h = cv.boundingRect(c)
        print(w,h,i)

        if w > 20  and w < 200 and h > 20 and h < 200:
            coord_x.append(x)
            coord_xm.append(x + w)
            coord_y.append(y)
            coord_ym.append(y + h)
            cv.rectangle(img, (x-5, y-5), (x + w + 5, y + h + 5), (0,255,0), 2)

    cv.imshow("bounding box",img)
    cv.waitKey(0)
    cv.destroyAllWindows() 
    return img, coord_x, coord_xm, coord_y, coord_ym

def correct_skew(image, delta = 1, limit = 5):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape = False, order = 0)
        histogram = np.sum(data, axis = 1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return score

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        score = determine_score(image, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv.warpAffine(image, M, (w, h), flags = cv.INTER_CUBIC, borderMode = cv.BORDER_REPLICATE)

    return best_angle, rotated

def calculateResult(data_array):
    prev_symbol = 0
    result = 0
    current_value = 0
    for i in data_array:
        print(i)
        if i < 10:
            current_value *= 10 
            current_value += i
        else:
            print('Value Before symbol', current_value)
            if prev_symbol != 0:
                if prev_symbol == 10:
                    result += current_value
                elif prev_symbol == 11:
                    result -= current_value
                elif prev_symbol == 12:
                    result *= current_value
                else:
                    result /= current_value
            else:
                result = current_value
            prev_symbol = i
            current_value = 0

        print("current value ", current_value)

    print(result)
    if prev_symbol != 0:
        if prev_symbol == 10:
            result += current_value
        elif prev_symbol == 11:
            result -= current_value
        elif prev_symbol == 12:
            result *= current_value
        else:
            result /= current_value
    
    return result

def predict_digit(img):
    
    cv.imshow("cropped image :", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    img = np.array(img)
    img = img.reshape(1,28,28,1)
    img = img.astype(float)/255.0
    val = model.predict(img)
    print("predicted Value: ", val)
    return np.argmax(val)

def imageSegmentation(img):
    
    cv.imshow("input image :", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, img_thres = cv.threshold(img_gray, 190, 255, cv.THRESH_OTSU)
    denoised = cv.fastNlMeansDenoising(img_thres)

    #image = cv.bitwise_not(denoised)
    a_, deskew = correct_skew(denoised)
    img_bounded, x, xm, y, ym = drawBoundingBox(deskew)
    x_copy = list(x)
    x_copy.sort()
    data_array = []
    i = 0
    while i < len(x):
        index = x.index(x_copy[i])
        cropped_img = img_bounded[y[index] + 2 : ym[index] - 2, x[index]+2 : xm[index] -2]
        cropped_img = cv.cvtColor(cropped_img, cv.COLOR_BGR2GRAY)
        try:
            test_x = cv.resize(cropped_img, (28,28), interpolation = cv.INTER_AREA)
            #cv.imwrite("input"+str(i)+".jpg", test_x)
            val = predict_digit(test_x)
            print("value : ",val)
        except:
            print("Exception case :", i)
            continue
        data_array.append(val)
        i += 1

    print(data_array)
    return data_array

if __name__ == "__main__":
    img = cv.imread('img.jpg')
    data_array = imageSegmentation(img)
    result = calculateResult(data_array)
    print("result : ", result)


