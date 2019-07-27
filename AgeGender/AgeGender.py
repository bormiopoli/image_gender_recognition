# Import required modules
import cv2 as cv
import time
import argparse
import os


def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes


def store_elements(gender: str, confidence: float, face: bytearray) -> dict:
    """
    :param gender: a string representing the gender
    :param confidence: the accuracy of the prediction
    :param face: the cropped image in bgr encoding
    :return: a structured dictionary containing the results
    """
    return {"gender":gender, "confidence":confidence, "crop":face}




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Use this script to run age and gender recognition using OpenCV.')
    parser.add_argument('--input',
                        help='Path to input image or video file. Skip this argument to capture frames from a camera.')

    args = parser.parse_args()

    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"

    genderProto = "gender_deploy.prototxt"
    genderModel = "/home/fedex/PycharmProjects/crisalix_pipeline/learnopencv/AgeGender/Model/gender_net.caffemodel"

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    genderList = ['Male', 'Female']

    # Load network
    genderNet = cv.dnn.readNet(genderModel, genderProto)
    faceNet = cv.dnn.readNet(faceModel, faceProto)

    results = {}
    nr = 1

    for el in os.listdir("/home/fedex/PycharmProjects/crisalix_pipeline/learnopencv/AgeGender/images"):
        print("---------------------------------------------------------------------------")
        el = os.path.join("/home/fedex/PycharmProjects/crisalix_pipeline/learnopencv/AgeGender/images", el)
        # Open a video file or an image file or a camera stream
        try:
            cap = cv.VideoCapture(args.input if args.input else el)
            padding = 20
            while cv.waitKey(1) < 0:
                # Read frame
                t = time.time()
                hasFrame, frame = cap.read()
                if not hasFrame:
                    cv.waitKey()
                    break

                frameFace, bboxes = getFaceBox(faceNet, frame)
                if not bboxes:
                    print("No face Detected for {}, Checking next frame".format(el))
                    continue

                for bbox in bboxes:

                    # EXTRACT THE FACE CROP
                    face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

                    # PREPARE DATA FOR NN
                    blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                    genderNet.setInput(blob)

                    # GET PREDICTIONS
                    genderPreds = genderNet.forward()

                    gender = genderList[genderPreds[0].argmax()]

                    # PRINT RESULTS AND SAVED THEM IN A DICT TO RETURN
                    print("FILE: {2} Gender : {0}, conf = {1}".format(gender, genderPreds[0].max(), el.split(os.sep)[-1]))
                    elements = store_elements(gender, genderPreds[0].max(), face)
                    results[el.split(os.sep)[-1]] = elements
                    label = "{}".format(gender)


                print("time : {:.3f}".format(time.time() - t))

        except Exception as err:
            print(err)

