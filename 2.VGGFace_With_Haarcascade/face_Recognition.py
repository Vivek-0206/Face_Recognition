"""All function of face recognition."""

# Load model
def create_model():
    """
    Load Model with keras load_model function.
    """
    from tensorflow.keras.models import load_model

    model = load_model("model.h5")
    print("Model Loaded...")

    return model


# Preprocess_image
def preprocess_image(image_path):
    """Loads image from path and resizes it"""
    import tensorflow as tf
    import numpy as np

    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.imagenet_utils.preprocess_input(img)

    return img


# FindSimilarity
def findCosineSimilarity(source_representation, test_representation):
    """Find CosinesSimilarity"""
    import numpy as np

    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))

    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


# Predict using Webcam
def predict_persion(model):
    """Predict on webcam and return name of detected person if it's already known """
    import os
    import cv2
    import tensorflow as tf
    import numpy as np
    
    found = 0
    final_name = ""

    people_pictures = "./group_of_faces/"
    all_people_faces = dict()
    for file in os.listdir(people_pictures):
        person_face = file.split(".")[0]
        all_people_faces[person_face] = model.predict(
            preprocess_image("./group_of_faces/%s.jpg" % (person_face))
        )[0, :]

    print("Face representations retrieved successfully")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    print("Start Recogintion.....")

    face_detector = cv2.CascadeClassifier(
        "Haarcascades/haarcascade_frontalface_default.xml"
    )

    while True:
        ret, img = cap.read()
        faces = face_detector.detectMultiScale(img, 1.3, 5)

        for (x, y, w, h) in faces:
            if w > 100:  # Adjust accordingly if your webcam resoluation is higher

                cv2.rectangle(
                    img, (x, y), (x + w, y + h), (255, 0, 0), 2
                )  # draw rectangle to main image
                # preprocess_image
                detected_face = img[
                    int(y) : int(y + h), int(x) : int(x + w)
                ]  # crop detected face
                detected_face = cv2.resize(
                    detected_face, (224, 224)
                )  # resize to 224x224
                img_pixels = tf.keras.preprocessing.image.img_to_array(detected_face)
                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_pixels /= 255

                # Predict on Webcam
                captured_representation = model.predict(img_pixels)[0, :]
                for i in all_people_faces:
                    person_name = i
                    representation = all_people_faces[i]
                    similarity = findCosineSimilarity(
                        representation, captured_representation
                    )
                    if similarity < 0.30:
                        final_name = person_name[5:]
                        cv2.putText(
                            img,
                            person_name[5:],
                            (int(x + w + 15), int(y - 12)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2,
                        )
                        found = 1
                        break

                # connect face and text
                cv2.line(
                    img, (int((x + x + w) / 2), y + 15), (x + w, y - 20), (255, 0, 0), 1
                )
                cv2.line(img, (x + w, y - 20), (x + w + 10, y - 20), (255, 0, 0), 1)

                if found == 0:  # if found image is not in our people database
                    cv2.putText(
                        img,
                        "unknown",
                        (int(x + w + 15), int(y - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        2,
                    )
                    final_name = "unknown"

        cv2.imshow("img", img)

        if "vivek" in final_name:
            print("welcome vivek")
            break
        if "smit" in final_name:
            print("welcome smit")
            break
        if cv2.waitKey(1) == 13:  # 13 is the Enter Key
            break

    # Release cam and close all windows
    cap.release()
    cv2.destroyAllWindows()

    return final_name
