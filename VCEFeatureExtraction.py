import cv2
import numpy as np
from keras.applications.resnet import preprocess_input
from keras.utils import img_to_array
from PIL import Image
from VisualContentEncoder import create_visual_content_encoder


def load_and_preprocess_frame(video_frame):
    video_frame = cv2.resize(video_frame, (224, 224))
    frame_array = np.asarray(video_frame, dtype=np.float32)
    frame_array = np.expand_dims(frame_array, axis=0)
    frame_array = preprocess_input(frame_array)
    return frame_array


if __name__ == '__main__':
    # Load your ResNet50 model
    input_shape = (224, 224, 3)
    model = create_visual_content_encoder(input_shape)

    # # extract vce feature vectors from a video
    # # Load the video
    # video_path = "./gazebasevr/big buck bunny s1.mp4"
    # video_capture = cv2.VideoCapture(video_path)
    # vce_features = []
    # # Process each frame of the video
    # while video_capture.isOpened():
    #     ret, frame = video_capture.read()
    #     if not ret:
    #         break
    #     frame_array = load_and_preprocess_frame(frame)
    #     # Make predictions using the frame
    #     vce_features.append(model.predict(frame_array))
    # video_capture.release()
    # cv2.destroyAllWindows()
    # np.save('video1_feature.npy', vce_features)
    # loaded_array = np.load('video1_feature.npy')
    # print(len(loaded_array))

    # extract vce feature vector from an image
    # Load the image using Pillow
    image_path = './text visual.PNG'
    image = Image.open(image_path)
    image = image.convert('RGB')
    # Resize the image to the input size expected by the ResNet-50 model
    image = image.resize((224, 224))

    # Convert the image to a NumPy array and preprocess it
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    vce_features = []
    vce_prediction = model.predict(image_array)
    for _ in range(920):
        # Make predictions using the preprocessed image
        vce_of_cur_frame = [val for val in vce_prediction]
        vce_features.append(vce_of_cur_frame)
    np.save('text1_feature.npy', vce_features)
    loaded_array = np.load('text1_feature.npy')
    print(len(loaded_array))
