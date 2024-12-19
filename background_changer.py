# Importing the necessary modules
import cv2
import mediapipe as mp

# Initializing MediaPipe Selfie Segmentation 
mp_selfie_segmentation = mp.solutions.selfie_segmentation
# Using the model optimized for separating background and foreground
segmenter = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)  # 0 for optimized selfies, 1 for separating background and foreground

# Accessing the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

# Loading the background image
bg_img = cv2.imread("bg.jpg")
if bg_img is None:
    print("Error: Unable to access background image.")
    exit()

# Check if the camera is opened
while cap.isOpened():
    # Capture the current frame from the webcam
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Flipping the frame for a mirror effect
    
    # Check if frame is read correctly
    if ret == False:
        print("Unable to access webcam :(")
        break
    
    # Convert the frame from BGR to RGB, as MediaPipe requires RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame to segment foreground (person) and background
    result = segmenter.process(frame_rgb)
    
    if result.segmentation_mask is not None:
        # The segmentation mask separates the foreground (person) and background
        mask = result.segmentation_mask
        
        # Resize the background image to match the webcam frame size
        bg_resized = cv2.resize(bg_img, (frame.shape[1], frame.shape[0]))

        # Threshold the segmentation mask to create a binary mask:
        # Pixels with values > 0.5 are considered as foreground (person), and <= 0.5 as background
        mask = (mask > 0.5).astype('uint8')  # Convert the mask to uint8 format (0 for background, 255 for foreground)
        
        # Create an inverted mask to isolate the background (background = white, person = black)
        mask_inv = cv2.bitwise_not(mask * 255)
        
        # Extract the person (foreground) using the mask
        person = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Extract the background using the inverted mask
        background_convert = cv2.bitwise_and(bg_resized, bg_resized, mask=mask_inv)
        
        # Combine the foreground and background to create the final frame
        converted = cv2.add(person, background_convert)
        
        # Display the final frame with the custom background
        cv2.imshow('Custom Background Media', converted)

    # Display the original webcam feed (remove this line if you only need the processed frame)
    cv2.imshow('My Webcam', frame)
    
    # Exit the loop when the 'Esc' key is pressed
    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        cap.release()
        break
