import cv2

# Function to extract frames
def frame_capture(path):

	# Path to video file
	vidObj = cv2.VideoCapture(path)

	# Used as counter variable
	count = 0

	# checks whether frames were extracted
	success = 1

	while success:

		# vidObj object calls read
		# function extract frames
		success, image = vidObj.read()
        
		# Saves the frames with frame-count
		cv2.imwrite("frame%d.jpg" % count, image)

		count += 1

def load_videos(path):
    


# Driver Code
if __name__ == '__main__':

	# Calling the function
	FrameCapture("/Users/motorbreath/mipt/thesis/data/casia/casia-mfsd_train_renamed/data/train/001M/C001_HR_E1_IN_TG_00D_PT+HR+1_0_1.avi")
