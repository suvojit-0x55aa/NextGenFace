import mediapipe as mp
import numpy as np
import torch
import cv2
import os


# Path to the bundled face_landmarker model
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "resources", "face_landmarker.task")


class LandmarksDetectorMediapipe:
	def __init__(self, mask, device, is_video=False, refine_landmarks=False):
		'''
		init landmark detector with given mask on target device
		:param mask: valid mask for the 468 landmarks of shape [n]
		:param device:
		:param is_video: set to true if passing frames sequentially in order
		:param refine_landmarks: if the facemesh attention module should be applied. Note: requires mediapipe 0.10
		'''
		assert(mask.dim() == 1)
		assert(mask.max().item() <= 467 and mask.min().item() >= 0)

		self.device = device

		if not os.path.isfile(_MODEL_PATH):
			raise FileNotFoundError(
				f"Face landmarker model not found at {_MODEL_PATH}. "
				"Download from https://storage.googleapis.com/mediapipe-models/"
				"face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
			)

		base_options = mp.tasks.BaseOptions(model_asset_path=_MODEL_PATH)
		options = mp.tasks.vision.FaceLandmarkerOptions(
			base_options=base_options,
			running_mode=mp.tasks.vision.RunningMode.IMAGE,
			num_faces=1,
			min_face_detection_confidence=0.5,
			min_face_presence_confidence=0.5,
			output_face_blendshapes=False,
		)
		self.landmarksDetector = mp.tasks.vision.FaceLandmarker.create_from_options(options)
		self.mask = mask.to(self.device)

	def detect(self, images):
		'''
		detect landmakrs on a batch of images
		:param images: tensor [n, height, width, channels]
		:return: tensor [n, landmarksNumber, 2]
		'''
		assert(images.dim() == 4)
		landmarks = []
		for i in range(len(images)):
			land = self._detect((images[i].detach().cpu().numpy() * 255.0).astype('uint8'))
			landmarks.append(land)

		torch.set_grad_enabled(True) #it turns out that the landmark detector disables the autograd engine. this line fixes this
		return torch.tensor(landmarks, device = self.device)

	def _detect(self, image):

		height, width, _ = image.shape

		# Convert to mediapipe Image format (expects RGB)
		mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
		results = self.landmarksDetector.detect(mp_image)
		mask = self.mask.detach().cpu().numpy()

		if results.face_landmarks and len(results.face_landmarks) > 0:
			face_landmarks = results.face_landmarks[0]
			landmarks = np.array(
				[(lm.x * width, lm.y * height) for lm in face_landmarks]
			)
		else:
			raise RuntimeError('No face was found in this image')

		return landmarks[mask]

	def drawLandmarks(self, image, landmarks):
		'''
		draw landmakrs on top of image (for debug)
		:param image: tensor representing the image [h, w, channels]
		:param landmarks:  tensor representing the image landmarks [n, 2]
		:return:
		'''
		assert(image.dim() == 3 and landmarks.dim() == 2 and landmarks.shape[-1] ==2)
		clone = np.copy(image.detach().cpu().numpy() * 255.0)
		land = landmarks.cpu().numpy()
		for x in land:
			cv2.circle(clone, (int(x[0]), int(x[1])), 1, (0, 0, 255), -1)
		return clone
