import cv2
import numpy as np
from net.unetplusplus import UnetPlusPLus
from config.config import *
from scripts.loss import *
from torchvision import transforms

mode_path = '/opt/projects/unetplusplus/py/unetplusplus.pth'
# img_path = "/opt/projects/unetplusplus/cpp/cat1.jpg"
img_path = "/opt/projects/image_algorithm/src/segment/coco/images/000000000036/img/000000000036.jpg"

data_transforms = transforms.Compose([
    transforms.ToTensor()
])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UnetPlusPLus(num_classes=num_classes, filters=filters, deep_supervision=deep_supervision).to(device)
model.load_state_dict(torch.load(mode_path, map_location=device))
print("Successful load weight.")



img = cv2.imread(img_path)
img = cv2.resize(img, resize)
cv2.imshow("img0", img)
tmp = img.transpose((2,0,1))
img = np.array([img]).transpose((0,3,1,2))
img = img / 255.
img = torch.from_numpy(img).float()
img = img.to(device)
y_hat = model(img)
y_hat = y_hat.squeeze(0).detach().cpu().numpy()
y_hat[y_hat>=0.5] = 255
y_hat[y_hat<0.5] = 0
y_hat = y_hat.transpose((1,2,0))
cv2.resize(y_hat,resize)
cv2.imshow("img", y_hat)
cv2.waitKey()
cv2.destroyAllWindows()
