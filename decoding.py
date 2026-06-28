from PIL import Image
import torch
import torchvision.transforms as transforms

def msg2str(msg):
    return "".join([('1' if el else '0') for el in msg])

def str2msg(str):
    return [True if el=='1' else False for el in str]

msg_extractor = torch.jit.load("D:\python_projects\signature\stable_signature-main\hidden\models\dec_48b_whit.torchscript.pt").to("cuda")
transform_imnet = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

img = Image.open("D:\python_projects\signature\stable_signature-main\output\\res\wm_image\\result_20260105_123425.png")
img = transform_imnet(img).unsqueeze(0).to("cuda")
msg = msg_extractor(img) # b c h w -> b k
bool_msg = (msg>0).squeeze().cpu().numpy().tolist()
print(msg2str(bool_msg))
print("110110011111101011010011111110110111111101011101")


