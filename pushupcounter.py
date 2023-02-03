import cv2
import time
import torch
import argparse
import numpy as np
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.plots import output_to_keypoint, plot_skeleton_kpts, color_list, plot_one_box
from utils.general import non_max_suppression_kpt, strip_optimizer, xyxy2xywh
from torchvision import transforms

# 1.0 = Import libraries ==========================
from trainer import findAngle
from PIL import ImageFont, ImageDraw,  Image

@torch.no_grad()
def run(poseweights='yolov7-w6-pose.pt', source='pushup.mp4', device='cpu', curltracker=False, drawskeleton=False):
    path = source
    ext = path.split('/')[-1].split('.')[-1].strip().lower()
    if ext in ['mp4', 'webm', 'avi'] or ext not in ['mp4', 'webm', 'avi'] and ext.isnumeric():
        input_path = int(input) if path.isnumeric() else path
        device = select_device(opt.device)
        half = device.type != 'cpu'
        model = attempt_load(poseweights, map_location=device)
        _= model.eval()
        
        cap = cv2.VideoCapture(input_path)
        webcam = False
        
        if (cap.isOpened() == False):
            print('Error while trying to read video. Please check path again')
            
        fw, fh = int(cap.get(3)), int(cap.get(4))
        if ext.isnumeric():
            webcam = True
            fw, fh = 1280, 720
        vid_write_image = letterbox(cap.read()[1], (fw), stride=64, auto=True)[0]
        resize_height, resize_width = vid_write_image.shape[:2]
        out_video_name = 'output' if path.isnumeric() else f"{input_path.split('/')[-1].split('.')[0]}"
        out = cv2.VideoWriter(f'{out_video_name}_keypoint.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (resize_width, resize_height))
        if webcam:
            out = cv2.VideoWriter(f'{out_video_name}_kptsr.mp4', cv2.VideoWriter_fourcc(
                *'mp4v'), 30, (fw, fh))
            
        frame_count, total_fps = 0, 0
        
        # 2.1= Variables =======================================
        bcount = 0
        direction = 0
        
        # 2.2= Load custom font ==================================
        fontpath = 'sfpro.ttf'
        font = ImageFont.truetype(fontpath, 32)
        font1 = ImageFont.truetype(fontpath, 160)
        #======================================================
        
        while cap.isOpened:
            
            print(f'Frame {frame_count} Processing')
            ret, frame = cap.read()
            if ret:
                orig_image = frame
                
                # Preprocess Image
                image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
                if webcam:
                    image = cv2.resize(image, (fw, fh), interpolation=cv2.INTER_LINEAR)
                image = letterbox(image, (fw),stride=64, auto=True)[0]
                image = transforms.ToTensor()(image)
                image = torch.tensor(np.array([image.numpy()]))
                
                image = image.to(device)
                image = image.float()
                start_time = time.time()
                
                with torch.no_grad():
                    output, _ = model(image)
                    
                output = non_max_suppression_kpt(output, 0.5, 0.65, nc = model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
                output = output_to_keypoint(output)
                img = image[0].permute(1, 2, 0) * 255
                img = img.cpu().numpy().astype(np.uint8)
                
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                # 4.0= Pushup Tracking & Counting ==========================
                if curltracker:
                    for idx in range(output.shape[0]):
                        kpts = output[idx, 7:].T
                        # Right_arm=(5, 7, 9) left_arm=(6, 8, 10)
                        angle = findAngle(img, kpts, 5, 7, 9, draw=True)
                        percentage = np.interp(angle, (210, 290), (0, 100))
                        bar = np.interp(angle, (220, 290), (int(fh)-100, 100))
                        
                        color = (254, 118, 136)
                        # check for the pushup press
                        
                        if percentage == 100:
                            if direction == 0:
                                bcount += 0.5
                                direction = 1
                        if percentage == 0:
                            if direction == 1:
                                bcount += 0.5
                                direction = 0
                            
                        cv2.line(img, (100, 100), (100, int(fh)-100), (255, 255, 255), 30)
                        cv2.line(img, (100, int(bar)), (100, int(fh)-100), color, 30)  
                        
                        if (int(percentage)<10):
                            cv2.line(img, (155, int(bar)), (190, int(bar)), color, 40)
                        elif ((int(percentage) >= 10) and (int(percentage)<100)):
                            cv2.line(img, (155, int(bar)), (200, int(bar)), color, 40)
                        else:
                            cv2.line(img, (155, int(bar)), (210, int(bar)), color, 40)
                    
                        color1 = (254, 118, 138)       
                        im = Image.fromarray(img)
                        draw = ImageDraw.Draw(im)
                        draw.rounded_rectangle([fw-300, (fh//2)-100, fw-50, (fh//2)+100], fill=color1, radius=40)
                        draw.text(
                            (145, int(bar)-17), f'{int(percentage)}%', font=font, fill=(255, 255, 255))
                        draw.text(
                            (fw-230, (fh//2)-100), f'{int(bcount)}', font=font1, fill=(255, 255, 255))
                        img = np.array(im)
                
                # ============================================================================================================
                
                # if drawskeleton:
                #     for idx in range(output.shape[0]):
                #         plot_skeleton_kpts(img, output[idx, 7:].T, 3)
                # Display Image
                if webcam:
                    cv2.imshow("Detection", img)
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        break
                    
                else:
                    img_ = img.copy()
                    img_ = cv2.resize(
                        img_, (960, 540), interpolation=cv2.INTER_LINEAR)
                    cv2.imshow("Detection1", img_)
                    cv2.waitKey(1)
                    
                end_time = time.time()
                fps = 1 / (end_time - start_time)
                total_fps += fps
                frame_count += 1
                out.write(img)    
            else:
                break
            
        cap.release()
        cv2.destroyAllWindows()
        avg_fps = total_fps / frame_count
        print(f'Average FPS: {avg_fps:.3f}')
        

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str, help='path to video or 0 for webcam')
    parser.add_argument('--device', type=str, default='cpu', help='cpu/0,1,2,3(gpu)')
    parser.add_argument('--curltracker', type=bool, default=True, help='set true to check bicep count tracker')
    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    strip_optimizer(opt.device, opt.poseweights)
    main(opt)
            
                           