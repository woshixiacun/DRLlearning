from AIDetector_pytorch import Detector
import imutils
import cv2
import os
import pickle

def main():

    name = 'demo'
    
    #det 是检测器实例，可以用来对帧图像进行检测
    det = Detector()
    # 创建一个视频捕获对象 cap，用于从指定路径读取视频文件,可以理解为打开视频文件，准备逐帧读取。
    cap = cv2.VideoCapture('C:/Users/Clavi/Desktop/DRLlearning/Yolov5-Deepsort-main/data.mp4')
    
    fps = int(cap.get(5))  #cap.get(5) 对应 cv2.CAP_PROP_FPS 属性
    print('fps:', fps)
    t = int(1000/fps)

    videoWriter = None  #初始化视频写入器对象为空，稍后根据视频帧大小创建。

    fr = 0
    res_bboxes = []

    while True:
        # try:
        _, im = cap.read() #从视频中读取一帧。im 是当前帧图像（numpy 数组），_ 表示不关心的返回值（布尔值，表示是否读取成功）。
        
        if im is None:
            break   #如果读到结尾（即帧为空），则跳出循环。
        # print(f'frame [{fr}], ori_shape = [{im.shape}]')
        # write_img(im, f'{fr}ori.png') 
        
        result, bboxes = det.feedCap(im)  #将当前帧送入检测器处理。feedCap() 方法会执行 AI 推理，例如检测是否是 AI 生成图像、检测物体等。
        res_bboxes.append(bboxes)

        result = result['frame']    # im.shape = [500,888,3]
        result = imutils.resize(result, height=500)  #调 整帧的显示大小，高度为 500 像素，保持宽高比不变。

        # if videoWriter is None:  #第一次处理时创建视频写入器对象
        #     fourcc = cv2.VideoWriter_fourcc(  
        #         'm', 'p', '4', 'v')  # opencv3.0   fourcc是视频编码格式（mp4v）
        # # cv2.VideoWriter() 用来创建一个输出视频文件, 第四个参数 (width, height) 是输出视频帧的宽高，单位像素。shape 返回 (高, 宽, 通道)，
        # #而 VideoWriter 要的是 (宽, 高)，所以顺序反过来写。
        #     videoWriter = cv2.VideoWriter(
        #         'result.mp4', fourcc, fps, (result.shape[1], result.shape[0]))

        # videoWriter.write(result)  #当前处理后的帧写入输出视频文件
        # cv2.imshow(name, result)  #在名为 'demo' 的窗口中显示当前帧
        # cv2.waitKey(t)  #等待 t 毫秒，相当于控制播放速度。如果设为 0 则会等待键盘输入。

        # if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
        #     # 点x退出  检测窗口是否被关闭（点击右上角 X），如果被关闭则跳出循环。
        #     break
        # # except Exception as e:
        # #     print(e)
        # #     break
        # write_img(result, f'{fr}res.png')
        # fr+=1

    append_bboxes_pkl(res_bboxes)
    
    # 释放资源
    cap.release()  #关闭视频读取
    videoWriter.release() # 关闭视频写入
    cv2.destroyAllWindows() #关闭所有显示窗口
    print("SUCCESS")

def write_img(img, img_name):
    save_dir = "C:/Users/Clavi/Desktop/DRLlearning/result_pics"
    f_path = os.path.join(save_dir, img_name)
    cv2.imwrite(f_path, img)
    
def append_bboxes_pkl(bboxes):
    file_path = "C:/Users/Clavi/Desktop/DRLlearning/result_pics/res_bboxes.pkl"
    # 'ab' 模式：文件不存在则创建，存在则追加
    with open(file_path, 'ab') as f:
        pickle.dump(bboxes, f)

# ---------------- 可选：一次性读取全部历史 ----------------
def load_all_bboxes_pkl(file_path="C:/Users/Clavi/Desktop/DRLlearning/result_pics/res_bboxes.pkl") :
    """
    把之前所有追加写入的 bboxes 按轮次全部读出来。
    返回 list[list[tuple]]，外层是轮次，内层是该轮 bboxes。
    """
    if not os.path.exists(file_path):
        return []

    all_epochs = []
    with open(file_path, 'rb') as f:
        while True:
            try:
                all_epochs.append(pickle.load(f))
            except EOFError:
                break
    return all_epochs

if __name__ == '__main__':
    
    main()