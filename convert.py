import os
import nncase
import numpy as np
import cv2
import nncase_kpu

# 自动定位并消除插件路径未设置的警告
os.environ['NNCASE_PLUGIN_PATH'] = os.path.dirname(nncase_kpu.__file__)

# 配置参数
onnx_model_path = "best.onnx"
kmodel_path = "yolov8n.kmodel"
image_dir = "images"

# [修改 1]：严格对齐 ONNX 模型的实际尺寸 320x320
input_shape = [1, 3, 320, 320] 

def read_calibration_images(img_dir, shape):
    data_list = []
    for filename in os.listdir(img_dir):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            img_path = os.path.join(img_dir, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (shape[3], shape[2]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0) # 保持 [1, 3, 320, 320] 维度
            data_list.append(img)
            
    # [修改 2]：直接返回列表，不使用 concatenate 拼接，防止维度被错误剥离
    return data_list 

def main():
    compile_options = nncase.CompileOptions()
    compile_options.target = "k230"
    compile_options.dump_ir = False
    compile_options.dump_asm = False
    compile_options.dump_dir = "tmp"

    compiler = nncase.Compiler(compile_options)

    print("正在读取 ONNX 模型...")
    model_content = open(onnx_model_path, "rb").read()
    import_options = nncase.ImportOptions()
    compiler.import_onnx(model_content, import_options)

    print("正在进行量化校准...")
    calib_data_list = read_calibration_images(image_dir, input_shape)
    
    ptq_options = nncase.PTQTensorOptions()
    ptq_options.samples_count = len(calib_data_list)
    ptq_options.calibrate_method = "Kld"
    ptq_options.quant_type = "uint8"
    ptq_options.w_quant_type = "uint8"
    # 将列表的列表传给 set_tensor_data
    ptq_options.set_tensor_data([calib_data_list]) 
    
    compiler.use_ptq(ptq_options)

    print("正在编译 K230 专属模型...")
    compiler.compile()
    kmodel = compiler.gencode_tobytes() 
    with open(kmodel_path, "wb") as f:
        f.write(kmodel)
    print(f"转换成功！模型已保存为 {kmodel_path}")

if __name__ == "__main__":
    main()
