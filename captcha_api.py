# -*- coding: UTF-8 -*-
"""
FastAPI验证码识别Web服务
使用ONNX模型提供高性能的验证码识别API
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import base64
import captcha_setting
from datetime import datetime
import time

# ===================== 配置 =====================
ONNX_MODEL_PATH = 'models/model.onnx'  # ONNX模型路径
USE_GPU = True  # 是否使用GPU

# ===================== FastAPI应用 =====================
app = FastAPI(
    title="验证码识别API",
    description="使用ONNX模型的高性能验证码识别服务",
    version="1.0.0"
)

# 允许跨域请求（根据需要配置）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该指定具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== 全局模型加载 =====================
class CaptchaPredictor:
    """验证码预测器"""
    
    def __init__(self, model_path: str, use_gpu: bool = True):
        """初始化预测器"""
        # 设置执行提供者
        providers = []
        if use_gpu and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        
        # 加载ONNX模型
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"✓ ONNX模型加载成功: {model_path}")
        print(f"  执行提供者: {self.session.get_providers()[0]}")
    
    def preprocess(self, image: Image.Image) -> np.ndarray:
        """
        图像预处理
        
        参数:
            image: PIL.Image对象
        
        返回:
            numpy数组 [1, 1, H, W]
        """
        # 转换为灰度图
        if image.mode != 'L':
            image = image.convert('L')
        
        # 调整大小（如果需要）
        if image.size != (captcha_setting.IMAGE_WIDTH, captcha_setting.IMAGE_HEIGHT):
            image = image.resize((captcha_setting.IMAGE_WIDTH, captcha_setting.IMAGE_HEIGHT))
        
        # 转为numpy数组并归一化
        image_array = np.array(image).astype(np.float32) / 255.0
        
        # 添加batch和channel维度: [H, W] -> [1, 1, H, W]
        image_array = image_array.reshape(1, 1, captcha_setting.IMAGE_HEIGHT, captcha_setting.IMAGE_WIDTH)
        
        return image_array
    
    def decode(self, output: np.ndarray) -> tuple:
        """
        解码模型输出为验证码文本
        
        参数:
            output: 模型输出 [1, 144]
        
        返回:
            (验证码文本, 各位置置信度列表)
        """
        result = []
        confidences = []
        
        for i in range(captcha_setting.MAX_CAPTCHA):
            start = i * captcha_setting.ALL_CHAR_SET_LEN
            end = (i + 1) * captcha_setting.ALL_CHAR_SET_LEN
            
            # 获取logits
            logits = output[0, start:end]
            
            # Softmax计算概率
            exp_logits = np.exp(logits - np.max(logits))
            softmax_probs = exp_logits / exp_logits.sum()
            
            # 获取最大概率的字符
            char_idx = np.argmax(softmax_probs)
            confidence = float(softmax_probs[char_idx])
            
            result.append(captcha_setting.ALL_CHAR_SET[char_idx])
            confidences.append(confidence)
        
        return ''.join(result), confidences
    
    def predict(self, image: Image.Image) -> dict:
        """
        预测验证码
        
        参数:
            image: PIL.Image对象
        
        返回:
            包含识别结果的字典
        """
        # 预处理
        image_array = self.preprocess(image)
        
        # 推理
        start_time = time.time()
        outputs = self.session.run([self.output_name], {self.input_name: image_array})
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # 解码
        text, confidences = self.decode(outputs[0])
        avg_confidence = float(np.mean(confidences))
        
        return {
            'captcha': text,
            'confidence': avg_confidence,
            'confidences': confidences,
            'inference_time_ms': round(inference_time, 2)
        }

# 初始化全局预测器
try:
    predictor = CaptchaPredictor(ONNX_MODEL_PATH, use_gpu=USE_GPU)
    MODEL_LOADED = True
except Exception as e:
    print(f"✗ 模型加载失败: {e}")
    predictor = None
    MODEL_LOADED = False

# ===================== 数据模型 =====================
class PredictionResponse(BaseModel):
    """预测响应模型"""
    success: bool
    captcha: Optional[str] = None
    confidence: Optional[float] = None
    confidences: Optional[list] = None
    inference_time_ms: Optional[float] = None
    message: Optional[str] = None
    timestamp: str

class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str
    model_loaded: bool
    provider: Optional[str] = None
    timestamp: str

# ===================== API路由 =====================

@app.get("/", tags=["信息"])
async def root():
    """根路径 - API信息"""
    return {
        "name": "验证码识别API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": MODEL_LOADED,
        "endpoints": {
            "predict_file": "/predict",
            "predict_base64": "/predict/base64",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["健康检查"])
async def health_check():
    """健康检查"""
    return HealthResponse(
        status="healthy" if MODEL_LOADED else "unhealthy",
        model_loaded=MODEL_LOADED,
        provider=predictor.session.get_providers()[0] if MODEL_LOADED else None,
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse, tags=["预测"])
async def predict_captcha(file: UploadFile = File(...)):
    """
    上传图片文件识别验证码
    
    参数:
        file: 验证码图片文件（支持 png, jpg, jpeg, bmp, gif）
    
    返回:
        识别结果，包括验证码文本和置信度
    
    示例:
        curl -X POST "http://localhost:8000/predict" \\
             -F "file=@captcha.png"
    """
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    # 检查文件类型
    allowed_extensions = ['png', 'jpg', 'jpeg', 'bmp', 'gif']
    file_ext = file.filename.split('.')[-1].lower()
    if file_ext not in allowed_extensions:
        return PredictionResponse(
            success=False,
            message=f"不支持的文件类型: {file_ext}。支持的类型: {', '.join(allowed_extensions)}",
            timestamp=datetime.now().isoformat()
        )
    
    try:
        # 读取图片
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # 预测
        result = predictor.predict(image)
        
        return PredictionResponse(
            success=True,
            captcha=result['captcha'],
            confidence=result['confidence'],
            confidences=result['confidences'],
            inference_time_ms=result['inference_time_ms'],
            message="识别成功",
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        return PredictionResponse(
            success=False,
            message=f"识别失败: {str(e)}",
            timestamp=datetime.now().isoformat()
        )

@app.post("/predict/base64", response_model=PredictionResponse, tags=["预测"])
async def predict_captcha_base64(image_base64: str):
    """
    使用Base64编码的图片识别验证码
    
    参数:
        image_base64: Base64编码的图片字符串
    
    返回:
        识别结果，包括验证码文本和置信度
    
    请求体示例:
        {
            "image_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
        }
    
    示例:
        curl -X POST "http://localhost:8000/predict/base64" \\
             -H "Content-Type: application/json" \\
             -d '{"image_base64": "base64_encoded_string"}'
    """
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        # 解码Base64
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        
        # 预测
        result = predictor.predict(image)
        
        return PredictionResponse(
            success=True,
            captcha=result['captcha'],
            confidence=result['confidence'],
            confidences=result['confidences'],
            inference_time_ms=result['inference_time_ms'],
            message="识别成功",
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        return PredictionResponse(
            success=False,
            message=f"识别失败: {str(e)}",
            timestamp=datetime.now().isoformat()
        )

class Base64Request(BaseModel):
    """Base64请求模型"""
    image_base64: str

@app.post("/predict/base64/json", response_model=PredictionResponse, tags=["预测"])
async def predict_captcha_base64_json(request: Base64Request):
    """
    使用JSON格式提交Base64图片识别验证码（推荐）
    
    参数:
        request: 包含image_base64字段的JSON对象
    
    返回:
        识别结果，包括验证码文本和置信度
    
    请求体示例:
        {
            "image_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
        }
    """
    return await predict_captcha_base64(request.image_base64)

# ===================== 启动事件 =====================

@app.on_event("startup")
async def startup_event():
    """应用启动时执行"""
    print("=" * 80)
    print("验证码识别API服务启动")
    print("=" * 80)
    print(f"模型路径: {ONNX_MODEL_PATH}")
    print(f"模型状态: {'已加载' if MODEL_LOADED else '未加载'}")
    if MODEL_LOADED:
        print(f"执行提供者: {predictor.session.get_providers()[0]}")
    print(f"API文档: http://localhost:8000/docs")
    print("=" * 80)

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时执行"""
    print("验证码识别API服务关闭")

# ===================== 运行服务 =====================

if __name__ == "__main__":
    import uvicorn
    
    # 运行服务
    uvicorn.run(
        app,
        host="0.0.0.0",  # 监听所有网络接口
        port=8000,       # 端口号
        log_level="info"
    )
