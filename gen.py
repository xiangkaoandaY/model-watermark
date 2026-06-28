import os
import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL
from PIL import Image
from tqdm import tqdm

# ================= 配置区域 =================
# 提示词文件路径
PROMPTS_FILE = "D:\python_projects\signature\stable_signature-main\output\gen_prompt.txt"
# 输出目录
OUTPUT_DIR = "D:\python_projects\signature\stable_signature-main\output\\fid_image"

# 模型路径配置 (请替换为你本地的路径或HuggingFace ID)
# BASE_MODEL_ID = "D:\python_projects\project\signature\stable_signature-main\output\\vae_1_4"
BASE_MODEL_ID = "D:\python_projects\sig\stable-diffusion-v1-4"

# VAE 配置
# VAE A: 使用底模自带的VAE，或者指定一个路径
VAE_A_PATH = "D:\python_projects\sig\stable-diffusion-v1-4\\vae"
# VAE B: 你想要对比的第二个VAE
VAE_B_PATH = "D:\python_projects\project\signature\stable_signature-main\output\\vae_1_4"

# 生成参数
BATCH_SIZE = 1  # 显存够大可以改大，但为了文件对应简单，建议为1
HEIGHT = 512
WIDTH = 512
STEPS = 30
GUIDANCE_SCALE = 7
SEED = 42 # 固定种子以保证复现性（可选）

# ===========================================

def load_models():
    """
    加载模型架构：
    1. 加载两个不同的VAE
    2. 加载一个共享的Pipeline (Unet/TextEncoder等)
    """
    print(f"正在加载 VAE A: {VAE_A_PATH} ...")
    vae_a = AutoencoderKL.from_pretrained(VAE_A_PATH, torch_dtype=torch.float16).to("cuda")
    
    print(f"正在加载 VAE B: {VAE_B_PATH} ...")
    # vae_b = AutoencoderKL.from_pretrained(VAE_B_PATH, torch_dtype=torch.float16).to("cuda")
    
    vae_b = AutoencoderKL.from_pretrained(VAE_B_PATH, low_cpu_mem_usage=False, device_map=None).to(dtype=torch.float16).to("cuda")

    print(f"正在加载主模型: {BASE_MODEL_ID} ...")
    # 我们先用 VAE A 初始化 Pipeline，但实际上我们会手动控制解码过程
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL_ID,
        vae=vae_a, # 初始挂载 VAE A
        torch_dtype=torch.float16,
        safety_checker=None # 批量生成建议关闭安全检查以防报错中断
    ).to("cuda")
    
    # 启用显存优化
    pipe.enable_model_cpu_offload() 
    # 如果显存非常紧张，可以使用 pipe.enable_sequential_cpu_offload()
    
    return pipe, vae_a, vae_b

def decode_latents(vae, latents):
    """
    使用指定的VAE将Latents解码为图片
    """
    # 1. 缩放 Latents (Stable Diffusion 标准缩放系数)
    latents = 1 / 0.18215 * latents
    
    # 2. VAE 解码
    with torch.no_grad():
        image = vae.decode(latents).sample

    # 3. 后处理：归一化并转为 PIL 格式
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    
    # 转为 PIL Image 对象
    image_pil = pipe.numpy_to_pil(image)[0]
    return image_pil

# 初始化环境
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    os.makedirs(os.path.join(OUTPUT_DIR, "ori_img"))
    os.makedirs(os.path.join(OUTPUT_DIR, "wm_img"))

# 读取提示词
with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
    prompts = [line.strip() for line in f.readlines() if line.strip()]

prompts = prompts[4999:]
print(f"共加载了 {len(prompts)} 个提示词。")

# 加载模型
pipe, vae_a, vae_b = load_models()

# 创建生成器（如果需要固定随机性）
generator = torch.Generator(device="cuda").manual_seed(SEED)

print("开始批量生成任务...")

# ================= 主循环 =================
for i, prompt in tqdm(enumerate(prompts), total=len(prompts)):
    try:
        # 文件名前缀，例如 0001
        file_prefix = f"{5000 + i:04d}"
        
        # --- 核心步骤 1: 生成 Latents (共用步骤) ---
        # output_type="latent" 让管线在跑完 Unet 后就停止，不进行 VAE 解码
        # 这样我们得到了一份“未显影的底片”
        output = pipe(
            prompt=prompt,
            height=HEIGHT,
            width=WIDTH,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE_SCALE,
            output_type="latent", 
            generator=generator,
            return_dict=True
        )
        latents = output.images # 这里的 images 其实是 latents tensor

        # --- 核心步骤 2: 使用 VAE A 解码 ---
        image_a = decode_latents(vae_a, latents)
        save_path_a = os.path.join(OUTPUT_DIR, "ori_img", f"{file_prefix}.png")
        image_a.save(save_path_a)

        # --- 核心步骤 3: 使用 VAE B 解码 ---
        image_b = decode_latents(vae_b, latents)
        save_path_b = os.path.join(OUTPUT_DIR, "wm_img", f"{file_prefix}.png")
        image_b.save(save_path_b)
        
        # 可选：将prompt保存到文本日志，方便对照
        # with open(os.path.join(OUTPUT_DIR, "log.txt"), "a") as log:
        #     log.write(f"{file_prefix}: {prompt}\n")

    except Exception as e:
        print(f"Error generating prompt index {i}: {e}")
        continue

print("任务完成！所有图片已保存。")