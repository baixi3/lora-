import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
from utils import apply_lora, load_lora_weight
from LoRA.lora_config import LoraConfig

lora_dict = {
    "base": {
        "path": None,
        "weight": None,
    },
    "cute": {
        "path": "./lora_llava_finetuned/cute.bin",
        "weight": None,
    },
    "cat": {
        "path": "./lora_llava_finetuned/cat.bin",
        "weight": None,
    }
}
# 模型路径
MODEL_PATH = "../models/llava-interleave-qwen-0.5b-hf"
lora_config = LoraConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ) # 复制下来可用模型参数权重等
    model.generation_config.pad_token_id = 151645
    processor = AutoProcessor.from_pretrained(MODEL_PATH) #给模型提供的输入

    # 加入 LoRA 结构
    model = apply_lora(model, lora_config.target_modules, lora_config.rank, lora_config.alpha)
    model = model.to(device)
    # print('model:')
    # print(model)

    # 加载 LoRA 权重到内存
    for each in lora_dict.values():
        if each["path"]:
            each["weight"] = torch.load(each["path"])

    model.eval()
    
    image_path = "SU7_Ultra.png"
    raw_image = Image.open(image_path)

    while True:
        user_input = input("🤗：")
        if user_input == "exit":
            break

        if user_input == "image":
            img_filename = input("请输入图片文件名：")  
            try:
                raw_image = Image.open(img_filename)
                print(f"图片'{img_filename}'加载成功")
                continue
            except Exception as e:
                print(f"图片加载失败：{e}")

        if user_input == "lora":
            # 加载指定的 lora 权重
            lora_name = input("请输入要替换的风格：")
            if lora_name not in lora_dict:
                print(f"名为 {lora_name} 的 lora 权重不存在")
                continue

            load_lora_weight(model, lora_dict[lora_name]["weight"])
            continue

        # 构造模型输入
        conversation = [{
            "role": "user",
            "content": [
                {"type": "text", "text": user_input},
                {"type": "image"},
            ],
        }]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(
            images=raw_image,
            text=prompt,
            return_tensors='pt',
        ).to(0)
        # 模型推理
        output = model.generate(
            **inputs,
            max_new_tokens=200,
        )
        # 将输出的 token 解码为文本
        output_text = processor.decode(output[0], skip_special_tokens=True)
        # 只输出 ‘ASSISTANT’ 后面的文本
        print(f'🤖：{output_text.split("assistant")[-1].strip()}')
        print(f'🤖：{output_text}')


if __name__ == "__main__":
    main()
