class LoraConfig:
    rank = 8
    alpha = 16
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'out_proj'] # 在原始模型中哪个位置
