import torch
import timm

model = timm.create_model('vit_base_patch16_224', pretrained=True)

backend = "fbgemm"
model.qconfig = torch.quantization.get_default_qconfig(backend)
torch.backends.quantized.engine = backend

quantized_model = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear,torch.nn.Conv2d}, dtype=torch.qint8)

scripted_quantized_model = torch.jit.script(quantized_model)
scripted_quantized_model.save("vit_b_16_quantized.pt")
