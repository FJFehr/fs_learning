import torch

from models.nvib_vision_transformer import NvibVisionTransformer as nvib_vit
from models.vision_transformer import VisionTransformer as vit

# Initialise the model

# Set seed
torch.manual_seed(0)
model = vit(
    img_size=[224],
    patch_size=16,
    in_chans=3,
    num_classes=1000,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4,
    qkv_bias=True,
    representation_size=None,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.0,
    norm_layer=torch.nn.LayerNorm,
    ape=False,
    patch_norm=True,
    use_checkpoint=False,
)
torch.manual_seed(0)
model_nvib = nvib_vit(
    img_size=[224],
    patch_size=16,
    in_chans=3,
    num_classes=1000,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4,
    qkv_bias=True,
    representation_size=None,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.0,
    norm_layer=torch.nn.LayerNorm,
    ape=False,
    patch_norm=True,
    use_checkpoint=False,
)

from models.nvib_vision_transformer import vit_small as nvib_vit
from models.vision_transformer import vit_small as vit

# Dino model
url_dino = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
state_dict_dino = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url_dino)

# Dino weights for small model
dino_vitsmall = vit(patch_size=16, num_classes=0)
dino_vitsmall.load_state_dict(state_dict_dino, strict=True)

# Dino weights for small model NVIB
dino_vitsmall_nvib = nvib_vit(patch_size=16, num_classes=0)
dino_vitsmall_nvib.load_state_dict(state_dict_dino, strict=True)

#Deit model
url_deit = "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth"
state_dict_deit = torch.hub.load_state_dict_from_url(url=url_deit)["model"]
for k in ['head.weight', 'head.bias']:
            if k in state_dict_deit:
                print(f"removing key {k} from pretrained checkpoint")
                del state_dict_deit[k]

# Deit weights for small model
deit_vitsmall = vit(patch_size=16, num_classes=0)
deit_vitsmall.load_state_dict(state_dict_deit, strict=True)


# Deit weights for small model NVIB
deit_vitsmall_nvib = nvib_vit(patch_size=16, num_classes=0)
deit_vitsmall_nvib.load_state_dict(state_dict_deit, strict=True)


# Test base model
def test_base_model():
    # Random input that is torch.Size([25, 3, 224, 224])
    x = torch.randn(2, 3, 224, 224)

    # Forward pass
    y = model(x)

    # Forward pass
    y_nvib = model_nvib(x)


    # check equality
    assert torch.allclose(y, y_nvib, atol=1e-4), "Models are not equal"

def test_dino_vitsmall():
    # Random input that is torch.Size([25, 3, 224, 224])
    x = torch.randn(2, 3, 224, 224)

    # Forward pass
    y = dino_vitsmall(x)

    # Forward pass
    y_nvib = dino_vitsmall_nvib(x)


    # check equality
    assert torch.allclose(y, y_nvib, atol=1e-4), "Models are not equal"

# Test deit_vitsmall
def test_deit_vitsmall():
    # Random input that is torch.Size([25, 3, 224, 224])
    x = torch.randn(2, 3, 224, 224)

    # Forward pass
    y = deit_vitsmall(x)

    # Forward pass
    y_nvib = deit_vitsmall_nvib(x)


    # check equality
    assert torch.allclose(y, y_nvib, atol=1e-4), "Models are not equal"

def main():
    test_base_model()
    test_dino_vitsmall()
    test_deit_vitsmall()

if __name__ == '__main__':
    main()