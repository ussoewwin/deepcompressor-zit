from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel

model = ZImageTransformer2DModel()
print(f"Default in_channels: {model.config.in_channels}")
print(f"Default patch_size: {model.config.patch_size}")

# Check what config keys determine latent channels
print(f"Config keys: {model.config.keys()}")
