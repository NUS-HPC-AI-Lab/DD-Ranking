import kornia


class ZCAWhitening:
    def __init__(self, params: dict):
        self.transform = kornia.enhance.ZCAWhitening()

    def __call__(self, images):
        return self.transform(images, include_fit=True)