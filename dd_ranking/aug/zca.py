import kornia


class ZCA_Whitening_Augmentation:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = kornia.enhance.ZCAWhitening()

    def __call__(self, images, labels):
        return self.transform(images, include_fit=True)