# Modify Diffusers Library

Step 1:
Modify Diffusers source code `diffusers/src/diffusers/pipelines/stable_diffusion/__init__.py` to import the customized pipelines.
```python
### Original code
try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_torch_and_transformers_objects import *  # noqa F403
else:
    ### ……
    from .pipeline_stable_diffusion_img2img import StableDiffusionImg2ImgPipeline
    from .pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipeline
    ### ……

### Modified code
try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_torch_and_transformers_objects import *  # noqa F403
else:
    ### ……
    from .pipeline_stable_diffusion_img2img import StableDiffusionImg2ImgPipeline
    from .pipeline_stable_diffusion_gen_latents import StableDiffusionGenLatentsPipeline
    from .pipeline_stable_diffusion_latents2img import StableDiffusionLatents2ImgPipeline
    from .pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipeline
    ### ……
```

```python
### Original code
else:
    _import_structure["pipelines"].extend(
        [
            ### ……
            "StableDiffusionImg2ImgPipeline",
            "StableDiffusionInpaintPipeline",
            ### ……
        ]
    )

### Modified code
else:
    _import_structure["pipelines"].extend(
        [
            ### ……
            "StableDiffusionImg2ImgPipeline",
            "StableDiffusionGenLatentsPipeline",
            "StableDiffusionLatents2ImgPipeline",
            "StableDiffusionInpaintPipeline",
            ### ……
        ]
    )
```

Step 2:
Modify Diffusers source code `diffusers/src/diffusers/pipelines/__init__.py` to import the customized pipelines.
```python
### Original code
    from .stable_diffusion import (
        ### ……
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipeline,
        ### ……
    )

### Modified code
    from .stable_diffusion import (
        ### ……
        StableDiffusionImg2ImgPipeline,
        StableDiffusionGenLatentsPipeline,
        StableDiffusionLatents2ImgPipeline,
        StableDiffusionInpaintPipeline,
        ### ……
    )
```

```python
### Original code
_import_structure["stable_diffusion"].extend(
        [   ### ……
            "StableDiffusionImg2ImgPipeline",
            "StableDiffusionInpaintPipeline",
            ### ……
        ]
    )

### Modified code
_import_structure["stable_diffusion"].extend(
        [   ### ……
            "StableDiffusionImg2ImgPipeline",
            "StableDiffusionGenLatentsPipeline",
            "StableDiffusionLatents2ImgPipeline",
            "StableDiffusionInpaintPipeline",
            ### ……
        ]
    )
```


Step 3:
Modify Diffusers source code `diffusers/src/diffusers/__init__.py` to import the customized pipelines.
```python
### Original code
try:
    if not (is_torch_available() and is_transformers_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_torch_and_transformers_objects import *  # noqa F403
else:
    from .pipelines import (
        ### ……
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipeline,
        ### ……

    )

### Modified code
try:
    if not (is_torch_available() and is_transformers_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_torch_and_transformers_objects import *  # noqa F403
else:
    from .pipelines import (
        ### ……
        StableDiffusionImg2ImgPipeline,
        StableDiffusionGenLatentsPipeline,
        StableDiffusionLatents2ImgPipeline,
        StableDiffusionInpaintPipeline,
        ### ……
    )
```

```python
### Original code
try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_torch_and_transformers_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    ### ……
    _import_structure["pipeline_stable_diffusion_img2img"] = ["StableDiffusionImg2ImgPipeline"]
    _import_structure["pipeline_stable_diffusion_inpaint"] = ["StableDiffusionInpaintPipeline"]
    ### ……

### Modified code
try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_torch_and_transformers_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    ### ……
    _import_structure["pipeline_stable_diffusion_img2img"] = ["StableDiffusionImg2ImgPipeline"]
    _import_structure["pipeline_stable_diffusion_gen_latents"] = ["StableDiffusionGenLatentsPipeline"]
    _import_structure["pipeline_stable_diffusion_latents2img"] = ["StableDiffusionLatents2ImgPipeline"]
    _import_structure["pipeline_stable_diffusion_inpaint"] = ["StableDiffusionInpaintPipeline"]
    ### ……
```