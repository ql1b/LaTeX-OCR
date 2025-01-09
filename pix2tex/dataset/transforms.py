import albumentations as alb
from albumentations.pytorch import ToTensorV2

train_transform = alb.Compose(
    [
        alb.Affine(shift_limit=0, scale_limit=(-.15, 0), rotate_limit=1, interpolator_params={"interpolation": 3}, p=1),
        alb.GridDistortion(distort_limit=0.1, interpolator_params={"interpolation": 3}, p=.5),
        alb.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
        alb.GaussNoise(var_limit=(10.0, 10.0), p=.2),
        alb.RandomBrightnessContrast(.05, (-.2, 0), True, p=0.2),
        alb.ImageCompression(quality_lower=95, quality_upper=95, compression_type='jpeg', p=.3),
        alb.ToGray(always_apply=True),
        alb.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
        ToTensorV2(),
    ]
)

test_transform = alb.Compose(
    [
        alb.ToGray(always_apply=True),
        alb.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
        ToTensorV2(),
    ]
)
