# DD-Ranking Metrics

DD-Ranking provides a set of metrics to evaluate the real informativeness of datasets distilled by different methods. The unfairness of existing evaluation is mainly caused by two factors, the label representation and the data augmentation. We design the label-robust score (LRS) and the augmentation robust score (ARS) to disentangle the impact of label representation and data augmentation on the evaluation, respectively.

## Evaluation Classes
* [HardLabelEvaluator](hard-label.md) computes HLR, IOR, and DD-Ranking score for methods using hard labels.
* [SoftLabelEvaluator](soft-label.md) computes HLR, IOR, and DD-Ranking score for methods using soft labels.
* [AugmentationRobustScore](ars.md) computes the augmentation robust score for methods using soft labels.
* [GeneralEvaluator](general.md) computes the traditional test accuracy for existing methods.
