# DD-Ranking Metrics

DD-Ranking provides a set of metrics to evaluate the real informativeness of datasets distilled by different methods. The unfairness of existing evaluation is mainly caused by two factors, the label representation and the data augmentation. We design the label-robust score (LRS) and the augmentation robust score (ARS) to disentangle the impact of label representation and data augmentation on the evaluation, respectively.

## Evaluation Classes
* [LabelRobustScoreHard](lrs-hard-label.md) computes HLR, IOR, and LRS for methods using hard labels.
* [LabelRobustScoreSoft](lrs-soft-label.md) computes HLR, IOR, and LRS for methods using soft labels.
* [AugmentationRobustScore](ars.md) computes the ARS for methods using soft labels.
* [GeneralEvaluator](general.md) computes the traditional test accuracy for existing methods.
