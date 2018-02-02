# ssd.StructuredPrediction

This branch contains code files for implementing the JaccardSegment localization loss. This is an attempt to implement the localization loss as given in [this](https://github.com/bermanmaxim/jaccardSegment) repo. The code might not work directly and need some more modifications.

## Major changes
- Modules contains ```jcloss.py``` and modifications to ```multiboxloss.py``` to incorporate modified localization loss.

## TODO
1. Make the code runnable with the loss.
2. Try to make the loss more stable.
