# This file provides backward compatibility.
# Please use the proper visualization modules in dog_detector.visualization package instead.
import warnings

# Import the main function for backward compatibility
from dog_detector.visualization.image_utils import visualize_predictions

# Show deprecation warning
warnings.warn(
    "Importing from dog_detector.visualization is deprecated. "
    "Please use dog_detector.visualization.image_utils or "
    "dog_detector.visualization.tensorboard_logger instead.",
    DeprecationWarning,
    stacklevel=2
)

# Keep the visualize_predictions function at the top level for backward compatibility
__all__ = ['visualize_predictions']
