import tensorflow as tf
import tensorflow_model_optimization as tfmot

from object_detection.core.freezable_batch_norm import FreezableBatchNorm
from tensorflow_model_optimization.python.core.quantization.keras import quantize_config, quantize_layer
from tensorflow.keras.utils import register_keras_serializable
from tensorflow_model_optimization.quantization.keras import default_8bit


@register_keras_serializable(package='CustomQuant', name='ConvWeightOnlyQuantizeConfig')
class ConvWeightOnlyQuantizeConfig(quantize_config.QuantizeConfig):
    """
    Quantize only Conv weights, not activations between Conv and BatchNorm.
    This allows the TFLite converter to fold BatchNorm into Conv.
    """
    
    def __init__(self, quant_min=-128, quant_max=127, symmetric=True, per_axis=False):
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.symmetric = symmetric
        self.per_axis = per_axis
    
    def get_quantize_layer(self, layer, quantize_layer_name):
        return quantize_layer.QuantizeLayer(
            layer,
            quantize_config=self,
            name=quantize_layer_name
        )
    
    def get_weight_quantizers(self, layer):
        return [tfmot.quantization.keras.quantizers.LastValueQuantizer(
            num_bits=8,
            symmetric=self.symmetric,
            narrow_range=False,
            per_axis=self.per_axis
        )]
    
    def get_output_quantizers(self, layer):
        return []
    
    def get_activations_and_quantizers(self, layer):
        return []
    
    def get_weights_and_quantizers(self, layer):
        weight_vars = layer.weights
        quantizer = self.get_weight_quantizers(layer)[0]
        return [(w, quantizer) for w in weight_vars]
    
    def set_quantize_activations(self, layer, activations):
        # No-op: we don't quantize activations between layers
        pass
    
    def set_quantize_weights(self, layer, weight_quantizers):
        layer.quantize_config = self
    
    def get_config(self):
        return {
            'quant_min': self.quant_min,
            'quant_max': self.quant_max,
            'symmetric': self.symmetric,
            'per_axis': self.per_axis
        }
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


def _annotate_layer(layer, quantize_config=None):
    if isinstance(
        layer,
        (
            tf.keras.layers.Conv2D,
            tf.keras.layers.DepthwiseConv2D,
        ),
    ):
        return tfmot.quantization.keras.quantize_annotate_layer(
            layer,
            quantize_config=quantize_config
        )
    
    return layer


def quantize_backbone(backbone, per_axis=False, symmetric=True, num_bits=8):
    """
    Convert a TFOD MobileNetV2 backbone into a
    TF-MOT QAT backbone while preserving weights.
    
    Only quantizes Conv weights, NOT activations between Conv and BatchNorm.
    This allows BatchNorm folding and avoids MUL/ADD in TFLite.
    
    Args:
        backbone: The original Keras backbone model
        per_axis: Whether to use per-axis quantization. Set to False for Vivante NPU.
        symmetric: Whether to use symmetric quantization
        num_bits: Number of bits for quantization (default 8)
    
    Returns:
        QAT-enabled backbone model
    """
    
    quantize_config = ConvWeightOnlyQuantizeConfig(
        per_axis=per_axis,
        symmetric=symmetric
    )
    
    # Create a wrapper function that captures the quantize_config
    def _annotate_layer_with_config(layer):
        return _annotate_layer(layer, quantize_config)
    
    with tfmot.quantization.keras.quantize_scope(
        {
            "FreezableBatchNorm": FreezableBatchNorm,
            "CustomQuant_ConvWeightOnlyQuantizeConfig": ConvWeightOnlyQuantizeConfig,
        }
    ):
        annotated = tf.keras.models.clone_model(
            backbone,
            clone_function=_annotate_layer_with_config,
        )
        
        qat_backbone = (
            tfmot.quantization.keras.quantize_apply(
                annotated
            )
        )
    
    return qat_backbone

def quantize_backbone_full(backbone):
    """
    Convert a TFOD MobileNetV2 backbone into a
    TF-MOT QAT backbone while preserving weights.
    """

    with tfmot.quantization.keras.quantize_scope(
        {
            "FreezableBatchNorm": FreezableBatchNorm,
        }
    ):
        annotated = tf.keras.models.clone_model(
            backbone,
            clone_function=_annotate_layer,
        )

        qat_backbone = (
            tfmot.quantization.keras.quantize_apply(
                annotated,
                scheme=default_8bit.Default8BitQuantizationScheme(disable_per_axis=True)
            )
        )

    return qat_backbone


def ensure_model_is_built_for_qat(
    detection_model,
    pipeline_config
):
    ssd_config = pipeline_config.model.ssd

    h = (
        ssd_config.image_resizer
        .fixed_shape_resizer
        .height
    )

    w = (
        ssd_config.image_resizer
        .fixed_shape_resizer
        .width
    )

    dummy = tf.zeros(
        [1, h, w, 3],
        dtype=tf.float32
    )

    image, shapes = detection_model.preprocess(dummy)

    detection_model.predict(
        image,
        shapes
    )
