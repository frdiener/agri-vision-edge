class Setup:

    def configure_gpu(self):
        """
        Configure the Kaggle GPU to use just one GPU and enable memory_growth to avoid OOM errors.
        """

        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        import tensorflow as tf

        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        print("GPU config done")


    def assert_od_api_setup(self):
        """

        """
        import object_detection
        import tensorflow as tf
        import google.protobuf

        print("TF:", tf.__version__)
        print("protobuf:", google.protobuf.__version__)
        print("OD API OK")
